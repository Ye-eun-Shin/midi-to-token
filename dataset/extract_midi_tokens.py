"""Extract MIDI note event tokens from Slakh2100 dataset.

This script processes MIDI files and converts them to YourMT3 tokens,
similar to how extract_causal_dac_32k.py extracts DAC codes from audio.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.tokenizer import NoteEventTokenizer
from utils.note_event_dataclasses import Event
from midi_dataset import Slakh2100MIDI


# Dataset configuration
DATASET_CONFIG = {
    "slakh2100": {
        "class": Slakh2100MIDI,
        "splits": ["train", "validation", "test"],
    },
}

# Token extraction parameters
DEFAULT_FPS = 50  # 50 frames per second (20ms per frame)
DEFAULT_NUM_HEADS = 4  # Number of concurrent events per frame
DEFAULT_FIXED_LENGTH_SEC = None  # None = use original MIDI length


def process_midi_to_tokens(
    midi_path: str,
    tokenizer: NoteEventTokenizer,
    fps: int = DEFAULT_FPS,
    num_heads: int = DEFAULT_NUM_HEADS,
    fixed_length_sec: float = None,
) -> np.ndarray:
    """
    Convert a MIDI file to YourMT3 token array.
    
    Args:
        midi_path: Path to the MIDI file
        tokenizer: YourMT3 tokenizer instance
        fps: Frames per second (default: 50, i.e., 20ms per frame)
        num_heads: Number of concurrent events per frame (default: 4)
        fixed_length_sec: Fixed length in seconds, or None for original length
        
    Returns:
        Token array of shape [Time_Steps, num_heads]
    """
    import pretty_midi
    from collections import defaultdict
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Collect notes by frame
    frame_dict = defaultdict(list)
    max_frame = 0
    
    # Iterate through all instruments and notes
    for inst in midi_data.instruments:
        is_drum = inst.is_drum
        for note in inst.notes:
            # Convert time (seconds) to frame index
            frame_idx = int(note.start * fps)
            
            frame_dict[frame_idx].append({
                'pitch': note.pitch,
                'is_drum': is_drum
            })
            max_frame = max(max_frame, frame_idx)
    
    # Get PAD token ID from tokenizer
    PAD_TOKEN_ID = tokenizer.pad_id
    
    # Determine total number of frames
    if fixed_length_sec is not None:
        total_frames = int(fixed_length_sec * fps)
    else:
        total_frames = max_frame + 1
    
    # Initialize token array with PAD tokens
    token_array = np.full((total_frames, num_heads), PAD_TOKEN_ID, dtype=np.int32)
    
    # Convert each frame's events to tokens
    for frame_idx, events in frame_dict.items():
        if frame_idx >= total_frames:
            break
            
        # Sort events: non-drum first, then by pitch (ascending)
        sorted_events = sorted(events, key=lambda x: (x['is_drum'], x['pitch']))
        
        # Truncate to num_heads
        truncated_events = sorted_events[:num_heads]
        
        for head_idx, event_data in enumerate(truncated_events):
            # Create YourMT3 Event object
            event_type = 'drum' if event_data['is_drum'] else 'pitch'
            mt3_event = Event(type=event_type, value=event_data['pitch'])
            
            # Encode to token ID
            token_id = tokenizer.codec.encode_event(mt3_event)
            
            # Store in array
            token_array[frame_idx, head_idx] = token_id
    
    return token_array


def create_metadata(metadata, base_dir, output_dir):
    """Create metadata with token paths."""
    metadata = metadata.copy()
    metadata["midi_token_path"] = metadata["midi_path"].apply(
        lambda x: str(
            Path(x.replace(str(base_dir), str(output_dir))).with_suffix(".pt")
        )
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MIDI note event tokens from Slakh2100 dataset"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        default=["all"],
        help="Dataset names list, or use 'all' to process all datasets",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/workspace/stream-music-gen/data/slakh2100/slakh2100",
        help="Root directory path for datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/toward-realtime/amt/data/midi_tokens_50hz",
        help="Output directory path for MIDI tokens",
    )
    parser.add_argument(
        "--generate_metadata_only",
        action="store_true",
        help="Only generate metadata without extracting tokens",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second for tokenization (default: 50)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help="Number of concurrent events per frame (default: 4)",
    )
    parser.add_argument(
        "--fixed_length_sec",
        type=float,
        default=None,
        help="Fixed length in seconds for all MIDI files (default: None, use original length)",
    )
    parser.add_argument(
        "--base_codec",
        type=str,
        default="mt3",
        help="Base codec for tokenization (default: mt3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize YourMT3 tokenizer
    print(f"[info] Initializing tokenizer with base_codec='{args.base_codec}'")
    tokenizer = NoteEventTokenizer(
        base_codec=args.base_codec,
    )
    print(f"[info] Tokenizer initialized. Vocabulary size: {tokenizer.num_tokens}")

    # Determine which datasets to process
    if "all" in args.datasets:
        datasets_to_process = list(DATASET_CONFIG.keys())
    else:
        datasets_to_process = args.datasets

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in datasets_to_process:
        dataset_config = DATASET_CONFIG[dataset_name]
        dataset_class = dataset_config["class"]
        splits = dataset_config["splits"]
        dataset_dir = Path(args.dataset_root)

        for split in splits:
            print(f"\n[info] Processing {dataset_name}, split: {split}")
            
            # Initialize dataset
            dataset = dataset_class(
                root_dir=str(dataset_dir),
                split=split,
                regenerate_metadata=False,
            )

            dataset_metadata = dataset.all_metadata
            token_metadata = create_metadata(
                dataset_metadata,
                dataset.base_dir,
                Path(args.output_dir) / dataset_name,
            )

            # Save metadata
            os.makedirs(Path(args.output_dir) / dataset_name, exist_ok=True)
            metadata_output_path = (
                Path(args.output_dir)
                / dataset_name
                / f"{split}_metadata.parquet"
            )
            if os.path.exists(metadata_output_path):
                os.remove(metadata_output_path)
            token_metadata.to_parquet(metadata_output_path)
            print(f"[info] Saved metadata to {metadata_output_path}")

            if not args.generate_metadata_only:
                # Extract tokens
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                
                for batch in tqdm(
                    dataloader,
                    desc=f"Extracting MIDI tokens for {dataset_name}, {split}",
                    total=len(dataset),
                ):
                    base_path = batch["base_path"][0]
                    output_path = (
                        Path(args.output_dir) / dataset_name / base_path
                    )
                    output_path = output_path.with_suffix(".pt")
                    
                    # Skip if already exists
                    if output_path.exists():
                        continue
                    
                    midi_path = batch["midi_path"][0]
                    
                    try:
                        # Extract tokens
                        tokens = process_midi_to_tokens(
                            midi_path=midi_path,
                            tokenizer=tokenizer,
                            fps=args.fps,
                            num_heads=args.num_heads,
                            fixed_length_sec=args.fixed_length_sec,
                        )
                        
                        # Save tokens
                        os.makedirs(output_path.parent, exist_ok=True)
                        torch.save(torch.from_numpy(tokens), output_path)
                        
                    except Exception as e:
                        print(f"\n[error] Failed to process {midi_path}: {e}")
                        continue

            print(f"[info] Completed {dataset_name}, {split}")

    print("\n[info] All processing completed!")
