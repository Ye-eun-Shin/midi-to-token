"""Dump MIDI tokens in mixdown format for training.

This script creates training examples by combining multiple MIDI stems,
similar to how dump_audio_mixdown.py creates audio mixdown examples.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
import random

# Token dataset parameters
DEFAULT_DURATION_FRAMES = 500  # 10 seconds at 50 FPS


def load_midi_token_dataloader(
    batch_size: int,
    num_workers: int,
    split: str,
    dataset_name: str,
    data_base_dir: str,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
):
    """
    Create a dataloader for MIDI tokens grouped by track.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        split: Dataset split ('train', 'validation', or 'test')
        dataset_name: Name of the dataset (e.g., 'slakh2100')
        data_base_dir: Base directory containing token data
        duration_frames: Number of frames per example
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import Dataset, DataLoader
    
    class MIDITokenDataset(Dataset):
        def __init__(self, metadata_path, data_base_dir, duration_frames):
            self.metadata = pd.read_parquet(metadata_path)
            self.data_base_dir = Path(data_base_dir)
            self.duration_frames = duration_frames
            
            # Group by track
            self.tracks = self.metadata.groupby("track_name").agg(list).reset_index()
            
        def __len__(self):
            return len(self.tracks)
        
        def __getitem__(self, idx):
            track = self.tracks.iloc[idx]
            track_name = track["track_name"]
            token_paths = track["midi_token_path"]
            stem_ids = track["stem_id"]
            instrument_names = track["instrument_name"]
            
            # Load all tokens for this track
            tokens_list = []
            valid_indices = []
            for i, token_path in enumerate(token_paths):
                full_path = self.data_base_dir / token_path
                if not full_path.exists():
                    continue
                try:
                    tokens = torch.load(full_path)
                    tokens_list.append(tokens)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue
            
            if len(tokens_list) == 0:
                # Return empty batch
                return {
                    "track_name": track_name,
                    "num_stems": 0,
                    "input_tokens": torch.zeros(1, self.duration_frames, 4),
                    "target_tokens": torch.zeros(1, self.duration_frames, 4),
                    "input_stem_ids": [],
                    "target_stem_ids": [],
                    "input_token_paths": [],
                    "target_token_paths": [],
                }
            
            # Randomly select input and target stems
            num_stems = len(tokens_list)
            if num_stems == 1:
                # Only one stem available
                input_indices = [0]
                target_indices = [0]
            else:
                # Randomly split stems into input and target
                all_indices = list(range(num_stems))
                random.shuffle(all_indices)
                split_point = random.randint(1, num_stems - 1)
                input_indices = all_indices[:split_point]
                target_indices = all_indices[split_point:]
            
            # Stack tokens (shape: [num_stems, time, num_heads])
            all_tokens = torch.stack(tokens_list, dim=0)
            
            # Get minimum length across all stems
            min_length = min(tokens.shape[0] for tokens in tokens_list)
            
            # Random crop if longer than duration_frames
            if min_length > self.duration_frames:
                start_frame = random.randint(0, min_length - self.duration_frames)
                end_frame = start_frame + self.duration_frames
                all_tokens = all_tokens[:, start_frame:end_frame, :]
            else:
                # Pad if shorter
                if min_length < self.duration_frames:
                    padding = torch.zeros(
                        num_stems, 
                        self.duration_frames - min_length, 
                        4,
                        dtype=all_tokens.dtype
                    )
                    all_tokens = torch.cat([all_tokens[:, :min_length, :], padding], dim=1)
            
            # Select input and target tokens
            input_tokens = all_tokens[input_indices]
            target_tokens = all_tokens[target_indices]
            
            # Prepare metadata
            input_stem_ids = [stem_ids[valid_indices[i]] for i in input_indices]
            target_stem_ids = [stem_ids[valid_indices[i]] for i in target_indices]
            input_token_paths = [token_paths[valid_indices[i]] for i in input_indices]
            target_token_paths = [token_paths[valid_indices[i]] for i in target_indices]
            
            return {
                "track_name": track_name,
                "num_stems": num_stems,
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "input_stem_ids": input_stem_ids,
                "target_stem_ids": target_stem_ids,
                "input_token_paths": input_token_paths,
                "target_token_paths": target_token_paths,
            }
    
    metadata_path = os.path.join(data_base_dir, dataset_name, f"{split}_metadata.parquet")
    dataset = MIDITokenDataset(metadata_path, data_base_dir, duration_frames)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if len(x) == 1 else x,  # Simple collate
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["slakh2100"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "validation", "test"],
        help="Split name",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        required=True,
        help="Maximum number of examples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./midi_mixdown",
        help="Output directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index",
    )
    parser.add_argument(
        "--duration_frames",
        type=int,
        default=DEFAULT_DURATION_FRAMES,
        help="Duration in frames (default: 500 frames = 10 seconds at 50 FPS)",
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="/workspace/toward-realtime/amt/data/midi_tokens_50hz",
        help="Base directory containing the MIDI token data",
    )
    args = parser.parse_args()

    batch_size = 1  # Process one track at a time
    num_example = args.start_index
    pbar = tqdm(
        total=args.max_examples,
        desc=f"Dumping MIDI tokens for {args.dataset}, {args.split}",
        initial=args.start_index,
    )
    
    dataloader = load_midi_token_dataloader(
        batch_size=batch_size,
        num_workers=args.num_workers,
        split=args.split,
        dataset_name=args.dataset,
        data_base_dir=args.data_base_dir,
        duration_frames=args.duration_frames,
    )

    while True:
        for batch in dataloader:
            if batch["num_stems"] == 0:
                continue
                
            output_path = (
                Path(args.output_dir)
                / args.dataset
                / args.split
                / f"{num_example:07d}"
            )
            os.makedirs(output_path, exist_ok=True)
            
            # Save input and target tokens
            torch.save(batch["input_tokens"], output_path / "input_tokens.pt")
            torch.save(batch["target_tokens"], output_path / "target_tokens.pt")

            # Select only the metadata we need
            metadata = {
                "track_name": batch["track_name"],
                "num_stems": batch["num_stems"],
                "input_stem_ids": batch["input_stem_ids"],
                "target_stem_ids": batch["target_stem_ids"],
                "input_token_paths": batch["input_token_paths"],
                "target_token_paths": batch["target_token_paths"],
            }

            with open(
                output_path / "metadata.json", "w", encoding="utf-8"
            ) as f:
                json.dump(metadata, f, indent=2)

            num_example += 1
            pbar.update(1)
            if num_example >= args.max_examples:
                print(f"\n[info] Completed dumping {num_example} examples")
                return


if __name__ == "__main__":
    main()
