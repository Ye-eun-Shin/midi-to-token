"""Slakh2100 MIDI Dataset for token extraction.

This dataset loads MIDI files from Slakh2100 and provides access to individual stems.
"""

import os
import sys
from typing import Dict, Literal
from pathlib import Path
import logging
import yaml

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# Add stream-music-gen to path for constants import
sys.path.insert(0, '/workspace/stream-music-gen')
from stream_music_gen.constants import (
    midi_prog_to_inst_class_id,
    inst_name_to_inst_class_id,
)


class Slakh2100MIDI(Dataset):
    """
    Slakh2100 MIDI Dataset for extracting note event tokens.
    Loads MIDI files instead of audio files.
    """

    def __init__(
        self,
        root_dir: str = "~/slakh2100",
        split: str = "train",
        regenerate_metadata: bool = False,
    ) -> None:
        """
        Initialize the Slakh2100MIDI dataset.

        Args:
            root_dir (str or Path): Root directory where the dataset is stored.
            split (str): The dataset split to use ('train', 'test', or 'validation').
            regenerate_metadata (bool): If True, regenerates metadata from scratch.
        """
        self.root_dir = (
            Path(root_dir) if isinstance(root_dir, str) else root_dir
        )
        self.split = split
        self.base_dir = (
            self.root_dir / "original" / "slakh2100_redux_16k"
        )

        if self.split not in ["train", "test", "validation"]:
            raise ValueError(
                "`split` must be one of ['train', 'test', 'validation']."
            )

        split_dir = (self.base_dir / self.split).expanduser()
        if not split_dir.exists() or not any(split_dir.iterdir()):
            raise RuntimeError(
                f"Dataset split {self.split} not found at {split_dir}. "
                "Please ensure the dataset is downloaded."
            )
        
        logging.info(
            f"Found dataset split {self.split} at {split_dir}."
        )

        self.all_metadata = self._load_metadata(regenerate_metadata)

    def _load_metadata(self, regenerate_metadata: bool = False) -> pd.DataFrame:
        """Load or generate metadata for MIDI files in the dataset."""
        original_dir = (self.base_dir / self.split).expanduser()

        metadata_path = (
            original_dir / f"midi_dataset_metadata_{self.split}.parquet"
        )
        if metadata_path.exists() and not regenerate_metadata:
            return pd.read_parquet(metadata_path)

        # Remove ._* files from macOS
        tracks = list(original_dir.glob("[!._]*/"))

        all_metadata = []
        for track in tqdm(tracks, desc=f"Loading MIDI metadata for {self.split}"):
            midi_dir = track / "MIDI"
            if not midi_dir.exists():
                continue
                
            midi_paths = list(midi_dir.glob("S*.mid"))
            yaml_text = (track / "metadata.yaml").read_text()
            # If the first character is not a valid yaml character, remove it
            if yaml_text[0] not in ["{", "-"]:
                yaml_text = yaml_text[1:]
            metadata = yaml.safe_load(yaml_text)
            track_id = metadata["audio_dir"].split("/")[0]
            
            for midi_path in midi_paths:
                stem_id = midi_path.stem  # e.g., "S00"
                if stem_id not in metadata["stems"]:
                    continue
                    
                stem_metadata = metadata["stems"][stem_id]
                if stem_metadata["is_drum"]:
                    instrument_class_id = inst_name_to_inst_class_id("Drums")
                else:
                    instrument_class_id = midi_prog_to_inst_class_id(
                        stem_metadata["program_num"] + 1
                    )
                
                all_metadata.append(
                    {
                        "track_name": track_id,
                        "midi_path": str(midi_path.relative_to(self.base_dir)),
                        "stem_id": stem_id,
                        "instrument_name": stem_metadata["inst_class"],
                        "program_num": stem_metadata["program_num"],
                        # program_num is 0-indexed
                        "instrument_class_id": instrument_class_id,
                        "is_drum": stem_metadata["is_drum"],
                    }
                )

        all_metadata = pd.DataFrame(all_metadata)
        if metadata_path.exists():
            os.remove(metadata_path)
        all_metadata.to_parquet(metadata_path)
        print(f"Saved MIDI metadata to {metadata_path}")
        return all_metadata

    def __len__(self) -> int:
        return len(self.all_metadata)

    def __getitem__(self, idx) -> Dict[str, str]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dict containing:
                - midi_path: Full path to the MIDI file
                - base_path: Relative path from base_dir
                - track_name: Track identifier
                - stem_id: Stem identifier (e.g., "S00")
                - instrument_name: Name of the instrument
                - program_num: MIDI program number
                - instrument_class_id: Instrument class ID
                - is_drum: Whether this is a drum track
        """
        row = self.all_metadata.iloc[idx]
        file_path = str(self.base_dir / row["midi_path"])
        base_path = str(Path(file_path).relative_to(self.base_dir))
        
        item = {
            "midi_path": file_path,
            "base_path": base_path,
            "track_name": row["track_name"],
            "stem_id": row["stem_id"],
            "instrument_name": row["instrument_name"],
            "program_num": row["program_num"],
            "instrument_class_id": row["instrument_class_id"],
            "is_drum": row["is_drum"],
        }
        
        return item


if __name__ == "__main__":
    # Test the dataset
    dataset = Slakh2100MIDI(
        root_dir="/workspace/stream-music-gen/data/slakh2100/slakh2100",
        split="train",
        regenerate_metadata=True,
    )
    print(f"Total MIDI files: {len(dataset)}")
    print(f"First item: {dataset[0]}")
