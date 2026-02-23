# MIDI Token Extraction Pipeline

This directory contains code for extracting MIDI note event tokens from the Slakh2100 dataset for training music generation models. The pipeline is designed to mirror the audio preprocessing pipeline in `stream-music-gen`.

## Pipeline Overview

The preprocessing consists of 2 main steps:

1. **Extract MIDI Tokens** (`extract_midi_tokens.py`) - Converts MIDI files to YourMT3 token arrays
2. **Dump MIDI Mixdown** (`dump_midi_mixdown.py`) - Creates training examples by grouping stems

## File Structure

```
dataset/
├── midi_dataset.py          # Slakh2100 MIDI dataset class
├── extract_midi_tokens.py   # MIDI → Token extraction
├── dump_midi_mixdown.py     # Create training examples
└── README.md                # This file
```

## Usage

### Step 1: Extract MIDI Tokens

This step processes all MIDI files and converts them to YourMT3 token arrays.

```bash
cd /workspace/toward-realtime/amt/src/dataset

python extract_midi_tokens.py \
    --datasets slakh2100 \
    --dataset_root /workspace/stream-music-gen/data/slakh2100/slakh2100 \
    --output_dir /workspace/toward-realtime/amt/data/midi_tokens_50hz \
    --fps 50 \
    --num_heads 4 \
    --num_workers 8
```

**Parameters:**
- `--datasets`: Dataset name(s) or 'all' (currently supports: slakh2100)
- `--dataset_root`: Root directory of the dataset
- `--output_dir`: Where to save extracted tokens
- `--fps`: Frames per second (default: 50, i.e., 20ms per frame)
- `--num_heads`: Number of concurrent events per frame (default: 4)
- `--fixed_length_sec`: Fixed length in seconds (optional, default: None = original length)
- `--base_codec`: Tokenizer codec (default: 'mt3')
- `--num_workers`: Number of parallel workers
- `--generate_metadata_only`: Only generate metadata without extraction

**Output Structure:**
```
midi_tokens_50hz/
└── slakh2100/
    ├── train_metadata.parquet
    ├── validation_metadata.parquet
    ├── test_metadata.parquet
    └── train/
        ├── Track00001/
        │   └── MIDI/
        │       ├── S00.pt  # Token array [time_steps, 4]
        │       ├── S01.pt
        │       └── ...
        └── ...
```

### Step 2: Dump MIDI Mixdown (Train/Valid/Test)

This step creates training examples by combining multiple stems from each track.

```bash
# Training set
python dump_midi_mixdown.py \
    --dataset slakh2100 \
    --split train \
    --max_examples 10000 \
    --output_dir /workspace/toward-realtime/amt/data/midi_mixdown \
    --data_base_dir /workspace/toward-realtime/amt/data/midi_tokens_50hz \
    --duration_frames 500 \
    --num_workers 8

# Validation set
python dump_midi_mixdown.py \
    --dataset slakh2100 \
    --split validation \
    --max_examples 1000 \
    --output_dir /workspace/toward-realtime/amt/data/midi_mixdown \
    --data_base_dir /workspace/toward-realtime/amt/data/midi_tokens_50hz \
    --duration_frames 500 \
    --num_workers 8

# Test set
python dump_midi_mixdown.py \
    --dataset slakh2100 \
    --split test \
    --max_examples 1000 \
    --output_dir /workspace/toward-realtime/amt/data/midi_mixdown \
    --data_base_dir /workspace/toward-realtime/amt/data/midi_tokens_50hz \
    --duration_frames 500 \
    --num_workers 8
```

**Parameters:**
- `--dataset`: Dataset name
- `--split`: Data split (train/validation/test)
- `--max_examples`: Maximum number of examples to create
- `--output_dir`: Output directory
- `--data_base_dir`: Base directory containing extracted tokens
- `--duration_frames`: Duration in frames (default: 500 = 10 seconds at 50 FPS)
- `--num_workers`: Number of parallel workers
- `--start_index`: Starting example index (for resuming)

**Output Structure:**
```
midi_mixdown/
└── slakh2100/
    ├── train/
    │   ├── 0000000/
    │   │   ├── input_tokens.pt   # [num_input_stems, time, 4]
    │   │   ├── target_tokens.pt  # [num_target_stems, time, 4]
    │   │   └── metadata.json     # Track and stem information
    │   ├── 0000001/
    │   └── ...
    ├── validation/
    └── test/
```

## Token Format

Each token file (`.pt`) contains a PyTorch tensor of shape `[time_steps, num_heads]`:
- **time_steps**: Number of time frames (1 frame = 20ms at 50 FPS)
- **num_heads**: Number of concurrent note events (default: 4)

Each token represents a MIDI note event encoded using the YourMT3 codec:
- Pitch events: Non-drum notes (sustained throughout note duration)
- Drum events: Drum/percussion notes (sustained throughout note duration)
- PAD tokens: No event at this position

**Note Duration Handling:**
When a MIDI note plays from onset to offset (e.g., a piano note held for 1 second), the corresponding pitch token is repeated across all frames during that duration. For example, a 1-second note at 50 FPS will have the same pitch token for 50 consecutive frames.

## Comparison with Audio Pipeline

| Audio Pipeline | MIDI Pipeline |
|----------------|---------------|
| `extract_causal_dac_32k.py` | `extract_midi_tokens.py` |
| `extract_rms.py` | *(Not needed for MIDI)* |
| `dump_audio_mixdown.py` | `dump_midi_mixdown.py` |

The MIDI pipeline follows the same structure but:
- Processes MIDI files instead of audio
- Uses YourMT3 tokenizer instead of DAC codec
- Doesn't require RMS extraction (no volume information in symbolic music)

## Dependencies

Required packages:
- `torch`
- `pandas`
- `tqdm`
- `pretty_midi`
- `numpy`

The YourMT3 tokenizer modules must be available:
- `utils.tokenizer`
- `utils.note_event_dataclasses`

## Notes

- The default FPS is 50 (20ms resolution), matching common music transcription models
- Each frame can contain up to 4 concurrent note events (configurable)
- Events within a frame are sorted: non-drum notes first, then by pitch (ascending)
- Each note's pitch token is repeated across all frames from onset to offset (sustained representation)
- Frame calculations: start_frame = int(note.start * fps), end_frame = int(round(note.end * fps))
- Excess events beyond `num_heads` are truncated
- The tokenizer uses the MT3 codec by default
