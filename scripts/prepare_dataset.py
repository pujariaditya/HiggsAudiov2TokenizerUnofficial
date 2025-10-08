#!/usr/bin/env python3
# Copyright 2025 HiggsAudiov2-Unofficial Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare LibriSpeech dataset for training by creating JSONL files.
Processes audio files and creates train.jsonl, valid.jsonl, and test.jsonl files.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def get_audio_duration(audio_path):
    """Get duration of an audio file in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None


def process_audio_file(audio_path, base_path):
    """Process a single audio file and return metadata."""
    try:
        # Extract IDs from path: speaker/chapter/speaker-chapter-segment.flac
        parts = audio_path.split(os.sep)
        filename = os.path.basename(audio_path)
        file_id = filename.replace('.flac', '')
        
        # Get audio duration
        duration = get_audio_duration(audio_path)
        if duration is None:
            return None
        
        # Create entry
        entry = {
            "id": f"librispeech_{file_id}",
            "audio_path": audio_path,
            "duration": round(duration, 3)
        }
        
        return entry
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_subset(subset_path, subset_name, max_workers=None):
    """Process a single subset (e.g., train-clean-100)."""
    if not os.path.exists(subset_path):
        print(f"Warning: Subset path {subset_path} does not exist, skipping...")
        return []
    
    print(f"\nProcessing {subset_name}...")
    
    # Find all FLAC files
    pattern = os.path.join(subset_path, "*", "*", "*.flac")
    audio_files = glob(pattern)
    
    if not audio_files:
        print(f"No audio files found in {subset_path}")
        return []
    
    print(f"Found {len(audio_files)} audio files in {subset_name}")
    
    entries = []
    
    # Process files in parallel
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_audio_file, audio_file, subset_path): audio_file 
            for audio_file in audio_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(audio_files), desc=f"Processing {subset_name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    entries.append(result)
                pbar.update(1)
    
    print(f"Successfully processed {len(entries)} files from {subset_name}")
    return entries


def write_jsonl(entries, output_path):
    """Write entries to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech dataset for training"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./Dataset/librispeech/LibriSpeech",
        help="Path to LibriSpeech dataset folder (default: ./Dataset/librispeech/LibriSpeech)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for JSONL files (default: data)"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs='+',
        default=None,
        help="Specific subsets to process (e.g., train-clean-100 dev-clean)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Expand user path
    dataset_path = os.path.expanduser(args.dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        sys.exit(1)
    
    # Define subset mappings
    train_subsets = ["train-clean-100", "train-clean-360", "train-other-500"]
    valid_subsets = ["dev-clean", "dev-other"]
    test_subsets = ["test-clean", "test-other"]
    
    # Filter subsets if specified
    if args.subsets:
        train_subsets = [s for s in train_subsets if s in args.subsets]
        valid_subsets = [s for s in valid_subsets if s in args.subsets]
        test_subsets = [s for s in test_subsets if s in args.subsets]
    
    # Process each subset group
    all_train_entries = []
    all_valid_entries = []
    all_test_entries = []
    
    # Process training subsets
    for subset in train_subsets:
        subset_path = os.path.join(dataset_path, subset)
        entries = process_subset(subset_path, subset, args.max_workers)
        all_train_entries.extend(entries)
    
    # Process validation subsets
    for subset in valid_subsets:
        subset_path = os.path.join(dataset_path, subset)
        entries = process_subset(subset_path, subset, args.max_workers)
        all_valid_entries.extend(entries)
    
    # Process test subsets
    for subset in test_subsets:
        subset_path = os.path.join(dataset_path, subset)
        entries = process_subset(subset_path, subset, args.max_workers)
        all_test_entries.extend(entries)
    
    # Write output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("Writing output files...")
    
    if all_train_entries:
        write_jsonl(all_train_entries, output_dir / "train.jsonl")
    
    if all_valid_entries:
        write_jsonl(all_valid_entries, output_dir / "valid.jsonl")
    
    if all_test_entries:
        write_jsonl(all_test_entries, output_dir / "test.jsonl")
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print(f"Train samples: {len(all_train_entries)}")
    print(f"Valid samples: {len(all_valid_entries)}")
    print(f"Test samples: {len(all_test_entries)}")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()