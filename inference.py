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

"""Inference script for HiggsAudiov2 (Unofficial Implementation).

HiggsAudiov2 audio tokenizer inference pipeline using DAC-style architecture.
Provides utilities for encoding and decoding audio files using trained models.
"""

from pathlib import Path
from typing import Union
import torch
from tqdm import tqdm

from HiggsAudioTokenizer.infer import CodecPipeline
from HiggsAudioTokenizer.exceptions import ModelError, ProcessingError


def main(
    checkpoint: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cuda",
    batch_size: int = 1,
    decode: bool = False
) -> None:
    """Main inference function for audio encoding/decoding.

    Processes audio files through the HiggsAudiov2 codec for compression or decompression.
    Supports batch processing of multiple files in a directory.
    
    Args:
        checkpoint: Path to trained model checkpoint file (.pt)
        input_path: Input audio file or directory containing files to process
        output_path: Output directory where results will be saved
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Number of files to process in parallel (currently unused,
            processes files sequentially)
        decode: If True, decode compressed .npz files to .wav audio;
            if False, encode audio files to compressed .npz format
    
    Raises:
        ValueError: If no appropriate files are found in input_path
        
    Note:
        Compressed files are saved as .npz containing discrete codes and metadata.
        Supports common audio formats: wav, mp3, flac, m4a, ogg.
    """
    # Convert paths
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint}...")
    
    # Create codec pipeline
    try:
        pipeline = CodecPipeline(model_path=checkpoint, device=device)
        print(f"Model loaded successfully. Sample rate: {pipeline.sample_rate} Hz")
    except Exception as e:
        raise ModelError(f"Failed to load model from {checkpoint}: {e}") from e
    
    # Determine files to process
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        if decode:
            # Look for compressed files
            files_to_process = list(input_path.glob("*.npz"))
            if not files_to_process:
                files_to_process = list(input_path.glob("*.dac"))
        else:
            # Look for audio files
            patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
            files_to_process = []
            for pattern in patterns:
                files_to_process.extend(input_path.glob(pattern))
    
    if not files_to_process:
        raise ValueError(f"No {'compressed' if decode else 'audio'} files found in {input_path}")
    
    print(f"Found {len(files_to_process)} files to process")
    print(f"Mode: {'Decoding' if decode else 'Encoding'}")
    
    # Process files
    successful = 0
    failed = 0
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            if decode:
                # Decode compressed file to audio
                output_file = output_path / file_path.with_suffix('.wav').name
                pipeline.decompress(file_path, output_file)
            else:
                # Encode audio file to compressed format
                output_file = output_path / file_path.with_suffix('.npz').name
                pipeline.compress(file_path, output_file, normalize=True)
            
            successful += 1
        except Exception as e:
            raise ProcessingError(f"Failed to process {file_path.name}: {e}") from e
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    # Example usage for testing
    import sys
    if len(sys.argv) > 1:
        main(
            checkpoint=sys.argv[1],
            input_path=sys.argv[2] if len(sys.argv) > 2 else ".",
            output_path=sys.argv[3] if len(sys.argv) > 3 else "output/",
        )