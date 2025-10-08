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

"""Command-line interface for HiggsAudiov2 (Unofficial Implementation).

HiggsAudiov2 audio tokenizer CLI using DAC-style architecture.
Provides commands for training, encoding, and decoding audio files.
"""

from pathlib import Path

import click
import torch

from train import main as train_fn
from inference import main as infer_fn


def common_options(f):
    """Common options decorator for CLI commands.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function with common CLI options
    """
    f = click.option("--cfg-path", required=True, type=Path, help="Path to configuration file")(f)
    f = click.option(
        "--options",
        default=[],
        multiple=True,
        type=str,
        help="Override fields in the config. A list of key=value pairs",
    )(f)
    return f


@click.group()
def audiotokenizer():
    """Higgs Audio V2 Tokenizer (Unofficial Implementation)."""
    pass


@audiotokenizer.command()
@common_options
def train(cfg_path, options):
    """Train a Higgs Audio V2 Tokenizer model from a configuration file.
    
    Args:
        cfg_path: Path to YAML configuration file
        options: List of config overrides as key=value pairs
    """
    train_fn(cfg_path=cfg_path, options=options)


@audiotokenizer.command()
@click.option("--checkpoint", required=True, type=Path, help="Path to model checkpoint")
@click.option("--input", required=True, type=Path, help="Input audio file or directory")
@click.option("--output", required=True, type=Path, help="Output directory for compressed .npz files")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to run on (cuda/cpu)")
@click.option("--batch-size", default=1, type=int, help="Batch size for processing")
def encode(checkpoint, input, output, device, batch_size):
    """Encode audio files to compressed format.
    
    Compresses audio files using the trained tokenizer model, producing .npz files
    containing discrete codes and metadata.
    
    Args:
        checkpoint: Path to trained model checkpoint
        input: Input audio file or directory containing audio files
        output: Output directory for compressed .npz files
        device: Device to run inference on (cuda/cpu)
        batch_size: Number of files to process in parallel
    """
    infer_fn(
        checkpoint=checkpoint,
        input_path=input,
        output_path=output,
        device=device,
        batch_size=batch_size,
        decode=False
    )


@audiotokenizer.command()
@click.option("--checkpoint", required=True, type=Path, help="Path to model checkpoint")
@click.option("--input", required=True, type=Path, help="Input compressed .npz file or directory")
@click.option("--output", required=True, type=Path, help="Output directory for decoded audio files")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to run on (cuda/cpu)")
@click.option("--batch-size", default=1, type=int, help="Batch size for processing")
def decode(checkpoint, input, output, device, batch_size):
    """Decode compressed files back to audio.
    
    Decompresses .npz files containing discrete codes back to audio files
    using the trained tokenizer model.
    
    Args:
        checkpoint: Path to trained model checkpoint
        input: Input .npz file or directory containing compressed files
        output: Output directory for decoded .wav files
        device: Device to run inference on (cuda/cpu)
        batch_size: Number of files to process in parallel
    """
    infer_fn(
        checkpoint=checkpoint,
        input_path=input,
        output_path=output,
        device=device,
        batch_size=batch_size,
        decode=True
    )


if __name__ == "__main__":
    audiotokenizer()