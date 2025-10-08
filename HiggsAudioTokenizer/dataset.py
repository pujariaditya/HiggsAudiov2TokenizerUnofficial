"""
Dataset module for audio tokenizer training.

Provides dataset classes for loading and processing audio files from JSONL manifests,
compatible with the training pipeline.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import TrainingConfig
from .processor import load_audio
from .exceptions import DatasetError, AudioLoadError


class AudioDataset(Dataset):
    """Simple audio dataset for tokenizer training.
    
    Loads audio files from a JSONL manifest file where each line contains:
    {"audio_path": "path/to/audio.wav", "duration": 10.5}
    
    Additional fields are ignored but preserved for flexibility.
    
    Args:
        manifest_path: Path to JSONL manifest file
        segment_length: Length of audio segments in samples
        sample_rate: Target sample rate for audio
        normalize: Whether to normalize audio to [-1, 1]
        random_segment: Extract random segments if True, first segment if False
        device: Device to place tensors on ("cpu" or "cuda")
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        segment_length: int = 24000,
        sample_rate: int = 24000,
        normalize: bool = True,
        random_segment: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.random_segment = random_segment
        self.device = device
        
        # Load manifest
        self.data = []
        manifest_path = Path(manifest_path)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest_path} not found")
        
        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    if 'audio_path' in item:
                        self.data.append(item)
                    else:
                        raise DatasetError(f"Line {line_num} in {manifest_path} missing 'audio_path' field")
                except json.JSONDecodeError as e:
                    raise DatasetError(f"Failed to parse line {line_num} in {manifest_path}: {e}") from e
        
        if not self.data:
            raise ValueError(f"No valid audio entries found in {manifest_path}")
        
        print(f"Loaded {len(self.data)} audio files from {manifest_path}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Load and process a single audio sample.
        
        Args:
            idx: Index of the sample to load
        
        Returns:
            Dictionary containing:
                - raw_wav: Audio tensor of shape (segment_length,)
                - id: Path to the audio file (identifier)
                - index: Index in the dataset
        """
        item = self.data[idx]
        audio_path = item['audio_path']
        
        try:
            # Load audio
            audio = load_audio(
                audio_path,
                sample_rate=self.sample_rate,
                mono=True,
                normalize=self.normalize,
                device=self.device
            )
            
            # Extract segment
            if audio.shape[1] > self.segment_length:
                if self.random_segment:
                    # Random segment for training
                    start = random.randint(0, audio.shape[1] - self.segment_length)
                    audio = audio[:, start:start + self.segment_length]
                else:
                    # First segment for validation
                    audio = audio[:, :self.segment_length]
            elif audio.shape[1] < self.segment_length:
                # Pad if too short
                pad_length = self.segment_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, pad_length))
            
            # Remove channel dimension to match HiggsAudiov2 format
            audio = audio.squeeze(0)  # Shape: (segment_length,)
            
            return {
                "raw_wav": audio,
                "id": audio_path,
                "index": idx
            }
            
        except Exception as e:
            raise AudioLoadError(f"Failed to load audio from {audio_path}: {e}") from e


def collater(samples: List[Dict[str, Union[torch.Tensor, str, int]]]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate samples into a batch.
    
    Pads audio samples to the same length and creates padding masks.
    
    Args:
        samples: List of sample dictionaries from the dataset, each containing:
            - raw_wav: Audio tensor
            - id: File identifier
            - index: Dataset index
        
    Returns:
        Batched dictionary containing:
            - raw_wav: Padded audio tensor of shape (B, 1, max_length)
            - padding_mask: Boolean mask for padding (B, max_length)
            - id: List of file identifiers
            - index: List of dataset indices
    """
    # Extract raw_wav from samples
    raw_wav = [s["raw_wav"] for s in samples]
    
    # Calculate lengths for padding mask
    raw_wav_length = torch.tensor([len(a) for a in raw_wav])
    
    # Pad sequences to same length
    raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
    
    # Add channel dimension to make it 3D: (batch, length) -> (batch, 1, length)
    raw_wav = raw_wav.unsqueeze(1)
    
    # Create padding mask (True where padded)
    padding_mask = torch.arange(raw_wav.size(2)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
    
    # Collect metadata
    id = [s["id"] for s in samples]
    index = [s["index"] for s in samples]
    
    return {
        "raw_wav": raw_wav,
        "padding_mask": padding_mask,
        "id": id,
        "index": index
    }


