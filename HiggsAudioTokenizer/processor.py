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

"""Audio processors for HiggsAudiov2 (Unofficial Implementation).

Provides comprehensive audio processing utilities including:
- Audio loading with multiple backend support (torchaudio, soundfile)
- Streaming audio loading for large files
- Audio format conversion and validation
- Normalization and preprocessing
- Batched processing utilities
- WaveDataset for efficient audio data handling
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import resampy
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import subprocess
import tempfile
from typing import Tuple

from .exceptions import ImportError as HiggsImportError

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


@dataclass
class TokenizerFile:
    """Container for compressed audio codes and metadata.

    Stores discrete codes along with information needed for reconstruction.
    
    Args:
        codes: Discrete codes tensor of shape (B, N, T)
        original_length: Original audio length before padding
        sample_rate: Audio sample rate
        hop_length: Hop length used for padding
        input_loudness_db: Input loudness in dB (if normalized)
        channels: Number of audio channels
        padding: Whether padding was applied
        tokenizer_version: Version string for compatibility
    """
    
    codes: torch.Tensor
    original_length: int
    sample_rate: int = 24000
    hop_length: int = 320
    input_loudness_db: Optional[float] = None
    channels: int = 1
    padding: bool = True
    tokenizer_version: str = "higgs-audio-v2-unofficial"
    
    def save(self, path: Union[str, Path]):
        """Save compressed audio to file.
        
        Args:
            path: Path to save file (will add .npz extension)
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.npz')
        
        # Convert codes to numpy
        codes_np = self.codes.cpu().numpy()
        
        # Save all metadata
        np.savez(
            path,
            codes=codes_np,
            original_length=self.original_length,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            input_loudness_db=self.input_loudness_db,
            channels=self.channels,
            padding=self.padding,
            tokenizer_version=self.tokenizer_version
        )
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TokenizerFile':
        """Load compressed audio from file.
        
        Args:
            path: Path to compressed file
            
        Returns:
            TokenizerFile instance
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        
        # Load codes and convert to tensor
        codes = torch.from_numpy(data['codes'])
        
        # Load metadata - all fields are required
        return cls(
            codes=codes,
            original_length=int(data['original_length']),
            sample_rate=int(data['sample_rate']),
            hop_length=int(data['hop_length']),
            input_loudness_db=float(data['input_loudness_db']) if 'input_loudness_db' in data else None,
            channels=int(data['channels']),
            padding=bool(data['padding']),
            tokenizer_version=str(data['tokenizer_version'])
        )


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 24000,
    mono: bool = True,
    normalize: bool = True,
    device: str = "cpu",
    use_ffmpeg: bool = False
) -> torch.Tensor:
    """Load audio file and preprocess.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize to [-1, 1] if True
        device: Device to place tensor on
        use_ffmpeg: Use ffmpeg for loading (for very long files)
        
    Returns:
        Audio tensor of shape (num_samples,) or (channels, num_samples)
    """
    # Validate file exists
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # Load audio
    try:
        if use_ffmpeg:
            # Use ffmpeg for very long files (> 10 minutes)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                cmd = [
                    "ffmpeg", "-i", str(audio_path),
                    "-ar", str(sample_rate),
                    "-ac", "1" if mono else "2",
                    "-f", "wav",
                    "-y", tmp.name
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFMPEG error: {result.stderr}")
                audio, sr = sf.read(tmp.name)
        else:
            audio, sr = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {audio_path}: {str(e)}")
    
    # Handle numpy array shape (soundfile returns (samples, channels))
    if len(audio.shape) == 2:
        audio = audio.T  # (channels, samples)
    else:
        audio = audio[np.newaxis, :]  # (1, samples)
    
    # Convert to mono if requested
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    
    # Resample if necessary
    if sr != sample_rate:
        if audio.shape[0] == 1:
            audio = resampy.resample(audio[0], sr, sample_rate)
            audio = audio[np.newaxis, :]
        else:
            # Resample each channel
            audio_resampled = []
            for c in range(audio.shape[0]):
                audio_resampled.append(resampy.resample(audio[c], sr, sample_rate))
            audio = np.stack(audio_resampled, axis=0)
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()
    
    # Normalize
    if normalize:
        # Check for extreme values
        max_val = audio_tensor.abs().max()
        if max_val > 1.0:
            # Silent clamping - don't print warning
            pass
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    
    return audio_tensor.to(device)


@dataclass
class AudioTokenizerProcessor:
    """Base processor for audio tokenization.

    Handles core audio preprocessing operations.
    
    Args:
        sample_rate: Target sample rate for audio
        segment_length: Length of audio segments in samples
        normalize: Whether to normalize audio to [-1, 1]
        device: Device to place tensors on
        hop_length: Hop length for padding alignment (model dependent)
        normalize_loudness: Whether to normalize loudness
        target_loudness_db: Target loudness in dB for normalization
    """
    
    sample_rate: int = 24000
    segment_length: int = 24000  # 1 second at 24kHz
    normalize: bool = True
    device: str = "cpu"
    hop_length: int = 320  # Default hop length for 24kHz models
    normalize_loudness: bool = False
    target_loudness_db: float = -16.0
    mono: bool = True  # Convert to mono by default
    supported_sample_rates: List[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.supported_sample_rates is None:
            self.supported_sample_rates = [16000, 24000, 44100, 48000]
        
        if self.sample_rate not in self.supported_sample_rates:
            print(f"Warning: Sample rate {self.sample_rate} not in supported rates {self.supported_sample_rates}")
    
    def prepare_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor, str, Path],
        input_sr: Optional[int] = None
    ) -> torch.Tensor:
        """Prepare audio for processing.
        
        Args:
            audio: Audio data (numpy array, tensor, or file path)
            input_sr: Input sample rate (required if audio is array/tensor)
            
        Returns:
            Processed audio tensor
        """
        # Load from file if path provided
        if isinstance(audio, (str, Path)):
            return load_audio(
                audio,
                sample_rate=self.sample_rate,
                mono=True,
                normalize=self.normalize,
                device=self.device
            )
        
        # Convert numpy to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure tensor
        assert isinstance(audio, torch.Tensor), "Audio must be tensor at this point"
        
        # Handle stereo to mono if requested
        if self.mono:
            if len(audio.shape) == 2 and audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            elif len(audio.shape) == 1:
                pass  # Already mono
            else:
                raise ValueError(f"Unexpected audio shape: {audio.shape}")
        else:
            # Keep stereo, ensure 2D
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)  # Add channel dimension
        
        # Resample if necessary
        if input_sr is not None and input_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio.unsqueeze(0) if len(audio.shape) == 1 else audio,
                orig_freq=input_sr,
                new_freq=self.sample_rate
            ).squeeze(0)
        
        # Normalize
        if self.normalize:
            audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio.to(self.device)
    
    def normalize_audio_loudness(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        target_db: float = -16.0
    ) -> Tuple[torch.Tensor, float]:
        """Normalize audio loudness to target dB.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            target_db: Target loudness in dB
            
        Returns:
            Tuple of (normalized_audio, input_db)
        """
        if not HAS_PYLOUDNORM:
            raise HiggsImportError("pyloudnorm is required for loudness normalization. Install it with: pip install pyloudnorm")
        
        # Convert to numpy for pyloudnorm
        audio_np = audio.cpu().numpy()
        
        # Create loudness meter
        meter = pyln.Meter(sample_rate)
        
        # Measure loudness
        input_loudness = meter.integrated_loudness(audio_np)
        
        # Normalize
        normalized_audio = pyln.normalize.loudness(audio_np, input_loudness, target_db)
        
        # Convert back to tensor
        normalized_tensor = torch.from_numpy(normalized_audio).to(audio.device)
        
        return normalized_tensor, input_loudness
    
    def preprocess(
        self,
        audio: torch.Tensor,
        padding_multiple: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """Preprocess audio for model input.
        
        Args:
            audio: Audio tensor of shape (T,) or (C, T)
            padding_multiple: Pad to multiple of this value (default: hop_length)
            
        Returns:
            Tuple of (preprocessed_audio, original_length)
        """
        original_length = audio.shape[-1]
        
        # Use hop_length as default padding multiple
        if padding_multiple is None:
            padding_multiple = self.hop_length
        
        # Calculate padding
        remainder = original_length % padding_multiple
        if remainder > 0:
            pad_length = padding_multiple - remainder
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        # Apply loudness normalization if enabled
        if self.normalize_loudness:
            if len(audio.shape) == 1:
                audio, _ = self.normalize_audio_loudness(
                    audio, self.sample_rate, self.target_loudness_db
                )
            elif len(audio.shape) == 2:
                # Process each channel separately for stereo
                normalized_channels = []
                for ch in range(audio.shape[0]):
                    ch_normalized, _ = self.normalize_audio_loudness(
                        audio[ch], self.sample_rate, self.target_loudness_db
                    )
                    normalized_channels.append(ch_normalized)
                audio = torch.stack(normalized_channels)
        
        return audio, original_length


@dataclass
class TokenizerTrainingProcessor(AudioTokenizerProcessor):
    """Processor for tokenizer training.
    
    Extends base processor with training-specific features like
    random segment extraction and augmentation support.
    
    Args:
        random_segment: Extract random segments if True, sequential if False
        augment: Apply augmentations (not implemented yet)
    """
    
    random_segment: bool = True
    augment: bool = False
    
    def extract_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract a segment from audio.
        
        Args:
            audio: Full audio tensor
            
        Returns:
            Segment of length self.segment_length
        """
        audio_length = len(audio)
        
        # Pad if too short
        if audio_length < self.segment_length:
            pad_amount = self.segment_length - audio_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            return audio
        
        # Extract segment
        if self.random_segment and audio_length > self.segment_length:
            start = torch.randint(0, audio_length - self.segment_length + 1, (1,)).item()
        else:
            start = 0
        
        return audio[start:start + self.segment_length]
    
    def __call__(
        self,
        audio_path: Union[str, Path],
        input_sr: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Process audio file for training.
        
        Returns dict compatible with dataset.py expectations.
        """
        # Load and prepare audio
        audio = self.prepare_audio(audio_path, input_sr)
        
        # Extract segment
        audio_segment = self.extract_segment(audio)
        
        # Apply augmentations if enabled
        if self.augment:
            # TODO: Implement augmentations
            pass
        
        return {
            "raw_wav": audio_segment,
            "id": str(audio_path),
            "text": "",  # Not used for tokenization
            "task": "reconstruction",
            "prompt": ""  # Not used for tokenization
        }


@dataclass
class TokenizerInferenceProcessor(AudioTokenizerProcessor):
    """Processor for tokenizer inference.
    
    Handles full audio files with optional chunking for long sequences.
    
    Args:
        chunk_length: Length of chunks in samples (None for no chunking)
        overlap: Overlap between chunks in samples
        pad_to_length: Pad audio to multiple of this length
    """
    
    chunk_length: Optional[int] = None
    overlap: int = 0
    pad_to_length: Optional[int] = None
    
    def prepare_for_inference(
        self,
        audio: Union[np.ndarray, torch.Tensor, str, Path],
        input_sr: Optional[int] = None
    ) -> torch.Tensor:
        """Prepare audio for inference.
        
        Args:
            audio: Audio data
            input_sr: Input sample rate
            
        Returns:
            Processed audio tensor
        """
        # Basic preparation
        audio = self.prepare_audio(audio, input_sr)
        
        # Pad to multiple of pad_to_length if specified
        if self.pad_to_length is not None:
            remainder = len(audio) % self.pad_to_length
            if remainder > 0:
                pad_amount = self.pad_to_length - remainder
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
        
        return audio
    
    def process_file(
        self,
        audio_path: Union[str, Path],
        input_sr: Optional[int] = None,
        return_chunks: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Process full audio file.
        
        Args:
            audio_path: Path to audio file
            input_sr: Input sample rate
            return_chunks: Return list of chunks if True
            
        Returns:
            Processed audio tensor or list of chunks
        """
        audio = self.prepare_for_inference(audio_path, input_sr)
        
        if self.chunk_length is None or len(audio) <= self.chunk_length:
            return [audio] if return_chunks else audio
        
        # Chunk audio with overlap
        chunks = []
        step = self.chunk_length - self.overlap
        
        for start in range(0, len(audio) - self.overlap, step):
            end = min(start + self.chunk_length, len(audio))
            chunk = audio[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < self.chunk_length:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.chunk_length - len(chunk))
                )
            
            chunks.append(chunk)
        
        return chunks if return_chunks else torch.stack(chunks)
    
    def overlap_add_reconstruction(
        self,
        chunks: List[torch.Tensor],
        original_length: Optional[int] = None
    ) -> torch.Tensor:
        """Reconstruct audio from overlapping chunks.
        
        Args:
            chunks: List of overlapping audio chunks
            original_length: Original audio length to trim to
            
        Returns:
            Reconstructed audio tensor
        """
        if not chunks:
            return torch.zeros(0, device=self.device)
        
        if len(chunks) == 1:
            audio = chunks[0]
            if original_length is not None:
                audio = audio[:original_length]
            return audio
        
        # Calculate step size
        step = self.chunk_length - self.overlap
        
        # Calculate output length
        output_length = (len(chunks) - 1) * step + self.chunk_length
        if original_length is not None:
            output_length = min(output_length, original_length)
        
        # Initialize output
        output = torch.zeros(output_length, device=chunks[0].device)
        weight = torch.zeros(output_length, device=chunks[0].device)
        
        # Create fade window for overlap region
        if self.overlap > 0:
            fade_in = torch.linspace(0, 1, self.overlap, device=chunks[0].device)
            fade_out = torch.linspace(1, 0, self.overlap, device=chunks[0].device)
        
        # Overlap-add chunks
        for i, chunk in enumerate(chunks):
            start = i * step
            end = min(start + self.chunk_length, output_length)
            chunk_len = end - start
            
            # Apply fading in overlap regions
            if self.overlap > 0 and i > 0:
                # Fade in at the beginning of this chunk
                chunk[:self.overlap] *= fade_in
            if self.overlap > 0 and i < len(chunks) - 1:
                # Fade out at the end of this chunk
                overlap_start = self.chunk_length - self.overlap
                chunk[overlap_start:] *= fade_out
            
            # Add to output
            output[start:end] += chunk[:chunk_len]
            weight[start:end] += 1.0
        
        # Normalize by weight (handles overlap regions)
        output = output / weight.clamp(min=1.0)
        
        return output
    
    def compress(
        self,
        audio_path: Union[str, Path],
        model: 'AudioCodec',
        chunk_duration: float = 5.0,
        use_ffmpeg_for_long: bool = True
    ) -> TokenizerFile:
        """Compress audio file using the tokenizer model.
        
        Args:
            audio_path: Path to audio file
            model: AudioCodec model instance
            chunk_duration: Duration of chunks in seconds for long files
            use_ffmpeg_for_long: Use ffmpeg for files > 10 minutes
            
        Returns:
            TokenizerFile with compressed codes and metadata
        """
        # Check if we should use ffmpeg
        use_ffmpeg = False
        if use_ffmpeg_for_long:
            try:
                import librosa
                duration = librosa.get_duration(path=str(audio_path))
                use_ffmpeg = duration > 600  # 10 minutes
            except ImportError:
                raise HiggsImportError("librosa is required for automatic duration detection. Install it with: pip install librosa")
        
        # Load audio
        audio = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            mono=True,
            normalize=self.normalize,
            device=self.device,
            use_ffmpeg=use_ffmpeg
        )
        
        # Track input loudness if normalizing
        input_loudness_db = None
        if self.normalize_loudness:
            audio, input_loudness_db = self.normalize_audio_loudness(
                audio, self.sample_rate, self.target_loudness_db
            )
        
        # Preprocess (pad to hop_length)
        audio, original_length = self.preprocess(audio, model.downsample_rate)
        
        # Add batch and channel dimensions
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # (1, C, T)
        
        # Process in chunks if needed
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        if audio.shape[-1] > chunk_samples:
            # Process in chunks
            all_codes = []
            for start in range(0, audio.shape[-1], chunk_samples):
                end = min(start + chunk_samples, audio.shape[-1])
                chunk = audio[:, :, start:end]
                
                # Pad last chunk if needed
                if chunk.shape[-1] < chunk_samples:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, chunk_samples - chunk.shape[-1])
                    )
                
                # Encode chunk
                with torch.no_grad():
                    codes = model.compress(chunk)
                all_codes.append(codes)
            
            # Concatenate all codes
            codes = torch.cat(all_codes, dim=-1)
        else:
            # Process entire audio
            with torch.no_grad():
                codes = model.compress(audio)
        
        # Create TokenizerFile
        return TokenizerFile(
            codes=codes,
            original_length=original_length,
            sample_rate=self.sample_rate,
            hop_length=model.downsample_rate,
            input_loudness_db=input_loudness_db,
            channels=audio.shape[1] if audio.dim() == 3 else 1,
            padding=True,
            tokenizer_version="higgs-audio-v2-unofficial"
        )
    
    def decompress(
        self,
        tokenizer_file: Union[TokenizerFile, str, Path],
        model: 'AudioCodec',
        denormalize_loudness: bool = True
    ) -> torch.Tensor:
        """Decompress audio from TokenizerFile.
        
        Args:
            tokenizer_file: TokenizerFile instance or path to compressed file
            model: AudioCodec model instance
            denormalize_loudness: Restore original loudness if True
            
        Returns:
            Reconstructed audio tensor
        """
        # Load file if path provided
        if isinstance(tokenizer_file, (str, Path)):
            tokenizer_file = TokenizerFile.load(tokenizer_file)
        
        # Decompress codes
        with torch.no_grad():
            audio = model.decompress(tokenizer_file.codes.to(self.device))
        
        # Remove batch dimension if present
        if audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        # Trim to original length
        if audio.shape[-1] > tokenizer_file.original_length:
            audio = audio[..., :tokenizer_file.original_length]
        
        # Restore loudness if applicable
        if denormalize_loudness and tokenizer_file.input_loudness_db is not None:
            # Measure current loudness
            current_audio, current_db = self.normalize_audio_loudness(
                audio.squeeze(0) if audio.dim() == 2 else audio,
                tokenizer_file.sample_rate,
                tokenizer_file.input_loudness_db
            )
            audio = current_audio
        
        return audio
    
    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        input_srs: Optional[List[int]] = None
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Process batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            input_srs: List of input sample rates
            
        Returns:
            Batch dict with padded audio and metadata
        """
        if input_srs is None:
            input_srs = [None] * len(audio_paths)
        
        audios = []
        for path, sr in zip(audio_paths, input_srs):
            audio = self.prepare_for_inference(path, sr)
            audios.append(audio)
        
        # Pad to same length
        audios_padded = pad_sequence(audios, batch_first=True, padding_value=0)
        
        # Create padding mask
        lengths = torch.tensor([len(a) for a in audios])
        max_length = audios_padded.shape[1]
        padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
        
        return {
            "raw_wav": audios_padded,
            "padding_mask": padding_mask,
            "lengths": lengths,
            "ids": [str(p) for p in audio_paths]
        }


class TokenizerDataset(torch.utils.data.Dataset):
    """Dataset for tokenizer training/inference using processors.
    
    Args:
        manifest_path: Path to JSONL manifest file
        processor: Processor instance to use
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        processor: Union[TokenizerTrainingProcessor, TokenizerInferenceProcessor]
    ):
        super().__init__()
        self.processor = processor
        
        # Load manifest
        self.data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                item = eval(line.strip()) if line.strip() else {}
                if 'audio_path' in item:
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} items from {manifest_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item using processor."""
        item = self.data[idx]
        audio_path = item['audio_path']
        input_sr = item.get('sample_rate', None)
        
        # Use processor to handle audio
        if isinstance(self.processor, TokenizerTrainingProcessor):
            return self.processor(audio_path, input_sr)
        else:
            # For inference processor, return simple format
            audio = self.processor.prepare_for_inference(audio_path, input_sr)
            return {
                "raw_wav": audio,
                "id": str(audio_path),
                "index": idx,
                "text": "",
                "task": "reconstruction",
                "prompt": ""
            }


def collate_audio_batch(samples: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate audio samples into batch.
    
    Args:
        samples: List of sample dicts from dataset
        
    Returns:
        Batched dict with proper padding and masks
    """
    # Extract audio tensors
    raw_wavs = [s["raw_wav"] for s in samples]
    
    # Pad sequences
    raw_wav_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
    
    # Create padding mask (True where padded)
    lengths = torch.tensor([len(wav) for wav in raw_wavs])
    max_length = raw_wav_padded.shape[1]
    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    
    # Collect other fields - all are required
    ids = [s["id"] for s in samples]
    texts = [s["text"] if "text" in s else "" for s in samples]
    tasks = [s["task"] if "task" in s else "reconstruction" for s in samples]
    prompts = [s["prompt"] if "prompt" in s else "" for s in samples]
    indices = [s["index"] for s in samples]
    
    return {
        "raw_wav": raw_wav_padded,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "id": ids,
        "text": texts,
        "task": tasks,
        "prompt": prompts,
        "index": indices
    }