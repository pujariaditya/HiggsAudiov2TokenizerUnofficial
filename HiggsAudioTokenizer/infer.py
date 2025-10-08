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

"""Inference pipeline for HiggsAudiov2 (Unofficial Implementation).

Provides high-level APIs for audio encoding/decoding, batch processing,
and evaluation of the trained codec. Based on the HiggsAudiov2 architecture.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import warnings

from .models.audiotokenizer import HiggsAudioTokenizer
from .config import AudioCodecConfig
from .utils import load_model
from .metrics import compute_codec_metrics


class CodecPipeline:
    """High-level inference pipeline for audio codec.
    
    Provides convenient APIs for audio compression and decompression with
    support for long audio processing through chunking.
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        model: HiggsAudioTokenizer | None = None,
        device: str | torch.device = "cuda",
        use_ema: bool = True
    ):
        """Initialize codec pipeline.
        
        Args:
            model_path: Path to model checkpoint or tag (e.g., '24khz').
            model: Pre-loaded model (if model_path not provided).
            device: Device to run inference on.
            use_ema: Whether to use EMA weights if available.
            
        Raises:
            ValueError: If neither model_path nor model is provided.
        """
        self.device = device
        
        if model is not None:
            self.model = model.to(device)
        elif model_path is not None:
            self.model = load_model(model_path, device=device)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.model.eval()
        self.sample_rate = self.model.sample_rate
        
        # Try to load EMA weights if requested
        if use_ema and model_path is not None and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'ema_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['ema_state_dict'])
                    print("Loaded EMA weights")
            except:
                pass
    
    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor | np.ndarray | str | Path,
        sample_rate: int | None = None,
        normalize: bool = True,
        return_metadata: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Encode audio to discrete codes.
        
        Args:
            audio: Audio tensor, numpy array, or path to audio file.
            sample_rate: Sample rate of input audio (if tensor/array).
                If None, assumes model sample rate.
            normalize: Whether to normalize audio to [-1, 1].
            return_metadata: Whether to return encoding metadata.
            
        Returns:
            Codes tensor of shape (batch, n_codebooks, time) or
            tuple of (codes, metadata) if return_metadata is True.
            
        Raises:
            ValueError: If audio format is invalid.
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = torchaudio.load(audio)
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Normalize
        audio_max = torch.tensor(1.0)
        if normalize:
            audio_max = audio.abs().max()
            if audio_max > 0:
                audio = audio / audio_max * 0.95
        
        # Encode using model's compress method (returns codes)
        codes = self.model.compress(audio)
        
        if return_metadata:
            metadata = {
                'sample_rate': self.sample_rate,
                'original_length': audio.shape[-1],
                'normalized': normalize,
                'input_scale': audio_max.item() if normalize else 1.0
            }
            return codes, metadata
        
        return codes
    
    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        denormalize_scale: float | None = None
    ) -> torch.Tensor:
        """Decode discrete codes to audio.
        
        Args:
            codes: Discrete codes of shape (batch, n_codebooks, time).
            denormalize_scale: Scale factor to denormalize audio.
                If provided, output is multiplied by this value.
            
        Returns:
            Decoded audio tensor of shape (batch, channels, time).
        """
        # Move to device
        codes = codes.to(self.device)
        
        # Decode using model's decompress method
        audio = self.model.decompress(codes)
        
        # Denormalize if scale provided
        if denormalize_scale is not None:
            audio = audio * denormalize_scale
        
        return audio
    
    @torch.no_grad()
    def compress(
        self,
        audio: torch.Tensor | np.ndarray | str | Path,
        output_path: str | Path | None = None,
        win_duration: float = 1.0,
        normalize: bool = True,
        verbose: bool = False
    ) -> dict[str, Any] | None:
        """Compress audio using chunked processing.
        
        Args:
            audio: Audio input (tensor, numpy array, or file path).
            output_path: Optional path to save compressed file (.npz format).
            win_duration: Window duration in seconds for chunking long audio.
            normalize: Whether to normalize audio to [-1, 1].
            verbose: Show progress bar during processing.
            
        Returns:
            Dictionary containing 'codes' and 'metadata' if output_path is None,
            otherwise saves to file and returns None.
            
        Raises:
            ValueError: If audio format is unsupported.
        """
        # Load audio if path provided
        original_length = None
        if isinstance(audio, (str, Path)):
            audio, sample_rate = torchaudio.load(audio)
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            sample_rate = self.sample_rate
        else:
            sample_rate = self.sample_rate
        
        # Ensure correct shape: (batch, channels, time)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # (1, C, T)
        
        # Store original info
        batch_size, n_channels, original_length = audio.shape
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        
        # Normalize audio
        input_scale = 1.0
        if normalize:
            input_scale = audio.abs().max().item()
            if input_scale > 0:
                audio = audio / input_scale * 0.95
        
        # Move to device
        audio = audio.to(self.device)
        
        # Calculate chunk parameters
        chunk_samples = int(win_duration * self.sample_rate)
        total_samples = audio.shape[-1]
        
        # Process in chunks if audio is longer than window
        all_codes = []
        if total_samples > chunk_samples:
            # Chunked processing
            hop_samples = chunk_samples // 2  # 50% overlap
            
            for start_idx in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples), 
                                  disable=not verbose, desc="Compressing"):
                end_idx = start_idx + chunk_samples
                chunk = audio[..., start_idx:end_idx]
                
                # Compress chunk
                codes = self.model.compress(chunk)
                all_codes.append(codes)
            
            # Handle last chunk if needed
            if total_samples % hop_samples != 0:
                last_chunk = audio[..., -chunk_samples:]
                codes = self.model.compress(last_chunk)
                all_codes.append(codes)
            
            # Concatenate all codes
            codes = torch.cat(all_codes, dim=-1)
        else:
            # Single chunk processing
            codes = self.model.compress(audio)
        
        # Prepare metadata
        metadata = {
            'original_length': original_length,
            'sample_rate': self.sample_rate,
            'input_scale': input_scale,
            'n_channels': n_channels,
            'win_duration': win_duration,
            'normalize': normalize,
            'chunk_samples': chunk_samples,
            'model_sample_rate': self.model.sample_rate,
            'n_codebooks': self.model.n_codebooks,
            'downsample_rate': self.model.downsample_rate
        }
        
        # Save or return
        if output_path:
            output_path = Path(output_path).with_suffix('.npz')
            np.savez_compressed(
                output_path,
                codes=codes.cpu().numpy(),
                **metadata
            )
            return None
        else:
            return {'codes': codes, 'metadata': metadata}
    
    @torch.no_grad()
    def decompress(
        self,
        compressed: dict[str, Any] | torch.Tensor | str | Path,
        output_path: str | Path | None = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """Decompress audio from codes.
        
        Args:
            compressed: Dictionary with 'codes' and 'metadata', codes tensor,
                or path to .npz file.
            output_path: Optional path to save decompressed audio (.wav).
            verbose: Show progress bar during processing.
            
        Returns:
            Decompressed audio tensor of shape (batch, channels, time).
            
        Raises:
            ValueError: If compressed format is invalid.
        """
        # Load compressed data
        if isinstance(compressed, (str, Path)):
            # Load from .npz file
            data = np.load(compressed)
            codes = torch.from_numpy(data['codes'])
            metadata = {k: data[k].item() if hasattr(data[k], 'item') else data[k] 
                       for k in data.files if k != 'codes'}
        elif isinstance(compressed, dict):
            # Dictionary format
            codes = compressed['codes']
            metadata = compressed['metadata']
        else:
            # Raw codes tensor
            codes = compressed
            metadata = {}
        
        # Ensure codes are on correct device and have batch dimension
        codes = codes.to(self.device)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # Add batch dimension
        
        # Decompress
        audio = self.model.decompress(codes)
        
        # Denormalize if needed
        if metadata.get('normalize', False) and 'input_scale' in metadata:
            audio = audio * metadata['input_scale']
        
        # Trim to original length if specified
        if 'original_length' in metadata:
            audio = audio[..., :metadata['original_length']]
        
        # Save if requested
        if output_path:
            output_path = Path(output_path).with_suffix('.wav')
            # Remove batch dimension if single batch
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            torchaudio.save(
                str(output_path), 
                audio.cpu(), 
                metadata.get('sample_rate', self.sample_rate)
            )
        
        return audio
    
    def batch_process(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        mode: str = "compress",
        win_duration: float = 1.0,
        normalize: bool = True,
        num_workers: int = 0,
        verbose: bool = True
    ) -> list[Path]:
        """Batch process multiple audio files.
        
        Args:
            input_paths: List of input file paths.
            output_dir: Output directory.
            mode: Processing mode - 'compress' or 'decompress'.
            win_duration: Window duration in seconds for compression.
            normalize: Whether to normalize audio during compression.
            num_workers: Number of parallel workers (not implemented, uses sequential).
            verbose: Show progress bar.
            
        Returns:
            List of output file paths.
            
        Raises:
            ValueError: If mode is not 'compress' or 'decompress'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        # Process files
        iterator = tqdm(input_paths, desc=f"Batch {mode}") if verbose else input_paths
        
        for input_path in iterator:
            input_path = Path(input_path)
            
            if mode == "compress":
                output_path = output_dir / input_path.with_suffix('.npz').name
                self.compress(
                    input_path,
                    output_path,
                    win_duration=win_duration,
                    normalize=normalize,
                    verbose=False
                )
            else:  # decompress
                output_path = output_dir / input_path.with_suffix('.wav').name
                self.decompress(
                    input_path,
                    output_path,
                    verbose=False
                )
            
            output_paths.append(output_path)
        
        return output_paths
    
    def evaluate(
        self,
        test_paths: list[str | Path],
        metrics: list[str] = ['sisnr', 'pesq', 'stoi'],
        win_duration: float = 1.0,
        normalize: bool = True,
        verbose: bool = True
    ) -> dict[str, float]:
        """Evaluate codec on test audio files.
        
        Args:
            test_paths: List of test audio file paths.
            metrics: List of metrics to compute. Supported: 'sisnr', 'pesq', 'stoi'.
            win_duration: Window duration in seconds for compression.
            normalize: Whether to normalize audio during compression.
            verbose: Show progress bar.
            
        Returns:
            Dictionary of average metrics and their standard deviations.
        """
        all_metrics = {metric: [] for metric in metrics}
        
        iterator = tqdm(test_paths, desc="Evaluating") if verbose else test_paths
        
        for test_path in iterator:
            # Load audio
            audio, sr = torchaudio.load(test_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            # Add batch dimension
            audio = audio.unsqueeze(0).to(self.device)
            
            # Compress and decompress using pipeline methods
            compressed = self.compress(
                audio,
                win_duration=win_duration,
                normalize=normalize,
                verbose=False
            )
            reconstructed = self.decompress(compressed, verbose=False)
            
            # Ensure same length
            min_len = min(audio.shape[-1], reconstructed.shape[-1])
            audio = audio[..., :min_len]
            reconstructed = reconstructed[..., :min_len]
            
            # Compute metrics
            file_metrics = compute_codec_metrics(
                audio,
                reconstructed,
                sample_rate=self.sample_rate,
                metrics_list=metrics
            )
            
            # Accumulate
            for metric in metrics:
                if metric in file_metrics:
                    all_metrics[metric].append(file_metrics[metric])
        
        # Average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
        
        return avg_metrics


def sliding_window_inference(
    model: HiggsAudioTokenizer,
    audio: torch.Tensor,
    window_size: int,
    hop_size: int,
    device: str | torch.device = "cuda"
) -> torch.Tensor:
    """Perform sliding window inference for very long audio.
    
    Args:
        model: HiggsAudioTokenizer model.
        audio: Long audio tensor of shape (batch, channels, samples).
        window_size: Window size in samples.
        hop_size: Hop size in samples.
        device: Device to run on.
        
    Returns:
        Reconstructed audio tensor with same shape as input.
    """
    batch_size, channels, length = audio.shape
    model = model.to(device)
    model.eval()
    
    # Initialize output
    output = torch.zeros_like(audio)
    counts = torch.zeros((batch_size, 1, length), device=device)
    
    # Sliding window
    with torch.no_grad():
        for start in range(0, length - window_size + 1, hop_size):
            end = start + window_size
            
            # Extract window
            window = audio[:, :, start:end].to(device)
            
            # Process window
            recon = model(window)['audio']
            
            # Accumulate
            output[:, :, start:end] += recon.cpu()
            counts[:, :, start:end] += 1
    
    # Handle final window if needed
    if length % hop_size != 0:
        start = length - window_size
        end = length
        
        window = audio[:, :, start:end].to(device)
        recon = model(window)['audio']
        
        output[:, :, start:end] += recon.cpu()
        counts[:, :, start:end] += 1
    
    # Average overlapping regions
    output = output / counts.clamp(min=1)
    
    return output


def create_inference_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    use_ema: bool = True
) -> HiggsAudioTokenizer:
    """Create model for inference from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
        use_ema: Whether to use EMA weights if available.
        
    Returns:
        HiggsAudioTokenizer model ready for inference.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'config' in checkpoint:
        config = AudioCodecConfig(**checkpoint['config']['tokenizer_config'])
    else:
        warnings.warn("No config in checkpoint, using default 24kHz config")
        config = AudioCodecConfig()
    
    # Create model
    model = HiggsAudioTokenizer(config).to(device)
    
    # Load weights
    if use_ema and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model


# Convenience functions
def encode_audio(
    audio_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
    device: str = "cuda"
) -> dict[str, Any] | None:
    """Encode audio file to codes.
    
    Args:
        audio_path: Path to input audio file.
        model_path: Path to model checkpoint.
        output_path: Optional path to save compressed .npz file.
        device: Device to run on.
        
    Returns:
        Dictionary with 'codes' and 'metadata' if output_path is None,
        otherwise saves to file and returns None.
    """
    pipeline = CodecPipeline(model_path, device=device)
    return pipeline.compress(audio_path, output_path)


def decode_audio(
    codes_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    device: str = "cuda"
) -> torch.Tensor:
    """Decode codes to audio file.
    
    Args:
        codes_path: Path to compressed .npz file.
        model_path: Path to model checkpoint.
        output_path: Path to save decoded .wav file.
        device: Device to run on.
        
    Returns:
        Decoded audio tensor.
    """
    pipeline = CodecPipeline(model_path, device=device)
    return pipeline.decompress(codes_path, output_path)


def reconstruct_audio(
    audio_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    device: str = "cuda"
) -> torch.Tensor:
    """Reconstruct audio through encoding and decoding.
    
    Args:
        audio_path: Path to input audio file.
        model_path: Path to model checkpoint.
        output_path: Path to save reconstructed .wav file.
        device: Device to run on.
        
    Returns:
        Reconstructed audio tensor.
    """
    pipeline = CodecPipeline(model_path, device=device)
    codes = pipeline.encode(audio_path)
    audio = pipeline.decode(codes)
    torchaudio.save(str(output_path), audio.squeeze(0).cpu(), pipeline.sample_rate)
    return audio