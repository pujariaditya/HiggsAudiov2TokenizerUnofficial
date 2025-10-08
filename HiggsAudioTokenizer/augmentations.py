"""Audio augmentation pipeline for HiggsAudiov2 (Unofficial Implementation).

Provides audio augmentation techniques for training including:
- Volume normalization
- Audio rescaling
- Phase shifting
- Random gain adjustment
"""

import logging
import random
from typing import Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class AugmentationSequential(nn.Module):
    """Sequential container for augmentations that passes sample_rate."""
    
    def __init__(self, *augmentations):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
    
    def forward(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        for augmentation in self.augmentations:
            audio = augmentation(audio, sample_rate)
        return audio


class BaseAugmentation(nn.Module):
    """Base class for all audio augmentations with probability control."""
    
    def __init__(self, p: float = 1.0, rng: Optional[random.Random] = None, seed: int = 42):
        super().__init__()
        self.p = p
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(self.seed)
        # Generator for th operations
        self.rngth = None
    
    def forward(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        if self.training and self.rng.random() < self.p:
            return self.apply_augmentation(audio, sample_rate)
        return audio
    
    def apply_augmentation(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply the actual augmentation. To be implemented by subclasses."""
        raise NotImplementedError


class VolumeNorm(BaseAugmentation):
    """Normalize audio volume to a target dB level."""
    
    def __init__(
        self,
        target_db: float = -16.0,
        p: float = 1.0,
        rng: Optional[random.Random] = None,
        seed: int = 42,
    ):
        super().__init__(p, rng, seed)
        self.target_db = target_db
    
    def apply_augmentation(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Calculate current RMS in dB
        eps = 1e-8
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        current_db = 20 * torch.log10(rms + eps)
        
        # Calculate gain needed
        gain_db = self.target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        return audio * gain


class RescaleAudio(BaseAugmentation):
    """Rescale audio to ensure it stays within specified bounds."""
    
    def __init__(
        self,
        max_val: float = 1.0,
        p: float = 1.0,
        rng: Optional[random.Random] = None,
        seed: int = 42,
    ):
        super().__init__(p, rng, seed)
        self.max_val = max_val
    
    def apply_augmentation(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        max_abs = torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
        
        # Only rescale if needed
        scale = torch.where(
            max_abs > self.max_val,
            self.max_val / (max_abs + 1e-8),
            torch.ones_like(max_abs)
        )
        
        return audio * scale


class ShiftPhase(BaseAugmentation):
    """Apply random phase shift in the frequency domain."""
    
    def __init__(
        self,
        max_shift: float = np.pi,
        p: float = 1.0,
        rng: Optional[random.Random] = None,
        seed: int = 42,
    ):
        super().__init__(p, rng, seed)
        self.max_shift = max_shift
    
    def apply_augmentation(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Handle 3D tensor (batch, channels, samples) by processing each channel
        if audio.dim() == 3:
            batch_size, num_channels, num_samples = audio.shape
            # Process each channel separately
            processed_channels = []
            for ch in range(num_channels):
                ch_audio = audio[:, ch, :]  # (batch, samples)
                ch_processed = self._apply_phase_shift(ch_audio)
                processed_channels.append(ch_processed)
            # Stack channels back
            return torch.stack(processed_channels, dim=1)
        else:
            # Handle 2D tensor (batch, samples)
            return self._apply_phase_shift(audio)
    
    def _apply_phase_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply phase shift to 2D audio tensor."""
        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Ensure audio length is compatible
        audio_padded = F.pad(audio, (0, n_fft))
        
        # Apply STFT
        spec = torch.stft(
            audio_padded,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(audio.device),
            return_complex=True
        )
        
        # Generate random phase shift for each batch item
        batch_size = audio.shape[0]
        phase_shift = torch.empty(batch_size, 1, 1, device=audio.device)
        for i in range(batch_size):
            phase_shift[i] = self.rng.uniform(-self.max_shift, self.max_shift)
        
        # Apply phase shift
        spec_shifted = spec * torch.exp(1j * phase_shift)
        
        # Convert back to time domain
        audio_shifted = torch.istft(
            spec_shifted,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(audio.device),
            return_complex=False
        )
        
        # Trim to original length
        return audio_shifted[..., :audio.shape[-1]]


def get_augmentations(rng: Optional[random.Random] = None, seed: int = 42) -> AugmentationSequential:
    """Get default augmentations matching HiggsAudiov2 configuration."""
    if rng is None:
        rng = random.Random(seed)
    
    return AugmentationSequential(
        VolumeNorm(target_db=-16.0, p=1.0, rng=rng),
        RescaleAudio(max_val=1.0, p=1.0, rng=rng),
        ShiftPhase(max_shift=np.pi, p=1.0, rng=rng)
    )


def create_augmentation_pipeline(
    config: Optional[Any] = None,
    rng: Optional[random.Random] = None,
    seed: int = 42,
) -> AugmentationSequential:
    """Create augmentation pipeline for training based on configuration."""
    if rng is None:
        rng = random.Random(seed)
    
    if config is None:
        return get_augmentations(rng=rng)
    
    # Build augmentations from config
    augmentations = []
    
    # Volume normalization
    if hasattr(config, 'volume_norm') and config.volume_norm:
        # Handle the new config format: volume_norm: {db: [const, -16]}
        if isinstance(config.volume_norm, dict) and 'db' in config.volume_norm:
            db_value = config.volume_norm['db']
            if isinstance(db_value, list) and len(db_value) >= 2:
                target_db = db_value[1]  # Get the -16 value
            else:
                target_db = -16.0
        else:
            target_db = -16.0
        augmentations.append(VolumeNorm(target_db=target_db, p=1.0, rng=rng))
    elif hasattr(config, 'volume_norm_prob') and config.volume_norm_prob > 0:
        # Handle the old config format
        target_db = getattr(config, 'volume_norm_target_db', -16.0)
        augmentations.append(VolumeNorm(target_db=target_db, p=config.volume_norm_prob, rng=rng))
    
    # Add default augmentations if not empty
    if not augmentations:
        return get_augmentations(rng=rng)
    
    # Always add RescaleAudio and ShiftPhase (matching HiggsAudiov2)
    augmentations.append(RescaleAudio(max_val=1.0, p=1.0, rng=rng))
    augmentations.append(ShiftPhase(max_shift=np.pi, p=1.0, rng=rng))
    
    return AugmentationSequential(*augmentations)