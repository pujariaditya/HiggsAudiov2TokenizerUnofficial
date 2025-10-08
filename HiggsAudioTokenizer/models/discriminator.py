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

"""Discriminator modules for HiggsAudiov2 (Unofficial Implementation).

Discriminators for HiggsAudiov2 audio tokenizer using DAC-style architecture.
Implements Multi-Period, Multi-Scale, and Multi-Resolution discriminators for GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Dict, Tuple, Optional
import numpy as np
from einops import rearrange
import torchaudio.functional as AF

# Import standard WNConv1d from layers
from .layers import WNConv1d as _WNConv1d


def WNConv1d(*args, **kwargs):
    """Weight normalized 1D convolution with optional activation.
    
    Special version for discriminator that adds LeakyReLU activation.
    """
    act = kwargs.pop("act", True)
    conv = _WNConv1d(*args, **kwargs)
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    """Weight normalized 2D convolution"""
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    """Multi-Period Discriminator."""
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    """Multi-Scale Discriminator.
    
    Uses torchaudio's resample function for proper anti-aliased downsampling
    to avoid artifacts that occur with simple average pooling.
    """
    def __init__(self, rate: int = 1, sample_rate: int = 24000):
        super().__init__()
        self.convs = nn.ModuleList([
            WNConv1d(1, 16, 15, 1, padding=7),
            WNConv1d(16, 64, 41, 4, groups=4, padding=20),
            WNConv1d(64, 256, 41, 4, groups=16, padding=20),
            WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
            WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
            WNConv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        # Resample if needed
        if self.rate > 1:
            # x shape: (B, 1, T)
            target_sr = self.sample_rate // self.rate
            
            # Remove channel dimension for resampling
            x = x.squeeze(1)  # (B, T)
            
            # Proper audio resampling with anti-aliasing
            x = AF.resample(
                x,
                orig_freq=self.sample_rate,
                new_freq=target_sr,
                rolloff=0.99,  # Good anti-aliasing (default)
                lowpass_filter_width=6  # Sharp filter (default)
            )
            
            # Add channel dimension back
            x = x.unsqueeze(1)  # (B, 1, T)
        
        fmap = []
        
        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        
        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    """Multi-Resolution Discriminator."""
    def __init__(
        self, 
        window_length: int = 512,
        hop_factor: float = 0.25,
        sample_rate: int = 24000,
        bands: List[Tuple[float, float]] = None
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = int(window_length * hop_factor)
        self.sample_rate = sample_rate
        
        # Default frequency bands (HiggsAudiov2/DAC-style architecture)
        if bands is None:
            bands = BANDS
        self.bands = bands
        
        # Calculate n_fft correctly
        n_fft = window_length // 2 + 1
        self.band_indices = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        
        # Channel count (HiggsAudiov2/DAC-style architecture)
        ch = 32
        
        # Create convolution stacks for each band
        convs = lambda: nn.ModuleList([
            WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        ])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        # x: (B, 1, T)
        x = x.squeeze(1)  # (B, T)
        
        # Compute STFT
        x_stft = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=torch.hann_window(self.window_length, device=x.device),
            return_complex=True,
            center=True,
        )
        
        # Convert complex to real tensor
        x_stft = torch.view_as_real(x_stft)  # (B, F, T, 2)

        # Rearrange to (B, 2, T, F) for HiggsAudiov2/DAC-style format
        x_stft = rearrange(x_stft, "b f t c -> b c t f")
        
        # Split into bands
        x_bands = [x_stft[..., b[0]:b[1]] for b in self.band_indices]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []
        
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        
        return fmap


class Discriminator(nn.Module):
    """Main discriminator class combining MPD, MSD, and MRD."""
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 24000,
        bands: list = None,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            downsampling factors for MSD, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 24000
        bands : list, optional
            Bands to run MRD at, by default BANDS
        """
        super().__init__()
        if bands is None:
            bands = BANDS
            
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdim=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps