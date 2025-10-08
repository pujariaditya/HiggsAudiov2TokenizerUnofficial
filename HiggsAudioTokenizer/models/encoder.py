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

"""Audio encoder module for HiggsAudiov2 (Unofficial Implementation).

Encoder network for HiggsAudiov2 audio tokenizer using DAC-style architecture.
Implements downsampling with residual units and channel doubling.
"""

import torch
import torch.nn as nn
import math
from typing import List

from .layers import WNConv1d, Snake1d


def init_weights(m: nn.Module) -> None:
    """Initialize weights for convolutional layers.
    
    Uses truncated normal distribution for weights and zeros for bias.
    
    Args:
        m: Module to initialize (only Conv1d layers are affected)
    """
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    """Residual unit with weight-normalized convolutions.
    
    Implements a residual block with dilated convolutions and Snake activation.
    Uses a 7x1 dilated conv followed by a 1x1 conv for channel mixing.
    
    Args:
        dim: Number of channels
        dilation: Dilation factor for the 7x1 convolution
    """
    
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Output tensor of shape (B, C, T)
        """
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Encoder block with downsampling.
    
    Consists of three residual units with increasing dilation (1, 3, 9)
    followed by a strided convolution for downsampling. The channel dimension
    is doubled during downsampling.
    
    Args:
        dim: Output channel dimension (input is dim//2)
        stride: Downsampling factor
    """
    
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Downsampled tensor of shape (B, C*2, T//stride)
        """
        return self.block(x)


class Encoder(nn.Module):
    """Encoder with automatic channel doubling.
    
    Progressively downsamples audio through multiple encoder blocks,
    doubling channels at each stage. Final projection maps to latent dimension.
    
    Args:
        d_model: Initial channel dimension
        strides: List of downsampling factors for each encoder block
        d_latent: Output latent dimension
    """
    
    def __init__(
        self,
        d_model: int = 64,
        strides: List[int] = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap block into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input audio tensor of shape (B, 1, T)
            
        Returns:
            Encoded latent tensor of shape (B, d_latent, T//prod(strides))
        """
        return self.block(x)


# Keep the old name for compatibility
AudioEncoder = Encoder