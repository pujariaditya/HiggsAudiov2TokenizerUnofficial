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

"""Audio decoder module for HiggsAudiov2 (Unofficial Implementation).

Decoder network for HiggsAudiov2 audio tokenizer using DAC-style architecture.
Implements upsampling with transposed convolutions and residual units for refinement.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional

from .layers import WNConv1d, WNConvTranspose1d, Snake1d
from .encoder import ResidualUnit


class DecoderBlock(nn.Module):
    """Decoder block with upsampling.
    
    Performs transposed convolution for upsampling followed by three
    residual units with increasing dilation (1, 3, 9) for refinement.
    Reduces channels during upsampling.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels (typically input_dim//2)
        stride: Upsampling factor
        out_pad: Output padding for transposed convolution
    """
    
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, out_pad: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=out_pad if out_pad > 0 else stride % 2,  # Special case handling
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.
        
        Args:
            x: Input tensor of shape (B, C_in, T)
            
        Returns:
            Upsampled tensor of shape (B, C_out, T*stride)
        """
        return self.block(x)


class Decoder(nn.Module):
    """Decoder for audio generation.
    
    Progressively upsamples latent representations back to audio waveform,
    halving channels at each stage. Uses transposed convolutions with
    residual refinement blocks.
    
    Args:
        input_channel: Number of channels in latent representation
        channels: Initial channel dimension for decoder
        rates: List of upsampling factors for each decoder block
        d_out: Output channels (1 for mono, 2 for stereo)
    """
    
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            # Special case for second block as in higgs-audio
            if i == 1:
                out_pad = 1
            else:
                out_pad = 0
            layers += [DecoderBlock(input_dim, output_dim, stride, out_pad)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            x: Latent tensor of shape (B, input_channel, T)
            
        Returns:
            Reconstructed audio of shape (B, d_out, T*prod(rates))
        """
        return self.model(x)


# Keep the old name for compatibility
AudioDecoder = Decoder