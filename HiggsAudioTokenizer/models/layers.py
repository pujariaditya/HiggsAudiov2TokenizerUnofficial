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

"""Weight-normalized layers for HiggsAudiov2 (Unofficial Implementation).

Provides custom neural network layers including weight-normalized convolutions
and the Snake activation function used in the HiggsAudiov2 DAC-style architecture.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    """Weight-normalized 1D convolution.

    Creates a weight-normalized Conv1d layer for HiggsAudiov2 (DAC-style architecture).
    Applies weight normalization to stabilize training of deep networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to both sides (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output (default: 1)
        bias: If True, adds a learnable bias (default: True)
        
    Returns:
        Weight-normalized Conv1d module
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized 1D transposed convolution.

    Creates a weight-normalized ConvTranspose1d layer for HiggsAudiov2 (DAC-style architecture).
    Used for upsampling in the decoder. Applies weight normalization for training stability.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of the transposed convolution (default: 1)
        padding: Zero-padding added to both sides (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        bias: If True, adds a learnable bias (default: True)
        dilation: Spacing between kernel elements (default: 1)
        
    Returns:
        Weight-normalized ConvTranspose1d module
    """
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# JIT-compiled snake function for 1.4x speedup
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake activation function for 1D inputs.
    
    Implementation of the Snake activation from:
    'Neural Networks Fail to Learn Periodic Functions and How to Fix It'
    https://arxiv.org/abs/2006.08195
    
    Particularly effective for modeling audio signals due to its
    periodic inductive bias.
    
    Args:
        channels: Number of channels (each channel has its own alpha parameter)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Snake activation.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Activated tensor of shape (B, C, T)
        """
        return snake(x, self.alpha)