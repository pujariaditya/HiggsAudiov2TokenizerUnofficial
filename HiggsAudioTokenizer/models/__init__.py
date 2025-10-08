# Copyright 2025 DAC-Unofficial Contributors
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

"""DAC (Unofficial Implementation) Model Components

This module provides all the core components for the DAC audio codec:
- Encoder and decoder networks
- Vector quantization modules
- Discriminator networks for adversarial training
- Custom layers and activation functions
"""

from .encoder import Encoder, AudioEncoder
from .decoder import Decoder, AudioDecoder
from .quantizer import ResidualVectorQuantizer
from .layers import WNConv1d, WNConvTranspose1d, Snake1d
from .discriminator import Discriminator, MPD, MSD, MRD
from .audiotokenizer import HiggsAudioTokenizer, HiggsAudioFeatureExtractor
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .core_vq import ResidualVectorQuantization, VectorQuantization, EuclideanCodebook

__all__ = [
    "Encoder", 
    "AudioEncoder",
    "Decoder",
    "AudioDecoder", 
    "ResidualVectorQuantizer",
    "ResidualVectorQuantization",
    "VectorQuantization",
    "EuclideanCodebook",
    "WNConv1d",
    "WNConvTranspose1d",
    "Snake1d",
    "Discriminator",
    "MPD",
    "MSD",
    "MRD",
    "HiggsAudioTokenizer",
    "HiggsAudioFeatureExtractor",
    "SemanticEncoder",
    "SemanticDecoder"
]