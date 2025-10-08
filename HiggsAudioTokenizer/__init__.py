"""Unofficial Descript Audio Codec (DAC) Implementation with HiggsAudio Extensions

Based on the official DAC: https://github.com/descriptinc/dac
Paper: Descript Audio Codec: High-Fidelity Audio Compression with Improved RVQGAN
Enhanced with HiggsAudio semantic features
"""

from .models.audiotokenizer import HiggsAudioTokenizer
from .models.semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .models.quantizer import ResidualVectorQuantizer
from .checkpoint_utils import load_higgs_audio_tokenizer, save_model_pretrained

__all__ = [
    "HiggsAudioTokenizer",
    "load_higgs_audio_tokenizer",
    "save_model_pretrained",
    "SemanticEncoder",
    "SemanticDecoder",
    "ResidualVectorQuantizer",
]