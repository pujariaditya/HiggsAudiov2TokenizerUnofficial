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

"""HiggsAudiov2 Audio Tokenizer (Unofficial Implementation).

Neural audio codec with semantic and acoustic modeling for high-quality audio compression.
Based on encoder/decoder architecture from xcodec with semantic feature integration.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence, Dict, Tuple
import numpy as np
from transformers import AutoModel
import torchaudio
import json
import librosa

from vector_quantize_pytorch import ResidualFSQ
from .encoder import Encoder as DACEncoder
from .decoder import Decoder as DACDecoder
from .quantizer import ResidualVectorQuantizer
from .semantic_module import Encoder, Decoder


class EncodedResult:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes


class HiggsAudioFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class HiggsAudioTokenizer(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [0.5, 1, 1.5, 2, 4],
        ratios: Sequence[int] = [8, 5, 4, 2, 3],  # downsampling by 960 (Higgs Audio V2)
        sample_rate: int = 24000,
        bins: Union[int, Sequence[int]] = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_model_path: str = None,  # Required: Path to pretrained semantic model
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        
        # Validate semantic_model_path
        if not semantic_model_path:
            raise ValueError("semantic_model_path is required but not provided")
        
        if not os.path.exists(semantic_model_path):
            raise ValueError(f"semantic_model_path does not exist: {semantic_model_path}")
        
        self.semantic_model_path = semantic_model_path

        # At 24kHz with hop_length=960, we get 25 Hz directly (no division needed)
        # 24000 / 960 = 25 Hz (matching Higgs Audio V2 paper)
        self.frame_rate = sample_rate / np.prod(ratios)  # 25 Hz with 960x downsampling

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        
        # Use DAC-style encoder and decoder
        self.encoder = DACEncoder(n_filters, ratios, D)
        self.decoder_2 = DACDecoder(D, 1024, ratios)
        
        self.last_layer_semantic = last_layer_semantic
        self.device = device
        
        # Initialize semantic model from provided path
        self.semantic_model = AutoModel.from_pretrained(self.semantic_model_path)
        self.semantic_sample_rate = 16000
        self.semantic_dim = 768
        self.encoder_semantic_dim = 768

        # Overwrite semantic model sr to ensure semantic_downsample_factor is an integer
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        self.semantic_model.eval()

        # make the semantic model parameters do not need gradient
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320)

        # Calculate quantizer dimension based on config
        # For official model: D=256 + encoder_semantic_dim=768 = 1024
        self.quantizer_dim = D + self.encoder_semantic_dim
        self.encoder_semantic = Encoder(input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim)
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim, output_channels=self.semantic_dim, decode_channels=self.semantic_dim
        )

        # out_D=D+768
        if isinstance(bins, int):  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)

        self.downsample_mode = downsample_mode
        if downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(sampling_rate=self.sample_rate)

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        # For RVQ, the codebook size is stored in 'bins'
        if hasattr(self.quantizer, 'bins'):
            return self.quantizer.bins
        # For other quantizers that have codebook_size attribute
        elif hasattr(self.quantizer, 'codebook_size'):
            return self.quantizer.codebook_size
        else:
            raise AttributeError(f"Quantizer {type(self.quantizer).__name__} does not have 'bins' or 'codebook_size' attribute")

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x):
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        # Process audio for HuBERT model
        x = x[:, 0, :]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True).hidden_states
        target = torch.stack(target, dim=1)
        
        # Average for all layers
        target = target.mean(1)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, :: self.semantic_downsample_factor, :]

        elif self.downsample_mode == "avg":
            target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)
        return target

    def forward(self, x: torch.Tensor, bw: int = None) -> Dict[str, torch.Tensor]:
        """Forward pass with semantic features.
        
        Args:
            x: Input audio tensor of shape (B, 1, T)
            bw: Target bandwidth
            
        Returns:
            Dictionary with audio, commit_loss, semantic_recon_loss
        """
        # Store original length for trimming later
        original_length = x.shape[-1]
        
        # Pad input to be a multiple of hop_length
        pad_length = (math.ceil(original_length / self.hop_length) * self.hop_length) - original_length
        if pad_length > 0:
            x = F.pad(x, (0, pad_length))
        
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        # Ensure temporal alignment before concatenation (similar to higgs-audio)
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            # First try re-encoding with padding
            if hasattr(self, 'semantic_downsample_factor') and self.semantic_downsample_factor > 0:
                pad_size = 160 * self.semantic_downsample_factor
                x_padded = F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(1)
                e_acoustic_padded = self.encoder(x_padded)
                
                # Check if padding helped
                if e_acoustic_padded.shape[2] == e_semantic.shape[2]:
                    e_acoustic = e_acoustic_padded
                    # Also update the semantic input for loss computation
                    e_semantic_input = self.get_regress_target(x_padded).detach()
        
        # If still mismatched, trim to minimum length
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, :e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, :e_acoustic.shape[2]]
                # Also trim semantic input for loss computation
                e_semantic_input = e_semantic_input[:, :e_acoustic.shape[2], :]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0)

        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)

        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        # Trim output to original length
        o = o[..., :original_length]
        
        return {
            "audio": o,
            "z": e_acoustic,
            "z_q": quantized_acoustic,
            "codes": codes,
            "vq/commitment_loss": commit_loss,
            "semantic_recon_loss": semantic_recon_loss
        }

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.sampling_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv, sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)
        else:
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)

        # return codes
        return EncodedResult(codes)

    def decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_code = vq_code.to(self.device)

        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o.detach().cpu().numpy()
    
    def compress(self, audio: torch.Tensor) -> torch.Tensor:
        """Compress audio to discrete codes.
        
        Args:
            audio: Audio tensor of shape (B, 1, T) or (B, T)
            
        Returns:
            Codes tensor of shape (B, N, T') where N is number of codebooks
        """
        with torch.no_grad():
            result = self._xcodec_encode(audio)
            return result.audio_codes
    
    def decompress(self, codes: torch.Tensor) -> torch.Tensor:
        """Decompress discrete codes to audio.
        
        Args:
            codes: Codes tensor of shape (B, N, T') where N is number of codebooks
            
        Returns:
            Audio tensor of shape (B, 1, T)
        """
        with torch.no_grad():
            audio_np = self.decode(codes)
            return torch.from_numpy(audio_np).to(codes.device)