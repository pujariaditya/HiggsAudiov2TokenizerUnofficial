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

"""Audio codec evaluation metrics for HiggsAudiov2 (Unofficial Implementation).

Provides evaluation metrics for audio quality assessment including:
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- SNR (Signal-to-Noise Ratio)
- LSD (Log-Spectral Distance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    warnings.warn("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    warnings.warn("STOI not available. Install with: pip install pystoi")


class SISDRLoss(nn.Module):
    def __init__(self, epsilon=1e-8, scaling=True):
        super().__init__()
        self.epsilon = epsilon
        self.scaling = scaling
    
    def forward(self, y_true, y_pred):
        # Ensure equal length
        min_len = min(y_true.shape[-1], y_pred.shape[-1])
        y_true = y_true[..., :min_len]
        y_pred = y_pred[..., :min_len]
        
        # Flatten batch and channel dimensions
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        
        # Zero mean
        y_true = y_true - torch.mean(y_true, dim=-1, keepdim=True)
        y_pred = y_pred - torch.mean(y_pred, dim=-1, keepdim=True)
        
        if self.scaling:
            # Compute optimal scaling factor for SI-SNR
            s_target = torch.sum(y_true * y_pred, dim=-1, keepdim=True) / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + self.epsilon)
            s_target = s_target * y_true
        else:
            # No scaling for regular SNR
            s_target = y_true
        
        # Compute SI-SNR/SNR
        e_noise = y_pred - s_target
        si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + self.epsilon))
        
        # Return negative for use as a loss function
        return -torch.mean(si_snr)


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, 
                 scales=[2048, 512],
                 hop_sizes=None,
                 mag_weight=1.0,
                 log_mag_weight=1.0,
                 pow=2.0):
        super().__init__()
        self.scales = scales
        self.hop_sizes = hop_sizes or [s // 4 for s in scales]
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight
        self.pow = pow
        self.epsilon = 1e-5
    
    def forward(self, y_true, y_pred):
        # Ensure equal length
        min_len = min(y_true.shape[-1], y_pred.shape[-1])
        y_true = y_true[..., :min_len]
        y_pred = y_pred[..., :min_len]
        
        losses = {}
        total_loss = 0.0
        
        for i, (n_fft, hop_length) in enumerate(zip(self.scales, self.hop_sizes)):
            window = torch.hann_window(n_fft).to(y_true.device)
            
            # STFT
            y_true_spec = torch.stft(
                y_true.reshape(-1, y_true.shape[-1]),
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            y_pred_spec = torch.stft(
                y_pred.reshape(-1, y_pred.shape[-1]),
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            
            # Magnitude and log magnitude
            y_true_mag = torch.abs(y_true_spec)
            y_pred_mag = torch.abs(y_pred_spec)
            
            # Magnitude loss
            mag_loss = F.l1_loss(y_true_mag, y_pred_mag)
            losses[f'mag_{n_fft}'] = mag_loss
            total_loss += self.mag_weight * mag_loss
            
            # Log magnitude loss
            y_true_log_mag = torch.log10(y_true_mag.clamp(self.epsilon).pow(self.pow))
            y_pred_log_mag = torch.log10(y_pred_mag.clamp(self.epsilon).pow(self.pow))
            log_mag_loss = F.l1_loss(y_true_log_mag, y_pred_log_mag)
            losses[f'log_mag_{n_fft}'] = log_mag_loss
            total_loss += self.log_mag_weight * log_mag_loss
        
        losses['total'] = total_loss / len(self.scales)
        return losses


class PESQ(nn.Module):
    def __init__(self, sample_rate=16000, mode='wb'):
        super().__init__()
        if not HAS_PESQ:
            raise ImportError("PESQ not available. Install with: pip install pesq")
        
        if sample_rate not in [8000, 16000]:
            raise ValueError("PESQ only supports 8kHz or 16kHz")
        
        self.sample_rate = sample_rate
        self.mode = mode
    
    def forward(self, y_true, y_pred):
        # Convert to numpy
        y_true_np = y_true.squeeze(1).cpu().numpy()
        y_pred_np = y_pred.squeeze(1).cpu().numpy()
        
        scores = []
        for i in range(y_true_np.shape[0]):
            try:
                score = pesq(self.sample_rate, y_true_np[i], y_pred_np[i], self.mode)
                scores.append(score)
            except Exception as e:
                # PESQ can fail on some inputs
                warnings.warn(f"PESQ failed for sample {i}: {e}")
                scores.append(1.0)  # Worst score
        
        return torch.tensor(np.mean(scores), device=y_true.device)


class STOI(nn.Module):
    def __init__(self, sample_rate=16000, extended=False):
        super().__init__()
        if not HAS_STOI:
            raise ImportError("STOI not available. Install with: pip install pystoi")
        
        self.sample_rate = sample_rate
        self.extended = extended
    
    def forward(self, y_true, y_pred):
        # Convert to numpy
        y_true_np = y_true.squeeze(1).cpu().numpy()
        y_pred_np = y_pred.squeeze(1).cpu().numpy()
        
        scores = []
        for i in range(y_true_np.shape[0]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Not enough STFT frames", category=RuntimeWarning)
                score = stoi(y_true_np[i], y_pred_np[i], self.sample_rate, extended=self.extended)
            scores.append(score)
        
        return torch.tensor(np.mean(scores), device=y_true.device)


class CodecMetrics(nn.Module):
    def __init__(self,
                 sample_rate=24000,
                 use_pesq=True,
                 use_stoi=True,
                 pesq_sample_rate=16000):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Basic metrics
        self.sisnr = SISDRLoss()
        self.multi_scale_spectral = MultiScaleSpectralLoss()
        
        # Perceptual metrics
        self.use_pesq = use_pesq and HAS_PESQ
        self.use_stoi = use_stoi and HAS_STOI
        
        if self.use_pesq:
            self.pesq = PESQ(sample_rate=pesq_sample_rate)
            self.pesq_sample_rate = pesq_sample_rate
        
        if self.use_stoi:
            self.stoi = STOI(sample_rate=sample_rate)
    
    def forward(self, y_true, y_pred):
        metrics = {}
        
        # Basic metrics
        metrics['sisnr'] = -self.sisnr(y_true, y_pred)  # Negate because SISDRLoss returns negative SI-SNR
        
        # Multi-scale spectral evaluation
        ms_losses = self.multi_scale_spectral(y_true, y_pred)
        metrics['multi_scale_spectral_loss'] = ms_losses['total']
        
        # Perceptual metrics
        if self.use_pesq:
            # Resample if needed
            if self.sample_rate != self.pesq_sample_rate:
                import torchaudio.functional as AF
                y_true_pesq = AF.resample(y_true, self.sample_rate, self.pesq_sample_rate)
                y_pred_pesq = AF.resample(y_pred, self.sample_rate, self.pesq_sample_rate)
            else:
                y_true_pesq = y_true
                y_pred_pesq = y_pred
            
            # PESQ expects mono
            if y_true_pesq.shape[1] > 1:
                y_true_pesq = torch.mean(y_true_pesq, dim=1, keepdim=True)
                y_pred_pesq = torch.mean(y_pred_pesq, dim=1, keepdim=True)
            
            metrics['pesq'] = self.pesq(y_true_pesq, y_pred_pesq)
        
        if self.use_stoi:
            # STOI expects mono
            y_true_stoi = y_true
            y_pred_stoi = y_pred
            if y_true_stoi.shape[1] > 1:
                y_true_stoi = torch.mean(y_true_stoi, dim=1, keepdim=True)
                y_pred_stoi = torch.mean(y_pred_stoi, dim=1, keepdim=True)
            
            metrics['stoi'] = self.stoi(y_true_stoi, y_pred_stoi)
        
        return metrics


def compute_codec_metrics(y_true, y_pred,
                         sample_rate=24000,
                         metrics_list=None):
    if metrics_list is None:
        metrics_list = ['sisnr', 'pesq', 'stoi']
    
    # Initialize CodecMetrics based on requested metrics
    codec_metrics = CodecMetrics(
        sample_rate=sample_rate,
        use_pesq='pesq' in metrics_list,
        use_stoi='stoi' in metrics_list
    )
    
    with torch.no_grad():
        all_metrics = codec_metrics(y_true, y_pred)
    
    # Filter to only requested metrics
    result = {}
    for metric in metrics_list:
        if metric in all_metrics:
            result[metric] = all_metrics[metric].item() if torch.is_tensor(all_metrics[metric]) else all_metrics[metric]
        elif metric == 'multi_scale_spectral_loss' and 'multi_scale_spectral_loss' in all_metrics:
            result[metric] = all_metrics[metric].item() if torch.is_tensor(all_metrics[metric]) else all_metrics[metric]
    
    return result