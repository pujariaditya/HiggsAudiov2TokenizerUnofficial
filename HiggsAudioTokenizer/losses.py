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

"""Loss functions for HiggsAudiov2 (Unofficial Implementation).

Aligned with audio codec loss implementations.

Recommended loss weights (lambdas) for HiggsAudiov2 training:
    - mel/loss: 100.0        # Mel-spectrogram loss (primary reconstruction loss)
    - adv/feat_loss: 2.0     # GAN feature matching loss
    - adv/gen_loss: 1.0      # GAN generator loss
    - vq/commitment_loss: 0.25  # VQ commitment loss
    - vq/codebook_loss: 1.0     # VQ codebook loss

Note: STFT loss is computed but not included in the total loss by default.
It can be used for monitoring or added with appropriate weight if desired.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Dict, Optional, Tuple


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for audio quality
    
    Computes STFT at multiple scales to capture both local and global
    audio characteristics. Essential for high-quality audio reconstruction.
    """
    
    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        overlap: float = 0.75,
        eps: float = 1e-5,
        clamp_eps: Optional[float] = None,  # Alias for eps for compatibility
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0
    ):
        super().__init__()
        
        self.window_lengths = window_lengths
        self.overlap = overlap
        self.eps = clamp_eps if clamp_eps is not None else eps
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.pow = pow
        
        # Precompute hop sizes (window_length // 4)
        self.hop_sizes = [w // 4 for w in self.window_lengths]
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-scale STFT loss
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            Dictionary with total loss and per-scale losses
        """
        # Remove channel dimension for STFT
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        losses = []
        
        for window_length, hop_size in zip(self.window_lengths, self.hop_sizes):
            # Compute STFT
            pred_stft = torch.stft(
                pred,
                n_fft=window_length,
                hop_length=hop_size,
                window=torch.hann_window(window_length).to(pred.device),
                return_complex=True
            )
            
            target_stft = torch.stft(
                target,
                n_fft=window_length,
                hop_length=hop_size,
                window=torch.hann_window(window_length).to(target.device),
                return_complex=True
            )
            
            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            # Apply power before computing losses
            pred_mag_pow = pred_mag.pow(self.pow)
            target_mag_pow = target_mag.pow(self.pow)
            
            mag_loss = F.l1_loss(pred_mag_pow, target_mag_pow)
            
            # Log magnitude loss (perceptually weighted) - using log10
            pred_log_mag = torch.log10(pred_mag_pow + self.eps)
            target_log_mag = torch.log10(target_mag_pow + self.eps)
            log_mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
            
            # Combine losses for this scale
            scale_loss = self.mag_weight * mag_loss + self.log_weight * log_mag_loss
            losses.append(scale_loss)
        
        # Average across scales
        total_loss = sum(losses) / len(losses)
        
        return {
            "total": total_loss,
            "per_scale": losses
        }


class MelSpectrogramLoss(nn.Module):
    """Mel-spectrogram loss for perceptual audio quality
    
    Uses mel-scale spectrograms which better match human perception.
    Aligned with descript-audio-codec implementation.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        win_lengths: List[int] = [2048, 512],
        n_mels: List[int] = [150, 80],
        pow: float = 2.0,
        normalized: bool = False,
        eps: float = 1e-5,
        mag_weight: float = 0.0,
        fmin: Optional[List[float]] = None,
        fmax: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.pow = pow
        self.normalized = normalized
        self.eps = eps
        self.mag_weight = mag_weight
        
        # Default fmin/fmax if not provided
        if fmin is None:
            fmin = [0.0] * len(win_lengths)
        if fmax is None:
            fmax = [sample_rate / 2] * len(win_lengths)
        
        # Create mel transforms for each scale
        self.mel_transforms = nn.ModuleList()
        
        for i, (win_length, n_mel) in enumerate(zip(win_lengths, n_mels)):
            hop_length = win_length // 4
            
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_length,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mel,
                f_min=fmin[i],
                f_max=fmax[i],
                power=pow,
                normalized=normalized,
                norm="slaney" if normalized else None,
                mel_scale="slaney"
            )
            self.mel_transforms.append(mel_transform)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-scale mel-spectrogram loss
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            Total mel loss
        """
        loss = 0.0
        
        for mel_transform in self.mel_transforms:
            # Ensure mel transform is on the same device
            mel_transform = mel_transform.to(pred.device)
            
            # Compute mel spectrograms
            pred_mel = mel_transform(pred)
            target_mel = mel_transform(target)
            
            # Clamp to avoid log of zero
            pred_mel = torch.clamp(pred_mel, min=self.eps)
            target_mel = torch.clamp(target_mel, min=self.eps)
            
            # Log mel loss - using log10
            pred_log_mel = torch.log10(pred_mel)
            target_log_mel = torch.log10(target_mel)
            mel_loss = F.l1_loss(pred_log_mel, target_log_mel)
            
            # Optional magnitude loss
            if self.mag_weight > 0:
                mag_loss = F.l1_loss(pred_mel, target_mel)
                mel_loss = mel_loss + self.mag_weight * mag_loss
            
            loss += mel_loss
        
        return loss / len(self.mel_transforms)


class GANLoss(nn.Module):
    """GAN losses for adversarial training
    
    Implements least-squares GAN loss formulation as used in descript-audio-codec.
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def discriminator_loss(
        real_scores: List[List[torch.Tensor]],
        fake_scores: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Compute discriminator loss using least-squares formulation
        
        Args:
            real_scores: Discriminator outputs for real audio (list of feature maps)
            fake_scores: Discriminator outputs for fake audio (list of feature maps)
            
        Returns:
            Total discriminator loss
        """
        loss = 0.0
        
        for real_out, fake_out in zip(real_scores, fake_scores):
            # Get the last element (final output) from each discriminator
            real_score = real_out[-1]
            fake_score = fake_out[-1]
            
            # Least-squares loss
            real_loss = torch.mean((real_score - 1) ** 2)
            fake_loss = torch.mean(fake_score ** 2)
            loss += real_loss + fake_loss
        
        return loss / len(real_scores)
    
    @staticmethod
    def generator_loss(
        fake_scores: List[List[torch.Tensor]],
        real_scores: Optional[List[List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute generator adversarial loss
        
        Args:
            fake_scores: Discriminator outputs for fake audio (list of feature maps)
            real_scores: Discriminator outputs for real audio (optional, for feature matching)
            
        Returns:
            Tuple of (generator adversarial loss, feature matching loss if real_scores provided)
        """
        loss = 0.0
        
        for fake_out in fake_scores:
            # Get the last element (final output) from each discriminator
            fake_score = fake_out[-1]
            
            # Least-squares loss - generator wants discriminator to output 1
            loss += torch.mean((fake_score - 1) ** 2)
        
        gen_loss = loss / len(fake_scores)
        
        # Compute feature matching loss if real scores provided
        feat_loss = None
        if real_scores is not None:
            feat_loss = GANLoss.feature_matching_loss(real_scores, fake_scores)
        
        return gen_loss, feat_loss
    
    @staticmethod
    def feature_matching_loss(
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Compute feature matching loss
        
        Args:
            real_features: List of discriminator features for real audio
            fake_features: List of discriminator features for fake audio
            
        Returns:
            Feature matching loss
        """
        loss = 0.0
        
        # Iterate through each discriminator's outputs
        for real_out, fake_out in zip(real_features, fake_features):
            # Compare all feature maps except the last one (which is the final score)
            for real_feat, fake_feat in zip(real_out[:-1], fake_out[:-1]):
                loss += F.l1_loss(fake_feat, real_feat.detach())
        
        # Average over all features
        n_feats = sum(len(out) - 1 for out in real_features)
        return loss / n_feats if n_feats > 0 else loss


class L1Loss(nn.L1Loss):
    """L1 Loss for waveform comparison

    Simple wrapper around torch L1Loss for consistency.
    Can compare raw waveforms or any tensor attributes.
    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss between predictions and targets
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            L1 loss value
        """
        return super().forward(pred, target)


class SISDRLoss(nn.Module):
    """Scale-Invariant Source-to-Distortion Ratio loss
    
    Computes SI-SDR between estimated and reference audio signals.
    Commonly used for audio quality evaluation.
    """
    
    def __init__(
        self,
        scaling: bool = True,
        reduction: str = "mean",
        zero_mean: bool = True,
        clip_min: Optional[float] = None,
        weight: float = 1.0,
    ):
        super().__init__()
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SI-SDR loss
        
        Args:
            pred: Predicted audio (batch, 1, time)
            target: Target audio (batch, 1, time)
            
        Returns:
            SI-SDR loss value
        """
        eps = 1e-8
        
        # Reshape to (batch, time, 1)
        nb = target.shape[0]
        references = target.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = pred.reshape(nb, 1, -1).permute(0, 2, 1)
        
        # Zero mean if requested
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0
        
        _references = references - mean_reference
        _estimates = estimates - mean_estimate
        
        # Compute scale factor
        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps
        
        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )
        
        # Compute SDR
        e_true = scale * _references
        e_res = _estimates - e_true
        
        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)
        
        # Clip if requested
        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)
        
        # Reduce
        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        
        return sdr


class SemanticLoss(nn.Module):
    """Semantic feature matching loss for audio tokenizer
    
    This loss ensures the encoded representations preserve semantic information 
    from the teacher model (HuBERT, WavLM, etc.) by matching features between
    the predicted and target semantic representations.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = False,
        weight: float = 10.0,
        detach_target: bool = True
    ):
        """Initialize semantic loss
        
        Args:
            loss_type: Type of loss to use ("mse", "l1", "cosine")
            normalize: Whether to normalize features before computing loss
            weight: Loss weight for combining with other losses
            detach_target: Whether to detach target features (stop gradient)
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.weight = weight
        self.detach_target = detach_target
        
        if loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:  # default to MSE
            self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute semantic feature matching loss
        
        Args:
            pred_features: Predicted semantic features (B, T, D) or (B, D, T)
            target_features: Target semantic features from teacher model
            
        Returns:
            Semantic loss value
        """
        if self.detach_target:
            target_features = target_features.detach()
        
        # Ensure same shape - handle different dimensions
        if pred_features.shape != target_features.shape:
            # Handle different time dimensions (can be at axis 1 or 2)
            if len(pred_features.shape) == 3 and len(target_features.shape) == 3:
                # Assume shape is (B, T, D) or (B, D, T)
                # Check which dimension is time by comparing with expected feature dimension
                if pred_features.shape[1] != target_features.shape[1]:
                    # Time is at dim 1
                    min_t = min(pred_features.shape[1], target_features.shape[1])
                    pred_features = pred_features[:, :min_t, :]
                    target_features = target_features[:, :min_t, :]
                elif pred_features.shape[2] != target_features.shape[2]:
                    # Time is at dim 2
                    min_t = min(pred_features.shape[2], target_features.shape[2])
                    pred_features = pred_features[:, :, :min_t]
                    target_features = target_features[:, :, :min_t]
        
        # Normalize if requested
        if self.normalize:
            pred_features = F.normalize(pred_features, p=2, dim=-1)
            target_features = F.normalize(target_features, p=2, dim=-1)
        
        # Compute loss
        if self.loss_type == "cosine":
            # Flatten for cosine similarity
            batch_size = pred_features.shape[0]
            pred_flat = pred_features.reshape(batch_size, -1)
            target_flat = target_features.reshape(batch_size, -1)
            # Create labels (all positive pairs)
            labels = torch.ones(batch_size, dtype=torch.long, device=pred_features.device)
            loss = self.loss_fn(pred_flat, target_flat, labels)
        else:
            loss = self.loss_fn(pred_features, target_features)
        
        return loss * self.weight


class VectorQuantizationLoss(nn.Module):
    """Vector Quantization losses wrapper
    
    Handles both commitment loss and codebook loss from the quantizer.
    Compatible with both RVQ (Residual Vector Quantization) and FSQ quantizers.
    """
    
    def __init__(
        self,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        use_l2: bool = False
    ):
        """Initialize VQ loss wrapper
        
        Args:
            commitment_weight: Weight for commitment loss (encoder penalty)
            codebook_weight: Weight for codebook loss (quantizer penalty)
            use_l2: Whether to use L2 loss instead of L1 for commitment
        """
        super().__init__()
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.use_l2 = use_l2
    
    def forward(
        self,
        quantized: torch.Tensor,
        encoded: torch.Tensor,
        commitment_loss: Optional[torch.Tensor] = None,
        codebook_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute VQ losses
        
        Args:
            quantized: Quantized representations
            encoded: Original encoded representations (before quantization)
            commitment_loss: Pre-computed commitment loss from quantizer
            codebook_loss: Pre-computed codebook loss from quantizer
            
        Returns:
            Dictionary with individual losses and total VQ loss
        """
        losses = {}
        
        # Commitment loss - penalizes encoder for producing representations far from codebook
        if commitment_loss is None:
            if self.use_l2:
                commitment_loss = F.mse_loss(encoded, quantized.detach())
            else:
                commitment_loss = F.l1_loss(encoded, quantized.detach())
        
        losses["vq/commitment_loss"] = commitment_loss * self.commitment_weight
        
        # Codebook loss - encourages codebook to cover encoder representations
        if codebook_loss is None:
            if self.use_l2:
                codebook_loss = F.mse_loss(quantized, encoded.detach())
            else:
                codebook_loss = F.l1_loss(quantized, encoded.detach())
        
        losses["vq/codebook_loss"] = codebook_loss * self.codebook_weight
        
        # Total VQ loss
        losses["vq/total_loss"] = losses["vq/commitment_loss"] + losses["vq/codebook_loss"]
        
        return losses


class HiggsAudioTokenizerLoss(nn.Module):
    """Combined loss for audio tokenizer training
    
    Implements a comprehensive loss function that includes:
    - Reconstruction losses (Mel-spectrogram + optional STFT)
    - Semantic feature matching loss
    - Vector quantization losses
    - GAN losses (when training with discriminator)
    """
    
    def __init__(
        self,
        # Reconstruction losses
        mel_loss_weight: float = 100.0,
        stft_loss_weight: float = 0.0,  # Optional, not used by default
        waveform_loss_weight: float = 0.0,  # Optional L1 waveform loss
        
        # Semantic loss
        semantic_loss_weight: float = 10.0,
        semantic_loss_type: str = "mse",
        
        # VQ losses
        vq_commitment_weight: float = 0.25,
        vq_codebook_weight: float = 1.0,
        
        # GAN losses
        adv_gen_loss_weight: float = 1.0,
        adv_feat_loss_weight: float = 2.0,
        
        # Loss configurations
        sample_rate: int = 24000,
        device: str = "cuda"
    ):
        """Initialize combined tokenizer loss
        
        Args:
            mel_loss_weight: Weight for mel-spectrogram loss
            stft_loss_weight: Weight for multi-scale STFT loss
            waveform_loss_weight: Weight for L1 waveform loss
            semantic_loss_weight: Weight for semantic feature matching
            semantic_loss_type: Type of semantic loss ("mse", "l1", "cosine")
            vq_commitment_weight: Weight for VQ commitment loss
            vq_codebook_weight: Weight for VQ codebook loss
            adv_gen_loss_weight: Weight for adversarial generator loss
            adv_feat_loss_weight: Weight for feature matching loss
            sample_rate: Audio sample rate
            device: Device for computation
        """
        super().__init__()
        
        # Store weights
        self.mel_loss_weight = mel_loss_weight
        self.stft_loss_weight = stft_loss_weight
        self.waveform_loss_weight = waveform_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.semantic_loss_type = semantic_loss_type
        self.adv_gen_loss_weight = adv_gen_loss_weight
        self.adv_feat_loss_weight = adv_feat_loss_weight
        
        # Initialize loss modules
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_fft=2048,
            win_lengths=[2048, 512],
            n_mels=[150, 80],
            pow=2.0,
            normalized=False
        )
        
        if stft_loss_weight > 0:
            self.stft_loss = MultiScaleSTFTLoss(
                window_lengths=[2048, 512],
                mag_weight=1.0,
                log_weight=1.0
            )
        
        if waveform_loss_weight > 0:
            self.waveform_loss = L1Loss()
        
        self.semantic_loss = SemanticLoss(
            loss_type=self.semantic_loss_type,
            weight=1.0  # Weight applied separately in forward()
        )
        
        self.vq_loss = VectorQuantizationLoss(
            commitment_weight=vq_commitment_weight,
            codebook_weight=vq_codebook_weight
        )
        
        self.gan_loss = GANLoss()
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_audio: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        discriminator_outputs: Optional[Dict[str, List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined tokenizer loss
        
        Args:
            model_output: Dictionary from model forward pass containing:
                - "audio": Reconstructed audio
                - "vq/commitment_loss": VQ commitment loss
                - "semantic_recon_loss": Semantic reconstruction loss
                - "z": Encoded features (optional)
                - "z_q": Quantized features (optional)
            target_audio: Original audio tensor
            semantic_features: Target semantic features (if not in model_output)
            discriminator_outputs: Dict with "real" and "fake" discriminator outputs
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains all individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # Reconstruction losses
        pred_audio = model_output["audio"]
        
        # Mel-spectrogram loss (primary reconstruction loss)
        mel_loss = self.mel_loss(pred_audio, target_audio)
        losses["mel/loss"] = mel_loss
        total_loss += self.mel_loss_weight * mel_loss
        
        # Optional STFT loss
        if self.stft_loss_weight > 0:
            stft_loss_dict = self.stft_loss(pred_audio, target_audio)
            losses["stft/loss"] = stft_loss_dict["total"]
            total_loss += self.stft_loss_weight * stft_loss_dict["total"]
        
        # Optional waveform L1 loss
        if self.waveform_loss_weight > 0:
            waveform_loss = self.waveform_loss(pred_audio, target_audio)
            losses["waveform/loss"] = waveform_loss
            total_loss += self.waveform_loss_weight * waveform_loss
        
        # Semantic loss
        if "semantic_recon_loss" in model_output:
            semantic_loss = model_output["semantic_recon_loss"]
        elif semantic_features is not None and "semantic_pred" in model_output:
            semantic_loss = self.semantic_loss(
                model_output["semantic_pred"],
                semantic_features
            )
        else:
            semantic_loss = torch.tensor(0.0, device=pred_audio.device)
        
        losses["semantic/loss"] = semantic_loss
        total_loss += self.semantic_loss_weight * semantic_loss
        
        # VQ losses
        if "vq/commitment_loss" in model_output:
            vq_losses = self.vq_loss(
                quantized=model_output.get("z_q"),
                encoded=model_output.get("z"),
                commitment_loss=model_output["vq/commitment_loss"],
                codebook_loss=model_output.get("vq/codebook_loss")
            )
            losses.update(vq_losses)
            total_loss += vq_losses["vq/total_loss"]
        
        # GAN losses (if discriminator outputs provided)
        if discriminator_outputs is not None:
            # Generator adversarial loss
            gen_loss, feat_loss = self.gan_loss.generator_loss(
                discriminator_outputs["fake"],
                discriminator_outputs.get("real")
            )
            
            losses["adv/gen_loss"] = gen_loss
            total_loss += self.adv_gen_loss_weight * gen_loss
            
            if feat_loss is not None:
                losses["adv/feat_loss"] = feat_loss
                total_loss += self.adv_feat_loss_weight * feat_loss
        
        losses["total_loss"] = total_loss
        
        return total_loss, losses
    
    def discriminator_loss(
        self,
        real_scores: List[List[torch.Tensor]],
        fake_scores: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute discriminator loss
        
        Args:
            real_scores: Discriminator outputs for real audio
            fake_scores: Discriminator outputs for fake audio
            
        Returns:
            Tuple of (discriminator_loss, loss_dict)
        """
        disc_loss = self.gan_loss.discriminator_loss(real_scores, fake_scores)
        
        return disc_loss, {"disc/loss": disc_loss}


# Usage Examples:

"""
Example 1: Basic tokenizer training without GAN
```python
from HiggsAudioTokenizer.losses import HiggsAudioTokenizerLoss
from HiggsAudioTokenizer.models.audiotokenizer import HiggsAudioTokenizer

# Initialize loss function
criterion = HiggsAudioTokenizerLoss(
    mel_loss_weight=100.0,
    semantic_loss_weight=10.0,
    vq_commitment_weight=0.25,
    vq_codebook_weight=1.0,
    sample_rate=24000
)

# Forward pass through model
model = HiggsAudioTokenizer()
output = model(audio_batch)

# Compute loss
total_loss, loss_dict = criterion(
    model_output=output,
    target_audio=audio_batch
)

# Backpropagate
total_loss.backward()
```

Example 2: Training with discriminator
```python
# Initialize losses
criterion = HiggsAudioTokenizerLoss(
    mel_loss_weight=100.0,
    semantic_loss_weight=10.0,
    adv_gen_loss_weight=1.0,
    adv_feat_loss_weight=2.0
)

# Generator step
output = model(audio_batch)
fake_scores = discriminator(output["audio"])
real_scores = discriminator(audio_batch)

total_loss, loss_dict = criterion(
    model_output=output,
    target_audio=audio_batch,
    discriminator_outputs={
        "fake": fake_scores,
        "real": real_scores
    }
)

# Discriminator step
disc_loss, disc_loss_dict = criterion.discriminator_loss(
    real_scores=real_scores,
    fake_scores=fake_scores
)
```

Example 3: Using individual loss components
```python
from HiggsAudioTokenizer.losses import SemanticLoss, MelSpectrogramLoss

# Semantic loss only
semantic_criterion = SemanticLoss(
    loss_type="cosine",
    normalize=True,
    weight=1.0
)

# Mel loss only  
mel_criterion = MelSpectrogramLoss(
    sample_rate=24000,
    win_lengths=[2048, 512],
    n_mels=[150, 80]
)
```
"""
