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

"""Main training runner for HiggsAudiov2 (Unofficial Implementation).

Implements iteration-based training with GAN losses, distributed training support,
and comprehensive logging.
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .config import AudioCodecConfig, TrainingConfig, Config
from .models.audiotokenizer import HiggsAudioTokenizer
from .losses import MelSpectrogramLoss, MultiScaleSTFTLoss, GANLoss
from .dataset import AudioDataset, collater
from .utils import IterLoader
from .augmentations import get_augmentations
from .optims import (
    get_codec_optimizers_and_schedulers,
    apply_gradient_clipping,
    OptimizerConfig,
    SchedulerConfig
)
from .exceptions import ValidationError
from .logger import setup_logger, MetricLogger
from .metrics import compute_codec_metrics
from .checkpoint_utils import (
    save_model_checkpoint,
    load_checkpoint,
    get_latest_checkpoint
)
from .dist_utils import (
    init_distributed_mode,
    is_main_process,
    get_rank,
    get_world_size,
    main_process
)
from .storage_utils import CloudPath


class CodecRunner:
    """Main runner for audio codec training with GAN losses."""
    
    def __init__(
        self,
        config: Config,
        generator: HiggsAudioTokenizer,
        discriminator: Optional[nn.Module] = None,
        datasets: Dict[str, Any] = None,
        job_id: str = None,
        augmentation_pipeline: Optional[Any] = None
    ):
        """
        Initialize the codec training runner.
        
        Args:
            config: Full configuration object
            generator: Pre-initialized generator model
            discriminator: Optional pre-initialized discriminator for GAN training
            datasets: Dictionary of datasets (train, valid, test)
            job_id: Job ID for this run
            augmentation_pipeline: Optional augmentation pipeline
        """
        self.config = config
        self._generator = generator
        self._discriminator = discriminator
        if datasets is None:
            raise ValidationError("datasets parameter is required")
        self.datasets = datasets
        self.job_id = job_id or time.strftime('%Y%m%d_%H%M%S')
        self.augmentation_pipeline = augmentation_pipeline
        
        # Extract commonly used configs
        self.model_config = config.model
        self.run_config = config.run
        self.loss_config = config.loss
        self.dataset_config = config.dataset
        self.optimizer_config = config.optimizer
        
        # Setup device and distributed
        self.device = self.run_config.device
        self.use_distributed = self.run_config.use_distributed
        self.use_gan = discriminator is not None and self.model_config.use_discriminator
        
        # Move models to device
        self._generator.to(self.device)
        if self._discriminator is not None:
            self._discriminator.to(self.device)
        
        # Wrap models with DDP if distributed
        if self.use_distributed:
            # Convert BatchNorm to SyncBatchNorm for distributed training
            self._generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._generator)
            self.generator = DDP(
                self._generator,
                device_ids=[self.run_config.gpu] if self.run_config.gpu is not None else None,
                find_unused_parameters=False,  # No unused parameters in forward pass
                static_graph=False
            )
            
            if self._discriminator is not None:
                self._discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._discriminator)
                self.discriminator = DDP(
                    self._discriminator,
                    device_ids=[self.run_config.gpu] if self.run_config.gpu is not None else None,
                    find_unused_parameters=False,  # No unused parameters in forward pass
                    static_graph=False
                )
        else:
            self.generator = self._generator
            self.discriminator = self._discriminator
        
        # Create output directory
        self.output_dir = self.run_config.output_dir / self.run_config.experiment_name
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.loggers = setup_logger(
            self.output_dir,
            name=f"higgs_audio_tokenizer_{self.job_id}",
            use_tensorboard=self.run_config.tensorboard_enabled,
            use_wandb=self.run_config.wandb_enabled,
            wandb_project=self.run_config.wandb_project
        )
        
        # EMA not supported in simplified checkpoint system
        self.ema = None
        
        # Setup losses
        # Mel-spectrogram loss (primary reconstruction loss)
        self.mel_loss = MelSpectrogramLoss(**self.loss_config.mel_loss_params)
        self.mel_loss_weight = self.loss_config.mel_loss_weight
        
        # Optional waveform loss
        self.waveform_loss = nn.L1Loss() if self.loss_config.waveform_loss_weight > 0 else None
        self.waveform_loss_weight = self.loss_config.waveform_loss_weight
        
        # Multi-scale STFT loss (for validation)
        self.stft_loss = MultiScaleSTFTLoss(**self.loss_config.stft_loss_params) if self.loss_config.stft_loss_weight > 0 else None
        self.stft_loss_weight = self.loss_config.stft_loss_weight
        
        # VQ loss weights (losses computed in quantizer)
        self.commitment_weight = self.loss_config.commitment_weight
        
        # Semantic loss weight (for HuBERT feature matching)
        self.semantic_loss_weight = self.loss_config.semantic_loss_weight
        
        # GAN losses
        if self.use_gan and self.loss_config.use_gan_loss:
            self.gan_loss = GANLoss()
            self.adv_gen_loss_weight = self.loss_config.adv_gen_loss_weight
            self.adv_feat_loss_weight = self.loss_config.adv_feat_loss_weight
            self.discriminator_update_freq = self.loss_config.discriminator_update_freq
        
        # Setup optimizers and schedulers
        opt_config = get_codec_optimizers_and_schedulers(
            generator=self.generator,
            discriminator=self.discriminator,
            g_lr=self.optimizer_config.g_learning_rate,
            d_lr=self.optimizer_config.d_learning_rate,
            g_betas=tuple(self.optimizer_config.g_betas),
            d_betas=tuple(self.optimizer_config.d_betas),
            g_eps=self.optimizer_config.g_eps,
            d_eps=self.optimizer_config.d_eps,
            g_weight_decay=self.optimizer_config.g_weight_decay,
            d_weight_decay=self.optimizer_config.d_weight_decay,
            scheduler_type=self.optimizer_config.scheduler_type,
            warmup_steps=self.optimizer_config.warmup_steps,
            max_iterations=self.run_config.max_iterations,
            gamma=self.optimizer_config.gamma,
            min_lr=self.optimizer_config.min_lr
        )
        
        self.optimizers = opt_config['optimizers']
        self.schedulers = opt_config['schedulers']
        
        # Setup data
        self.setup_data()
        
        # Setup augmentations
        if self.augmentation_pipeline is None:
            self.augmentations = get_augmentations() if self.config.augmentations.use_augmentation else nn.Identity()
        else:
            self.augmentations = self.augmentation_pipeline
        
        # Training state
        self.iteration = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # AMP scaler
        self.scaler = GradScaler() if self.run_config.amp else None
        
        # Load checkpoint if resuming
        if self.run_config.resume_from:
            self.load_checkpoint(self.run_config.resume_from)
    
    def setup_data(self):
        """Setup data loaders."""
        # Use provided datasets
        if self.datasets:
            # Training data
            train_dataset = self.datasets.get('train')
            if train_dataset:
                # Create sampler for distributed training
                if self.use_distributed:
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                        shuffle=True
                    )
                    shuffle = False  # Sampler handles shuffling
                else:
                    train_sampler = None
                    shuffle = True
                
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.run_config.batch_size_train,
                    sampler=train_sampler,
                    shuffle=shuffle if train_sampler is None else False,
                    num_workers=self.dataset_config.num_workers,
                    collate_fn=collater,  # Use the imported collater function
                    pin_memory=self.dataset_config.pin_memory,
                    drop_last=True,
                    persistent_workers=self.dataset_config.persistent_workers and self.dataset_config.num_workers > 0,
                    prefetch_factor=self.dataset_config.prefetch_factor if self.dataset_config.num_workers > 0 else None
                )
                self.train_sampler = train_sampler
                # Wrap with IterLoader for infinite iteration
                self.train_loader = IterLoader(self.train_loader, use_distributed=self.use_distributed)
            
            # Validation data
            val_dataset = self.datasets.get('valid')
            if val_dataset:
                # Create sampler for distributed validation
                if self.use_distributed:
                    val_sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                        shuffle=False
                    )
                else:
                    val_sampler = None
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.run_config.batch_size_eval,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=self.dataset_config.num_workers,
                    collate_fn=collater,  # Use the imported collater function
                    pin_memory=self.dataset_config.pin_memory,
                    drop_last=False,
                    persistent_workers=self.dataset_config.persistent_workers and self.dataset_config.num_workers > 0,
                    prefetch_factor=self.dataset_config.prefetch_factor if self.dataset_config.num_workers > 0 else None
                )
                self.val_sampler = val_sampler
        else:
            # Create datasets if not provided
            # Training dataset
            train_dataset = AudioDataset(
                manifest_path=self.dataset_config.train_manifest,
                segment_length=self.dataset_config.segment_length,
                sample_rate=self.model_config.sample_rate,
                normalize=True,
                random_segment=True,
                device="cpu"
            )
            
            # Create sampler for distributed training
            if self.use_distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=True
                )
                shuffle = False
            else:
                train_sampler = None
                shuffle = True
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.run_config.batch_size_train,
                sampler=train_sampler,
                shuffle=shuffle if train_sampler is None else False,
                num_workers=self.dataset_config.num_workers,
                collate_fn=train_dataset.collater,  # Use dataset's collater method
                pin_memory=self.dataset_config.pin_memory,
                drop_last=True,
                persistent_workers=self.dataset_config.persistent_workers and self.dataset_config.num_workers > 0,
                prefetch_factor=self.dataset_config.prefetch_factor if self.dataset_config.num_workers > 0 else None
            )
            self.train_sampler = train_sampler
            # Wrap with IterLoader for infinite iteration
            self.train_loader = IterLoader(self.train_loader, use_distributed=self.use_distributed)
            
            # Validation dataset
            val_dataset = AudioDataset(
                manifest_path=self.dataset_config.valid_manifest,
                segment_length=self.dataset_config.segment_length,
                sample_rate=self.model_config.sample_rate,
                normalize=True,
                random_segment=False,
                device="cpu"
            )
            
            # Create sampler for distributed validation
            if self.use_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=False
                )
            else:
                val_sampler = None
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.run_config.batch_size_eval,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                collate_fn=val_dataset.collater,  # Use dataset's collater method
                pin_memory=self.dataset_config.pin_memory,
                drop_last=False,
                persistent_workers=self.dataset_config.persistent_workers and self.dataset_config.num_workers > 0,
                prefetch_factor=self.dataset_config.prefetch_factor if self.dataset_config.num_workers > 0 else None
            )
            self.val_sampler = val_sampler
    
    def train_discriminator_step(self, real_audio: torch.Tensor, 
                                fake_audio: torch.Tensor, 
                                accumulation_step: int = 0) -> Dict[str, float]:
        """
        Train discriminator for one step.
        
        Args:
            real_audio: Real audio samples
            fake_audio: Generated audio samples (detached)
            accumulation_step: Current accumulation step
            
        Returns:
            Dictionary of discriminator losses
        """
        # Only zero grad on first accumulation step
        if accumulation_step == 0:
            self.optimizers['discriminator'].zero_grad()
        
        # Forward pass with autocast for mixed precision
        with autocast(device_type='cuda', enabled=self.run_config.amp):
            real_scores = self.discriminator(real_audio)
            fake_scores = self.discriminator(fake_audio.detach())
            
            # Calculate discriminator loss using least-squares formulation
            total_d_loss = self.gan_loss.discriminator_loss(real_scores, fake_scores)
        
        # Track individual discriminator losses
        d_losses = {
            'd_total_loss': total_d_loss.item()
        }
        
        # Optionally track per-discriminator losses
        for i, (real_out, fake_out) in enumerate(zip(real_scores, fake_scores)):
            real_score = real_out[-1]  # Last element is the final score
            fake_score = fake_out[-1]
            real_loss = torch.mean((real_score - 1) ** 2)
            fake_loss = torch.mean(fake_score ** 2)
            d_losses[f'd_{i}_loss'] = (real_loss + fake_loss).item()
        
        # Scale loss by accumulation steps
        total_d_loss = total_d_loss / self.run_config.accum_grad_iters
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(total_d_loss).backward()
            
            # Only step optimizer on last accumulation step
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                apply_gradient_clipping(
                    self.optimizers['discriminator'],
                    grad_clip_norm=10.0
                )
                self.scaler.step(self.optimizers['discriminator'])
                self.scaler.update()
                # Update discriminator scheduler after optimizer step
                self.schedulers['discriminator'].step()
        else:
            total_d_loss.backward()
            
            # Only step optimizer on last accumulation step
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                apply_gradient_clipping(
                    self.optimizers['discriminator'],
                    grad_clip_norm=10.0
                )
                self.optimizers['discriminator'].step()
                # Update discriminator scheduler after optimizer step
                self.schedulers['discriminator'].step()
        
        return d_losses
    
    def train_generator_step(self, audio_batch: torch.Tensor, 
                            accumulation_step: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train generator (tokenizer) for one step.
        
        Args:
            audio_batch: Batch of audio samples
            accumulation_step: Current accumulation step
            
        Returns:
            Tuple of (reconstructed_audio, losses_dict)
        """
        # Only zero grad on first accumulation step
        if accumulation_step == 0:
            self.optimizers['generator'].zero_grad()
        
        # Apply augmentations
        audio_batch = self.augmentations(audio_batch, self.model_config.sample_rate)
        
        # Forward pass through tokenizer
        with autocast(device_type='cuda', enabled=self.run_config.amp):
            output = self.generator(audio_batch)
            reconstructed = output['audio']
            
            # Initialize losses dict
            g_losses = {}
            total_g_loss = 0.0
            
            # Mel-spectrogram loss (primary reconstruction loss)
            mel_loss = self.mel_loss(reconstructed, audio_batch)
            g_losses['mel_loss'] = mel_loss.item()
            total_g_loss += self.mel_loss_weight * mel_loss
            
            # Optional waveform loss
            if self.waveform_loss is not None:
                waveform_loss = self.waveform_loss(reconstructed, audio_batch)
                g_losses['waveform_loss'] = waveform_loss.item()
                total_g_loss += self.waveform_loss_weight * waveform_loss
            
            # VQ losses (from quantizer output)
            if 'vq/commitment_loss' in output:
                commitment_loss = output['vq/commitment_loss']
                g_losses['vq_commitment_loss'] = commitment_loss.item()
                total_g_loss += self.commitment_weight * commitment_loss
            
            if 'vq/codebook_loss' in output:
                codebook_loss = output['vq/codebook_loss']
                g_losses['vq_codebook_loss'] = codebook_loss.item()
                total_g_loss += codebook_loss
            
            # Semantic loss (HuBERT feature matching for Higgs Audio V2)
            if 'semantic_recon_loss' in output and self.semantic_loss_weight > 0:
                semantic_loss = output['semantic_recon_loss']
                g_losses['semantic_loss'] = semantic_loss.item()
                total_g_loss += self.semantic_loss_weight * semantic_loss
            
            # GAN losses
            if self.use_gan and self.iteration > 0:
                # Wrap discriminator calls in autocast for mixed precision
                with autocast(device_type='cuda', enabled=self.run_config.amp):
                    fake_scores = self.discriminator(reconstructed)
                    
                    # Generator adversarial loss
                    adv_gen_loss, _ = self.gan_loss.generator_loss(fake_scores)
                    g_losses['adv_gen_loss'] = adv_gen_loss.item()
                    total_g_loss += self.adv_gen_loss_weight * adv_gen_loss
                    
                    # Feature matching loss (using forward outputs directly)
                    if self.adv_feat_loss_weight > 0:
                        with torch.no_grad():
                            real_features = self.discriminator(audio_batch)
                        fake_features = self.discriminator(reconstructed)
                        
                        adv_feat_loss = self.gan_loss.feature_matching_loss(real_features, fake_features)
                        g_losses['adv_feat_loss'] = adv_feat_loss.item()
                        total_g_loss += self.adv_feat_loss_weight * adv_feat_loss
            
            g_losses['g_total_loss'] = total_g_loss.item()
        
        # Scale loss by accumulation steps
        total_g_loss = total_g_loss / self.run_config.accum_grad_iters
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(total_g_loss).backward()
            
            # Only step optimizer on last accumulation step
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                apply_gradient_clipping(
                    self.optimizers['generator'],
                    grad_clip_norm=self.optimizer_config.max_grad_norm
                )
                self.scaler.step(self.optimizers['generator'])
                self.scaler.update()
                # Update generator scheduler after optimizer step
                self.schedulers['generator'].step()
        else:
            total_g_loss.backward()
            
            # Only step optimizer on last accumulation step
            if accumulation_step == self.run_config.accum_grad_iters - 1:
                apply_gradient_clipping(
                    self.optimizers['generator'],
                    grad_clip_norm=self.optimizer_config.max_grad_norm
                )
                self.optimizers['generator'].step()
                # Update generator scheduler after optimizer step
                self.schedulers['generator'].step()
        
        # EMA update removed (not supported in simplified checkpoint system)
        
        return reconstructed, g_losses
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Train for one iteration with gradient accumulation.
        
        Returns:
            Dictionary of all losses
        """
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        
        all_losses = {}
        accumulated_g_losses = {}
        accumulated_d_losses = {}
        
        # Gradient accumulation loop
        for accum_step in range(self.run_config.accum_grad_iters):
            # Get batch - IterLoader handles iteration
            batch = next(self.train_loader)
            
            # Extract audio from the new batch format
            audio_batch = batch["raw_wav"].to(self.device)
            # Audio already has channel dimension from collater: (B, 1, T)
            
            # Train discriminator (with update frequency)
            if (self.use_gan and 
                self.iteration > 0 and 
                self.iteration % self.discriminator_update_freq == 0):
                # Generate fake audio
                with torch.no_grad():
                    with autocast(device_type='cuda', enabled=self.run_config.amp):
                        fake_audio = self.generator(audio_batch)['audio']
                
                # Train discriminator
                d_losses = self.train_discriminator_step(audio_batch, fake_audio, accum_step)
                
                # Accumulate losses for logging
                for k, v in d_losses.items():
                    if k in accumulated_d_losses:
                        accumulated_d_losses[k] += v
                    else:
                        accumulated_d_losses[k] = v
            
            # Train generator
            reconstructed, g_losses = self.train_generator_step(audio_batch, accum_step)
            
            # Accumulate losses for logging
            for k, v in g_losses.items():
                if k in accumulated_g_losses:
                    accumulated_g_losses[k] += v
                else:
                    accumulated_g_losses[k] = v
        
        # Average accumulated losses
        for k in accumulated_g_losses:
            all_losses[k] = accumulated_g_losses[k] / self.run_config.accum_grad_iters
        for k in accumulated_d_losses:
            all_losses[k] = accumulated_d_losses[k] / self.run_config.accum_grad_iters
        
        # Add learning rates to losses
        all_losses['lr_g'] = self.optimizers['generator'].param_groups[0]['lr']
        if self.use_gan:
            all_losses['lr_d'] = self.optimizers['discriminator'].param_groups[0]['lr']
        
        # Optionally synchronize losses across processes for monitoring
        # Note: This is not necessary for training but useful for consistent logging
        if self.use_distributed and self.iteration % self.run_config.log_interval == 0:
            # Convert losses to tensors and synchronize
            synced_losses = {}
            for key, value in all_losses.items():
                if key not in ['lr_g', 'lr_d']:  # Don't sync learning rates
                    loss_tensor = torch.tensor(value, device=self.device)
                    dist.all_reduce(loss_tensor)
                    synced_losses[key] = loss_tensor.item() / get_world_size()  # Average across GPUs
                else:
                    synced_losses[key] = value
            return synced_losses
        
        return all_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.run_config.num_valid_batches:
                    break
                
                # Extract audio from the new batch format
                audio_batch = batch["raw_wav"].to(self.device)
                # Add channel dimension back if needed (B, T) -> (B, 1, T)
                if audio_batch.dim() == 2:
                    audio_batch = audio_batch.unsqueeze(1)
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.run_config.amp):
                    output = self.generator(audio_batch)
                    reconstructed = output['audio']
                    
                    # Calculate validation losses
                    total_loss = 0.0
                    
                    # Mel-spectrogram loss
                    mel_loss = self.mel_loss(reconstructed, audio_batch)
                    total_loss += self.mel_loss_weight * mel_loss
                    
                    # Multi-scale STFT loss (for validation)
                    if self.stft_loss_weight > 0:
                        stft_dict = self.stft_loss(reconstructed, audio_batch)
                        stft_loss = stft_dict["total"]
                        total_loss += self.stft_loss_weight * stft_loss
                    
                    # VQ losses
                    if 'vq/commitment_loss' in output:
                        total_loss += self.commitment_weight * output['vq/commitment_loss']
                    if 'vq/codebook_loss' in output:
                        total_loss += output['vq/codebook_loss']
                    
                    # Semantic loss (HuBERT feature matching)
                    if 'semantic_recon_loss' in output and self.semantic_loss_weight > 0:
                        total_loss += self.semantic_loss_weight * output['semantic_recon_loss']
                
                val_losses.append(total_loss.item())
                
                # Compute metrics on all batches
                metrics = compute_codec_metrics(
                    audio_batch,
                    reconstructed,
                    sample_rate=self.model_config.sample_rate,
                    metrics_list=['sisnr', 'pesq', 'stoi']
                )
                val_metrics.append(metrics)
        
        # Average losses and metrics
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {}
        if val_metrics:
            for key in val_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in val_metrics])
        
        # Synchronize metrics across processes if distributed
        if self.use_distributed:
            # Convert to tensors for distributed synchronization
            val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor)
            avg_val_loss = val_loss_tensor.item() / get_world_size()  # Average across GPUs
            
            # Synchronize other metrics
            for key in list(avg_metrics.keys()):
                metric_tensor = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(metric_tensor)
                avg_metrics[key] = metric_tensor.item() / get_world_size()  # Average across GPUs
        
        return {
            'val_loss': avg_val_loss,
            **avg_metrics
        }
    
    def log_iteration(self, losses: Dict[str, float]):
        """Log training iteration."""
        # Update metric logger
        self.loggers.metric_logger.update(**losses)
        
        # Log to TensorBoard
        if hasattr(self.loggers, 'tensorboard'):
            self.loggers.tensorboard.add_metrics(
                losses,
                self.iteration,
                prefix='train'
            )
        
        # Log to WandB
        if hasattr(self.loggers, 'wandb'):
            self.loggers.wandb.log(
                {f'train/{k}': v for k, v in losses.items()},
                step=self.iteration
            )
        
        # Print progress
        if self.iteration % self.run_config.log_interval == 0:
            metric_str = self.loggers.metric_logger.__str__()
            print(f"Iteration {self.iteration}: {metric_str}")
    
    @main_process
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_{self.iteration:08d}.pth"
        
        # Prepare additional objects to save
        objects_to_save = {
            'optimizer': self.optimizers['generator'].state_dict(),
            'scheduler': self.schedulers['generator'].state_dict(),
            'iteration': self.iteration,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.model_dump()
        }
        
        # Add discriminator state if using GAN
        if self.use_gan and self.discriminator is not None:
            discriminator_to_save = self.unwrap_model(self.discriminator) if self.use_distributed else self.discriminator
            objects_to_save['discriminator'] = discriminator_to_save.state_dict()
            objects_to_save['optimizer_d'] = self.optimizers['discriminator'].state_dict()
            objects_to_save['scheduler_d'] = self.schedulers['discriminator'].state_dict()
        
        # Add scaler state if using AMP
        if self.scaler is not None:
            objects_to_save['scaler'] = self.scaler.state_dict()
        
        save_model_checkpoint(
            model=self.generator,
            save_path=checkpoint_path,
            use_distributed=self.run_config.use_distributed,
            drop_untrained_params=False,
            **objects_to_save
        )
        
        # Save JSON config alongside checkpoint
        config_json_path = checkpoint_path.with_suffix('.json')
        self.config.save_model_json(config_json_path)
        
        # Update latest checkpoint symlink/copy
        latest_path = self.output_dir / "checkpoint_latest.pth"
        try:
            # Try to create symlink (preferred)
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_path.name)
        except (OSError, NotImplementedError):
            # Fall back to copying if symlinks not supported
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
        
        # Best model
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pth"
            save_model_checkpoint(
                model=self.generator,
                save_path=best_path,
                use_distributed=self.run_config.use_distributed,
                drop_untrained_params=False,
                iteration=self.iteration,
                epoch=self.epoch,
                config=self.config.model_dump()
            )
            # Also save JSON config for best model
            best_config_json_path = best_path.with_suffix('.json')
            self.config.save_model_json(best_config_json_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last_n: int = 5):
        """Keep only the last N checkpoints."""
        # Get all numbered checkpoints (exclude latest and best)
        checkpoints = []
        for ckpt in self.output_dir.glob("checkpoint_*.pth"):
            # Skip latest and best checkpoints
            if ckpt.name in ["checkpoint_latest.pth", "checkpoint_best.pth"]:
                continue
            # Only include numbered checkpoints
            if ckpt.stem.replace("checkpoint_", "").isdigit():
                checkpoints.append(ckpt)
        
        checkpoints.sort()
        
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
    
    def unwrap_model(self, model):
        """Unwrap model from DDP if needed."""
        if self.use_distributed:
            return model.module
        return model
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        # Unwrap model for loading if using DDP
        generator_to_load = self.unwrap_model(self.generator) if self.use_distributed else self.generator
        
        info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=generator_to_load,
            optimizer=self.optimizers['generator'],
            scheduler=self.schedulers['generator'],
            device=self.device,
            strict=False,
            load_optimizer=True,
            load_scheduler=True
        )
        
        # Load discriminator state if present and using GAN
        if self.use_gan and self.discriminator is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'discriminator' in checkpoint:
                discriminator_to_load = self.unwrap_model(self.discriminator) if self.use_distributed else self.discriminator
                discriminator_to_load.load_state_dict(checkpoint['discriminator'])
            if 'optimizer_d' in checkpoint and self.optimizers.get('discriminator'):
                self.optimizers['discriminator'].load_state_dict(checkpoint['optimizer_d'])
            if 'scheduler_d' in checkpoint and self.schedulers.get('discriminator'):
                self.schedulers['discriminator'].load_state_dict(checkpoint['scheduler_d'])
            
            # Load scaler state if present
            if self.scaler is not None and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.iteration = info['iteration']
        self.epoch = info['epoch']
        self.best_loss = info['best_loss']
        
        print(f"Resumed from iteration {self.iteration}")
    
    @main_process
    def generate_samples(self):
        """Generate and log audio samples."""
        self.generator.eval()
        
        # Determine which samples to use
        if self.run_config.val_idx:
            # Use specific validation indices
            val_samples = []
            val_dataset = self.datasets.get('valid', self.datasets.get('test'))
            for idx in self.run_config.val_idx[:self.run_config.num_audio_samples]:
                if idx < len(val_dataset):
                    sample = val_dataset[idx]
                    val_samples.append(sample["raw_wav"])
            
            if val_samples:
                val_batch = torch.stack(val_samples).to(self.device)
                if val_batch.dim() == 2:
                    val_batch = val_batch.unsqueeze(1)
            else:
                # Fallback to regular batch if indices are invalid
                batch = next(iter(self.val_loader))
                val_batch = batch["raw_wav"].to(self.device)
                if val_batch.dim() == 2:
                    val_batch = val_batch.unsqueeze(1)
                val_batch = val_batch[:self.run_config.num_audio_samples]
        else:
            # Get a validation batch normally
            batch = next(iter(self.val_loader))
            # Extract audio from the new batch format
            val_batch = batch["raw_wav"].to(self.device)
            # Add channel dimension back if needed (B, T) -> (B, 1, T)
            if val_batch.dim() == 2:
                val_batch = val_batch.unsqueeze(1)
            val_batch = val_batch[:self.run_config.num_audio_samples]
        
        with torch.no_grad():
            # Generate reconstructions
            output = self.generator(val_batch)
            reconstructed = output['audio']
        
        # Log to TensorBoard
        if hasattr(self.loggers, 'tensorboard'):
            for i in range(val_batch.shape[0]):
                self.loggers.tensorboard.add_codec_samples(
                    val_batch[i],
                    reconstructed[i],
                    self.iteration,
                    self.model_config.sample_rate,
                    prefix=f"samples_{i}"
                )
        
        # Log to WandB
        if hasattr(self.loggers, 'wandb'):
            for i in range(val_batch.shape[0]):
                self.loggers.wandb.log_codec_comparison(
                    val_batch[i],
                    reconstructed[i],
                    self.iteration,
                    self.model_config.sample_rate
                )
    
    def train(self):
        """Entry point for training - delegates to run()."""
        self.run()
    
    def run(self):
        """Run the main training loop."""
        if is_main_process():
            print(f"Starting training from iteration {self.iteration}")
            print(f"Training for {self.run_config.max_iterations} iterations")
        
        # Training loop
        while self.iteration < self.run_config.max_iterations:
            # Train iteration
            losses = self.train_iteration()
            self.log_iteration(losses)
            
            # Validation
            if self.iteration % self.run_config.validation_interval == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                if hasattr(self.loggers, 'tensorboard'):
                    self.loggers.tensorboard.add_metrics(
                        val_metrics,
                        self.iteration,
                        prefix='val'
                    )
                
                if hasattr(self.loggers, 'wandb'):
                    self.loggers.wandb.log(
                        {f'val/{k}': v for k, v in val_metrics.items()},
                        step=self.iteration
                    )
                
                if is_main_process():
                    print(f"Validation at iteration {self.iteration}: {val_metrics}")
                
                # Check if best model
                val_loss = val_metrics.get(self.run_config.checkpoint_best_metric.replace('/', '_'), val_metrics['val_loss'])
                is_best = False
                
                if self.run_config.checkpoint_best_mode == "min":
                    is_best = val_loss < self.best_loss
                else:
                    is_best = val_loss > self.best_loss
                
                if is_best:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.run_config.early_stop_patience:
                    if is_main_process():
                        print(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                    break
            
            # Save checkpoint
            if self.iteration % self.run_config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Save at specific iterations if configured
            if (self.run_config.save_iters and 
                self.iteration in self.run_config.save_iters):
                self.save_checkpoint()
            
            # Generate samples
            sample_interval = self.run_config.sample_freq or self.run_config.validation_interval
            if (self.run_config.save_audio_samples and 
                self.iteration % sample_interval == 0):
                self.generate_samples()
            
            # Increment iteration
            self.iteration += 1
            
            # Synchronize in distributed training
            if self.use_distributed:
                dist.barrier()
        
        if is_main_process():
            print("Training completed!")
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Cleanup
        if hasattr(self.loggers, 'tensorboard'):
            self.loggers.tensorboard.close()
        if hasattr(self.loggers, 'wandb'):
            self.loggers.wandb.finish()


def main():
    """Main entry point for training."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='higgs-audio-v2-unofficial',
                       help='WandB project name')
    
    # Distributed training arguments
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                       help='Rank of the process')
    parser.add_argument('--dist-url', type=str, default='env://',
                       help='URL for distributed init')
    
    args = parser.parse_args()
    
    # Initialize distributed training
    init_distributed_mode(args)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create configs
    tokenizer_config = AudioCodecConfig(**config.tokenizer)
    training_config = TrainingConfig(**config.training)
    
    # Create discriminator if using GAN training
    discriminator = None
    if config.get('use_gan', False):
        # Import and create discriminator
        # This would need to be implemented based on the specific discriminator architecture
        warnings.warn("GAN training requested but discriminator not implemented")
    
    # Create runner
    runner = CodecRunner(
        tokenizer_config=tokenizer_config,
        training_config=training_config,
        discriminator=discriminator,
        output_dir=args.output_dir,
        resume_from=args.resume,
        use_wandb=args.use_wandb and is_main_process(),
        wandb_project=args.wandb_project,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run training
    runner.run()


if __name__ == '__main__':
    main()