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

"""Optimizer and learning rate scheduler utilities for HiggsAudiov2 (Unofficial Implementation).

Provides optimizer creation with parameter grouping and various scheduling strategies
for both generator and discriminator in GAN-based codec training.
"""

import math
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer

# Only AdamW optimizer is used, matching descript-audio-codec


@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation."""
    name: str = "adamw"  # Only AdamW is supported
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.8, 0.99)  # Default from descript-audio-codec
    weight_decay: float = 0.0  # No weight decay (matching descript-audio-codec)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1000.0  # Gradient clipping by norm
    grad_clip_value: Optional[float] = None  # Gradient clipping by value


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: str = "exponential"  # exponential, cosine, linear, constant
    warmup_steps: int = 0
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    gamma: float = 0.999996  # For exponential
    T_max: Optional[int] = None  # For cosine
    linear_end_lr: float = 0.0  # For linear


def get_parameter_groups(model: nn.Module, weight_decay: float = 0.0,
                        skip_list: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.
    
    Following the standard practice:
    - No weight decay for biases and normalization parameters
    - Regular weight decay for other parameters
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay value
        skip_list: Additional parameter names to skip weight decay
        
    Returns:
        List of parameter groups
    """
    decay = []
    no_decay = []
    
    if skip_list is None:
        skip_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should skip weight decay
        if (len(param.shape) == 1 or  # Bias and norm parameters
            name.endswith(".bias") or
            name.endswith(".b") or
            "ln" in name or
            "norm" in name or
            any(skip_name in name for skip_name in skip_list)):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def create_optimizer(model: nn.Module, config: OptimizerConfig, use_zero: bool = False) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer matching descript-audio-codec approach.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        use_zero: Whether to use ZeroRedundancyOptimizer for distributed training
        
    Returns:
        Configured AdamW optimizer
    """
    # Create AdamW optimizer without parameter grouping (matching descript-audio-codec)
    if use_zero:
        # Use ZeroRedundancyOptimizer for distributed training
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=AdamW,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    
    return optimizer


def create_optimizer_g(model: nn.Module, lr: float = 1e-4, betas: Tuple[float, float] = (0.8, 0.99),
                      weight_decay: float = 0.0, eps: float = 1e-8, use_zero: bool = False) -> torch.optim.Optimizer:
    """Create AdamW optimizer for generator (audio tokenizer)."""
    config = OptimizerConfig(
        name="adamw",
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        grad_clip_norm=1000.0  # Higher clip for generator (matching descript-audio-codec)
    )
    return create_optimizer(model, config, use_zero=use_zero)


def create_optimizer_d(model: nn.Module, lr: float = 1e-4, betas: Tuple[float, float] = (0.8, 0.99),
                      weight_decay: float = 0.0, eps: float = 1e-8, use_zero: bool = False) -> torch.optim.Optimizer:
    """Create AdamW optimizer for discriminator."""
    config = OptimizerConfig(
        name="adamw",
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        grad_clip_norm=10.0  # Lower clip for discriminator (matching descript-audio-codec)
    )
    return create_optimizer(model, config, use_zero=use_zero)


class ExponentialLRScheduler(_LRScheduler):
    """Exponential learning rate decay scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, gamma: float = 0.999996,
                 last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (self.gamma ** self.last_epoch)
                for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup scheduler that can be combined with other schedulers."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 warmup_start_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class LinearWarmupExponentialScheduler(_LRScheduler):
    """Combined linear warmup + exponential decay scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 gamma: float = 0.999996, warmup_start_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Exponential decay after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            return [base_lr * (self.gamma ** steps_after_warmup)
                    for base_lr in self.base_lrs]


class LinearWarmupCosineScheduler(_LRScheduler):
    """Combined linear warmup + cosine annealing scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 T_max: int, min_lr: float = 1e-6, warmup_start_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.T_max = T_max
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            progress = steps_after_warmup / (self.T_max - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) *
                    0.5 * (1.0 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


def create_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig,
                    max_iterations: Optional[int] = None) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        max_iterations: Maximum training iterations (for cosine scheduler)
        
    Returns:
        Configured scheduler
    """
    if config.name == "exponential":
        if config.warmup_steps > 0:
            scheduler = LinearWarmupExponentialScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                gamma=config.gamma,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = ExponentialLRScheduler(optimizer, gamma=config.gamma)
    
    elif config.name == "cosine":
        T_max = config.T_max or max_iterations
        if T_max is None:
            raise ValueError("T_max or max_iterations must be provided for cosine scheduler")
        
        if config.warmup_steps > 0:
            scheduler = LinearWarmupCosineScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                T_max=T_max,
                min_lr=config.min_lr,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=config.min_lr
            )
    
    elif config.name == "linear":
        if config.warmup_steps > 0:
            # Linear warmup followed by linear decay
            lambda_lr = lambda step: (
                config.warmup_start_lr / optimizer.param_groups[0]['lr'] + 
                (1 - config.warmup_start_lr / optimizer.param_groups[0]['lr']) * step / config.warmup_steps
                if step < config.warmup_steps
                else 1.0 - (1.0 - config.linear_end_lr / optimizer.param_groups[0]['lr']) * 
                (step - config.warmup_steps) / (max_iterations - config.warmup_steps)
            )
        else:
            # Just linear decay
            lambda_lr = lambda step: (
                1.0 - (1.0 - config.linear_end_lr / optimizer.param_groups[0]['lr']) * 
                step / max_iterations
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    elif config.name == "constant":
        # Constant learning rate (with optional warmup)
        if config.warmup_steps > 0:
            scheduler = LinearWarmupScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                warmup_start_lr=config.warmup_start_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    else:
        raise ValueError(f"Unknown scheduler: {config.name}")
    
    return scheduler


def get_codec_optimizers_and_schedulers(
    generator: nn.Module,
    discriminator: Optional[nn.Module] = None,
    g_lr: float = 1e-4,
    d_lr: float = 1e-4,
    g_betas: Tuple[float, float] = (0.8, 0.99),
    d_betas: Tuple[float, float] = (0.8, 0.99),
    g_eps: float = 1e-8,
    d_eps: float = 1e-8,
    g_weight_decay: float = 0.0,
    d_weight_decay: float = 0.0,
    scheduler_type: str = "exponential",
    warmup_steps: int = 0,
    max_iterations: Optional[int] = None,
    gamma: float = 0.999996,
    min_lr: float = 1e-6,
    use_zero: bool = False
) -> Dict[str, Any]:
    """
    Get optimizers and schedulers for codec training.
    
    Args:
        generator: Generator model (audio tokenizer)
        discriminator: Optional discriminator model
        g_lr: Generator learning rate
        d_lr: Discriminator learning rate
        g_betas: Generator Adam betas
        d_betas: Discriminator Adam betas
        g_eps: Generator Adam epsilon
        d_eps: Discriminator Adam epsilon
        g_weight_decay: Generator weight decay
        d_weight_decay: Discriminator weight decay
        scheduler_type: Type of scheduler to use
        warmup_steps: Warmup steps
        max_iterations: Maximum iterations
        gamma: Exponential decay rate
        min_lr: Minimum learning rate
        use_zero: Whether to use ZeroRedundancyOptimizer for distributed training
        
    Returns:
        Dictionary with optimizers and schedulers
    """
    # Create optimizers using AdamW only
    optimizer_g = create_optimizer_g(generator, lr=g_lr, betas=g_betas, weight_decay=g_weight_decay, use_zero=use_zero)
    
    optimizers = {'generator': optimizer_g}
    
    if discriminator is not None:
        optimizer_d = create_optimizer_d(discriminator, lr=d_lr, betas=d_betas, weight_decay=d_weight_decay, use_zero=use_zero)
        optimizers['discriminator'] = optimizer_d
    
    # Create schedulers (matching descript-audio-codec)
    scheduler_config = SchedulerConfig(
        name=scheduler_type,
        gamma=gamma,
        warmup_steps=warmup_steps,
        warmup_start_lr=1e-6,
        min_lr=min_lr
    )
    
    scheduler_g = create_scheduler(optimizer_g, scheduler_config, max_iterations)
    schedulers = {'generator': scheduler_g}
    
    if discriminator is not None:
        scheduler_d = create_scheduler(optimizer_d, scheduler_config, max_iterations)
        schedulers['discriminator'] = scheduler_d
    
    return {
        'optimizers': optimizers,
        'schedulers': schedulers
    }


def apply_gradient_clipping(optimizer: torch.optim.Optimizer, 
                          grad_clip_norm: Optional[float] = None,
                          grad_clip_value: Optional[float] = None):
    """
    Apply gradient clipping to optimizer parameters.
    
    Args:
        optimizer: Optimizer with parameters to clip
        grad_clip_norm: Maximum gradient norm (L2)
        grad_clip_value: Maximum gradient value (element-wise)
    """
    if grad_clip_norm is not None:
        params = []
        for group in optimizer.param_groups:
            params.extend(group['params'])
        torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
    
    if grad_clip_value is not None:
        params = []
        for group in optimizer.param_groups:
            params.extend(group['params'])
        torch.nn.utils.clip_grad_value_(params, grad_clip_value)