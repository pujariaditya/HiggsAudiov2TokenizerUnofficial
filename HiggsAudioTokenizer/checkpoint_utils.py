"""Checkpoint utilities for HiggsAudiov2 (Unofficial Implementation).

Provides utilities for saving and loading model checkpoints during training,
including support for distributed training and cloud storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from HiggsAudioTokenizer.dist_utils import is_main_process
from HiggsAudioTokenizer.exceptions import CheckpointError
from HiggsAudioTokenizer.storage_utils import CloudPath, is_cloud_path, torch_save_to_cloud

logger = logging.getLogger(__name__)


def maybe_unwrap_dist_model(model: nn.Module, use_distributed: bool) -> nn.Module:
    if use_distributed and hasattr(model, 'module'):
        return model.module
    return model


def get_state_dict(model, drop_untrained_params: bool = True) -> dict[str, Any]:
    """Get model state dict. Optionally drop untrained parameters to keep only those that require gradient.

    Args:
        model: Model to get state dict from
        drop_untrained_params: Whether to drop untrained parameters

    Returns:
        dict: Model state dict
    """
    if not drop_untrained_params:
        return model.state_dict()

    param_grad_dict = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    return state_dict


def save_model_checkpoint(
    model: nn.Module,
    save_path: Union[str, os.PathLike, CloudPath],
    use_distributed: bool = False,
    drop_untrained_params: bool = False,
    **objects_to_save,
) -> None:
    """Save model checkpoint.

    Args:
        model (nn.Module): Model to save
        output_dir (str): Output directory to save checkpoint
        use_distributed (bool): Whether the model is distributed, if so, unwrap it. Default: False.
        is_best (bool): Whether the model is the best in the training run. Default: False.
        drop_untrained_params (bool): Whether to drop untrained parameters to save. Default: True.
        prefix (str): Prefix to add to the checkpoint file name. Default: "".
        extention (str): Extension to use for the checkpoint file. Default: "pth".
        **objects_to_save: Additional objects to save, e.g. optimizer state dict, etc.
    """
    # Only save on main process
    if not is_main_process():
        return
    
    # Validate save path
    save_path_str = str(save_path)
    if not is_cloud_path(save_path):
        save_dir = os.path.dirname(save_path_str)
        if save_dir and not os.path.exists(save_dir):
            raise FileNotFoundError("Directory {} does not exist. Please create it first.".format(save_dir))

    model_no_ddp = maybe_unwrap_dist_model(model, use_distributed)
    state_dict = get_state_dict(model_no_ddp, drop_untrained_params)
    save_obj = {
        "model": state_dict,
        **objects_to_save,
    }

    logger.info("Saving checkpoint to {}.".format(save_path))

    if is_cloud_path(save_path):
        torch_save_to_cloud(save_obj, save_path)
    else:
        torch.save(save_obj, save_path)


def load_checkpoint(
    checkpoint_path: Union[str, os.PathLike, CloudPath],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: Union[str, torch.device] = 'cpu',
    strict: bool = False,
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> dict[str, Any]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict matching
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        
    Returns:
        Dictionary with checkpoint information
    """
    # Validate checkpoint path
    checkpoint_path_str = str(checkpoint_path)
    if not is_cloud_path(checkpoint_path) and not os.path.exists(checkpoint_path_str):
        raise FileNotFoundError("Checkpoint file not found: {}".format(checkpoint_path_str))
    
    logger.info("Loading checkpoint from {}".format(checkpoint_path))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model = maybe_unwrap_dist_model(model, hasattr(model, 'module'))
    if 'model' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model' state dict")
    model.load_state_dict(checkpoint['model'], strict=strict)
    
    # Load optional components
    if optimizer is not None and load_optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            raise CheckpointError(f"Failed to load optimizer state: {e}") from e
    
    if scheduler is not None and load_scheduler and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            raise CheckpointError(f"Failed to load scheduler state: {e}") from e
    
    # Return checkpoint info - all fields are required
    if 'epoch' not in checkpoint:
        raise CheckpointError("Checkpoint missing required field 'epoch'")
    if 'iteration' not in checkpoint:
        raise CheckpointError("Checkpoint missing required field 'iteration'")
    
    info = {
        'epoch': checkpoint['epoch'],
        'iteration': checkpoint['iteration'],
        'best_loss': checkpoint.get('best_loss', float('inf')),  # This one can have a default
        'config': checkpoint.get('config', {})  # This one can have a default
    }
    
    return info


def get_latest_checkpoint(checkpoint_dir: Union[str, os.PathLike]) -> Path | None:
    """Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        return None
    
    # Sort by name (assumes naming like checkpoint_00001.pth)
    checkpoints.sort()
    latest = checkpoints[-1]
    
    return latest


def save_model_pretrained(
    model: nn.Module,
    save_directory: Union[str, os.PathLike]
) -> None:
    """Save model in pretrained format compatible with load_higgs_audio_tokenizer.
    
    Args:
        model: HiggsAudioTokenizer model to save
        save_directory: Directory to save model and config
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = save_directory / "model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Create minimal config matching pretrained format
    config = {
        "n_filters": model.encoder.n_filters,
        "D": model.encoder.out_channels,
        "codebook_dim": model.quantizer.codebook_dim if hasattr(model.quantizer, 'codebook_dim') else 64,
        "target_bandwidths": model.target_bandwidths,
        "ratios": list(model.encoder.ratios),
        "sample_rate": model.sample_rate,
        "bins": model.quantizer.bins if hasattr(model.quantizer, 'bins') else 1024,
        "n_q": model.n_q,
        "semantic_model_path": model.semantic_model_path,
        "last_layer_semantic": model.last_layer_semantic,
        "merge_mode": "concat",  # Default value from pretrained models
        "downsample_mode": model.downsample_mode,
        "semantic_mode": "classic",  # Default value from pretrained models
        "vq_scale": 1  # Default value from pretrained models
    }
    
    # Save config as JSON
    config_path = save_directory / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved pretrained model to {save_directory}")


def load_higgs_audio_tokenizer(tokenizer_name_or_path: Union[str, os.PathLike], device: str = "cuda") -> nn.Module:
    """Load HiggsAudioTokenizer from local path or HuggingFace hub.
    
    Args:
        tokenizer_name_or_path: Path to tokenizer directory or HuggingFace model ID
        device: Device to load model on
        
    Returns:
        HiggsAudioTokenizer model ready for inference
    """
    # Import here to avoid circular dependency
    from HiggsAudioTokenizer.models.audiotokenizer import HiggsAudioTokenizer
    
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    model_path = os.path.join(tokenizer_path, "model.pth")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = HiggsAudioTokenizer(
        **config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model