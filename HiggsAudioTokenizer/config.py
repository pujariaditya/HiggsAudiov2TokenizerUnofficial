"""Configuration system for HiggsAudiov2 (Unofficial Implementation).

Pydantic-based configuration system for audio codec training with GAN losses.
Based on the HiggsAudiov2 architecture.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict, Union
import yaml

from pydantic import BaseModel, field_validator, ConfigDict
from pydantic.v1.utils import deep_update
from pydantic_settings import BaseSettings, CliSettingsSource, YamlConfigSettingsSource

from .exceptions import ConfigError


# Legacy dataclass configs (kept for compatibility)
@dataclass
class AudioCodecConfig:
    """Configuration for HiggsAudiov2-style audio codec"""
    
    # Encoder settings (Higgs Audio V2 24kHz)
    encoder_dim: int = 32
    n_filters: int = 32  # Initial convolution filters (from pretrained config)
    encoder_rates: List[int] = None  # [8, 5, 4, 2, 3] for Higgs Audio V2 (960x downsampling)
    latent_dim: int = None  # Auto-calculated if None
    
    # Decoder settings (Higgs Audio V2 24kHz) 
    decoder_dim: int = 1536
    decoder_rates: List[int] = None  # [3, 2, 4, 5, 8] for Higgs Audio V2 (reverse of encoder)
    
    # Quantizer settings (Higgs Audio V2 24kHz)
    n_codebooks: int = 8  # Changed from 32 to match Higgs Audio V2 paper
    codebook_size: int = 1024
    codebook_dim: int = 64
    quantizer_dropout: float = 0.5
    
    # Audio settings
    sample_rate: int = 24000
    
    def __post_init__(self):
        """Validate required values"""
        if self.encoder_rates is None:
            raise ConfigError("encoder_rates must be specified")
            
        if self.decoder_rates is None:
            raise ConfigError("decoder_rates must be specified")
            
        if self.latent_dim is None:
            # Auto-calculate latent dimension based on encoder
            self.latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))
    
    @property
    def downsample_rate(self) -> int:
        """Total downsampling rate"""
        rate = 1
        for s in self.encoder_rates:
            rate *= s
        return rate
    
    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz"""
        return self.sample_rate / self.downsample_rate
    
    @property
    def total_codebook_size(self) -> int:
        """Total number of unique audio tokens"""
        return self.n_codebooks * self.codebook_size


@dataclass
class TrainingConfig:
    """Configuration for training the tokenizer"""
    
    # Data settings
    batch_size: int = 32
    num_workers: int = 4
    segment_length: int = 24000  # 1 second at 24kHz
    train_data_path: str = "data/train.jsonl"
    val_data_path: str = "data/val.jsonl"
    
    # Optimization
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_iterations: int = 100000
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    
    # Loss weights
    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    stft_weight: float = 1.0
    mel_weight: float = 15.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5000
    validation_interval: int = 1000
    log_interval: int = 100
    
    # Hardware
    device: str = "cuda"
    fp16: bool = True
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class InferenceConfig:
    """Configuration for inference/encoding"""
    
    model_path: str = "checkpoints/best_model.pt"
    device: str = "cuda"
    batch_size: int = 32
    
    # Audio processing
    sample_rate: int = 24000
    normalize_audio: bool = True
    
    # Output settings
    return_audio: bool = True
    return_tokens: bool = True


# New Pydantic-based configs

class ModelConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Model architecture configuration."""
    model_type: Literal["dac", "custom", "higgs_audio"] = "higgs_audio"
    
    # Encoder
    encoder_dim: int = 32
    n_filters: int = 32  # Initial convolution filters (from pretrained config)
    encoder_rates: List[int] = [8, 5, 4, 2, 3]  # Higgs Audio V2: 960x downsampling
    encoder_residual_layers: int = 1
    
    # Decoder
    decoder_dim: int = 1536
    decoder_rates: List[int] = [3, 2, 4, 5, 8]  # Higgs Audio V2: reverse of encoder
    decoder_residual_layers: int = 1
    
    # Quantizer
    n_codebooks: int = 8  # Higgs Audio V2: 8 codebooks for RVQ
    codebook_size: int = 1024
    codebook_dim: int = 64
    quantizer_dropout: float = 0.5
    
    # Audio
    sample_rate: int = 24000
    channels: int = 1
    
    # HiggsAudio specific parameters
    semantic_model_path: str  # Required: Path to pretrained semantic model (e.g., HuBERT)
    semantic_teacher: str = "hubert_base_general"  # Type of semantic model (from pretrained config)
    last_layer_semantic: bool = True
    merge_mode: str = "concat"
    downsample_mode: str = "step_down"
    semantic_mode: str = "classic"
    vq_scale: int = 1
    semantic_sample_rate: Optional[int] = None
    D: int = 256  # Acoustic feature dimension
    bins: Any = 1024  # Can be int (RVQ) or List[int] (RFSQ)
    n_q: int = 8  # Number of quantizers
    target_bandwidths: List[float] = [0.5, 1, 1.5, 2, 4]
    
    # Discriminator
    use_discriminator: bool = True
    discriminator_type: Literal["mpd_msd_mrd", "mpd_msd", "mpd"] = "mpd_msd_mrd"
    discriminator_period_scales: List[int] = [2, 3, 5, 7, 11]
    discriminator_scale_downsample_rates: List[int] = [4, 4, 4, 4]
    discriminator_fft_sizes: List[int] = [2048, 1024, 512]
    discriminator_bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    
    @property
    def downsample_rate(self) -> int:
        """Total downsampling rate."""
        rate = 1
        for s in self.encoder_rates:
            rate *= s
        return rate
    
    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sample_rate / self.downsample_rate


class OptimizerConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Optimizer configuration (using AdamW only, matching descript-audio-codec)."""
    # Generator optimizer
    g_optimizer: Literal["adamw"] = "adamw"  # Only AdamW supported
    g_learning_rate: float = 1e-4  # From descript-audio-codec
    g_betas: List[float] = [0.8, 0.99]  # From descript-audio-codec
    g_eps: float = 1e-8  # Standard PyTorch default
    g_weight_decay: float = 0.0  # Not explicitly set in descript-audio-codec
    
    # Discriminator optimizer
    d_optimizer: Literal["adamw"] = "adamw"  # Only AdamW supported
    d_learning_rate: float = 1e-4  # From descript-audio-codec
    d_betas: List[float] = [0.8, 0.99]  # From descript-audio-codec
    d_eps: float = 1e-8  # Standard PyTorch default
    d_weight_decay: float = 0.0  # Not explicitly set in descript-audio-codec
    
    # Scheduler (matching descript-audio-codec)
    scheduler_type: Literal["constant", "exponential", "cosine", "linear_warmup_cosine"] = "exponential"
    warmup_steps: int = 0  # No warmup in descript-audio-codec
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    gamma: float = 0.999996  # From descript-audio-codec
    
    # Gradient clipping (values from descript-audio-codec)
    max_grad_norm: Optional[float] = 1000.0  # Will be applied differently for G (1e3) and D (10.0)
    max_grad_value: Optional[float] = None


class AugmentationsConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Audio augmentation configuration."""
    use_augmentation: bool = True
    
    # Volume normalization (supports both old and new formats)
    volume_norm: Optional[Dict[str, Any]] = None  # New format: {db: [const, -16]}
    volume_norm_prob: float = 1.0  # Old format (deprecated)
    volume_norm_target_db: float = -16.0  # Old format (deprecated)
    
    # Audio rescaling
    rescale_prob: float = 0.1
    rescale_min: float = 0.9
    rescale_max: float = 1.1
    
    # Phase shift
    phase_shift_prob: float = 0.1
    phase_shift_max: float = 0.5
    
    # Random gain
    random_gain_prob: float = 0.1
    random_gain_min_db: float = -6.0
    random_gain_max_db: float = 6.0
    
    # Random EQ
    random_eq_prob: float = 0.0
    random_eq_num_bands: int = 3


class LossConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Loss function configuration aligned with Higgs Audio V2."""
    
    # Mel-spectrogram loss (primary reconstruction loss)
    mel_loss_weight: float = 100.0
    mel_loss_params: Dict[str, Any] = {
        "sample_rate": 24000,
        "win_lengths": [32, 64, 128, 256, 512, 1024, 2048],
        "n_mels": [5, 10, 20, 40, 80, 160, 320],
        "pow": 1.0,
        "normalized": False,
        "eps": 1e-5,
        "mag_weight": 0.0
    }
    
    # Semantic loss (Higgs Audio V2)
    semantic_loss_weight: float = 10.0  # HuBERT feature matching loss
    semantic_loss_type: Literal["mse", "l1", "cosine"] = "mse"
    
    # Optional waveform loss
    waveform_loss_weight: float = 0.0  # Set to 1.0 for non-adversarial training
    
    # Multi-scale STFT loss (primarily for validation)
    stft_loss_weight: float = 0.0  # Used in validation
    stft_loss_params: Dict[str, Any] = {
        "scales": [2048, 512],
        "overlap": 0.75,
        "eps": 1e-7,
        "mag_weight": 1.0,
        "log_weight": 1.0
    }
    
    # VQ losses
    commitment_weight: float = 0.25
    
    # GAN losses (adversarial training)
    use_gan_loss: bool = True
    adv_gen_loss_weight: float = 1.0  # Generator adversarial loss
    adv_feat_loss_weight: float = 2.0  # Feature matching loss
    
    # Discriminator update frequency
    discriminator_update_freq: int = 1


class DatasetConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Dataset configuration."""
    train_manifest: Path = Path("data/train.jsonl")
    valid_manifest: Path = Path("data/valid.jsonl")
    test_manifest: Path = Path("data/test.jsonl")
    
    segment_length: int = 24000  # Default segment length in samples
    hop_length: Optional[int] = None  # For sliding window
    audio_backend: Literal["torchaudio", "soundfile"] = "torchaudio"
    
    # Duration-based settings (in seconds) - if set, override segment_length
    train_duration: Optional[float] = None  # Duration for training segments
    valid_duration: Optional[float] = None  # Duration for validation segments  
    test_duration: Optional[float] = None   # Duration for test segments
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Cropping for validation
    valid_cropping: Literal["random", "start", "center"] = "start"
    test_cropping: Literal["random", "start", "center"] = "start"
    
    @field_validator("train_manifest", "valid_manifest", "test_manifest", mode="after")
    @classmethod
    def check_files(cls, path: Path) -> Path:
        """Validate manifest files exist."""
        if not path.exists():
            raise ValueError(f"Manifest file {path} does not exist")
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Manifest file {path} must be a JSONL file")
        return path


class RunConfig(BaseModel, extra="forbid", validate_assignment=True):
    """Training run configuration."""
    # Basic settings
    seed: int = 42
    output_dir: Path = Path("outputs/")
    experiment_name: Optional[str] = None
    job_id: Optional[str] = None
    
    # Training settings
    max_iterations: int = 100000
    batch_size_train: int = 32
    batch_size_eval: int = 32
    accum_grad_iters: int = 1
    
    # Validation
    validation_interval: int = 1000
    num_valid_batches: int = 10  # Number of validation batches to process
    num_valid_samples: Optional[int] = None  # Limit validation samples
    save_audio_samples: bool = True
    num_audio_samples: int = 4
    sample_freq: Optional[int] = None  # Audio sample logging frequency (if None, use validation_interval)
    val_idx: Optional[List[int]] = None  # Specific validation sample indices to log
    
    # Checkpointing
    checkpoint_interval: int = 5000
    checkpoint_keep_last: int = 5
    checkpoint_best_metric: str = "valid/loss/total"
    checkpoint_best_mode: Literal["min", "max"] = "min"
    resume_from: Optional[Path] = None
    save_iters: Optional[List[int]] = None  # Save at specific iterations
    
    # Logging
    log_interval: int = 100
    wandb_enabled: bool = True
    wandb_project: str = "higgs-audio-v2-unofficial"
    wandb_entity: Optional[str] = None
    tensorboard_enabled: bool = True
    
    # Hardware
    device: Literal["cuda", "cpu"] = "cuda"
    amp: bool = True
    compile_model: bool = False
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    
    # Distributed training
    use_distributed: bool = False
    dist_backend: Literal["nccl", "gloo"] = "nccl"
    dist_url: str = "env://"
    world_size: int = 1
    rank: int = 0
    gpu: Optional[int] = None
    local_rank: int = 0
    
    # Cloud storage
    use_cloud_storage: bool = False
    cloud_provider: Literal["gcs", "s3"] = "gcs"
    cloud_bucket: Optional[str] = None
    cloud_prefix: Optional[str] = None
    
    # Debugging
    detect_anomaly: bool = False
    profile: bool = False
    profile_steps: int = 10
    
    # Model
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_step: int = 0
    
    # Early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4


class Config(BaseSettings):
    """Main configuration combining all sub-configs."""
    
    model: ModelConfig
    optimizer: OptimizerConfig
    augmentations: AugmentationsConfig
    loss: LossConfig
    dataset: DatasetConfig
    run: RunConfig
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, cli_args: Optional[List[str]] = None) -> "Config":
        """Load config from YAML file with optional CLI overrides."""
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Apply CLI overrides if provided
        if cli_args:
            yaml_config = cls._apply_cli_overrides(yaml_config, cli_args)
        
        return cls(**yaml_config)
    
    @classmethod
    def from_sources(
        cls,
        yaml_file: Optional[Path] = None,
        env_file: Optional[Path] = None,
        cli_args: Optional[List[str]] = None,
        **kwargs
    ) -> "Config":
        """Load config from multiple sources with priority: CLI > env > yaml > defaults."""
        config_dict = {}
        
        # Load from YAML
        if yaml_file and yaml_file.exists():
            with open(yaml_file, "r") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config is None:
                    raise ConfigError(f"Config file {yaml_file} is empty or invalid")
                config_dict = deep_update(config_dict, yaml_config)
        
        # Apply kwargs
        config_dict = deep_update(config_dict, kwargs)
        
        # Apply CLI overrides
        if cli_args:
            config_dict = cls._apply_cli_overrides(config_dict, cli_args)
        
        return cls(**config_dict)
    
    @staticmethod
    def _apply_cli_overrides(config_dict: Dict[str, Any], cli_args: List[str]) -> Dict[str, Any]:
        """Apply CLI overrides to config dictionary."""
        for arg in cli_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                keys = key.split(".")
                
                # Navigate to the nested dict
                d = config_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                
                # Parse the value based on its content
                if value.lower() in ["true", "false"]:
                    d[keys[-1]] = value.lower() == "true"
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    d[keys[-1]] = int(value)
                elif '.' in value:
                    try:
                        d[keys[-1]] = float(value)
                    except ValueError:
                        d[keys[-1]] = value
                else:
                    d[keys[-1]] = value
        
        return config_dict
    
    def save_to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def extract_model_config(self) -> Dict[str, Any]:
        """Extract minimal model configuration matching pretrained format."""
        return {
            "n_filters": self.model.n_filters,
            "D": self.model.D,
            "codebook_dim": self.model.codebook_dim,
            "target_bandwidths": self.model.target_bandwidths,
            "ratios": self.model.encoder_rates,
            "sample_rate": self.model.sample_rate,
            "bins": self.model.bins,
            "n_q": self.model.n_q,
            "semantic_model_path": self.model.semantic_model_path,
            "last_layer_semantic": self.model.last_layer_semantic,
            "merge_mode": self.model.merge_mode,
            "downsample_mode": self.model.downsample_mode,
            "semantic_mode": self.model.semantic_mode,
            "vq_scale": self.model.vq_scale
        }
    
    def save_model_json(self, path: Path) -> None:
        """Save minimal model config to JSON file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        model_config = self.extract_model_config()
        with open(path, "w") as f:
            json.dump(model_config, f, indent=2)
    
    def pretty_print(self) -> None:
        """Pretty print the configuration."""
        import json
        from pathlib import Path
        
        def convert_paths(obj):
            """Convert Path objects to strings for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        print("Configuration:")
        config_dict = convert_paths(self.model_dump())
        print(json.dumps(config_dict, indent=2))


# Helper function to create default config
def create_default_config() -> Config:
    """Create a default configuration."""
    return Config(
        model=ModelConfig(),
        optimizer=OptimizerConfig(),
        augmentations=AugmentationsConfig(),
        loss=LossConfig(),
        dataset=DatasetConfig(),
        run=RunConfig()
    )