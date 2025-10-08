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

"""Utilities for loading and managing tokenizer models"""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import resampy
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, DistributedSampler

from .config import Config, DatasetConfig, ModelConfig, RunConfig, AudioCodecConfig
from .dist_utils import get_rank, get_world_size, is_dist_avail_and_initialized
from .models.audiotokenizer import HiggsAudioTokenizer
from .storage_utils import GSPath, is_gcs_path
from .exceptions import ConfigError, ModelError

logger = logging.getLogger(__name__)


# Constants
TARGET_SAMPLE_RATE = 16_000


# Model registry for different versions and tags
MODEL_REGISTRY = {
    "24khz": {
        "url": None,  # URL for downloading pretrained model
        "description": "24kHz audio tokenizer",
        "config": AudioCodecConfig(
            sample_rate=24000,
            encoder_dim=64,
            encoder_rates=[2, 4, 5, 8],
            decoder_dim=1536,
            decoder_rates=[8, 5, 4, 2],
            n_codebooks=32,
            codebook_size=1024,
            codebook_dim=8,
            quantizer_dropout=0.5
        )
    },
    "16khz": {
        "url": None,
        "description": "16kHz audio tokenizer",
        "config": AudioCodecConfig(
            sample_rate=16000,
            encoder_dim=64,
            encoder_rates=[2, 4, 8, 8],
            decoder_dim=1536,
            decoder_rates=[8, 8, 4, 2],
            n_codebooks=32,
            codebook_size=1024,
            codebook_dim=8,
            quantizer_dropout=0.5
        )
    },
    "44khz": {
        "url": None,
        "description": "44.1kHz audio tokenizer",
        "config": AudioCodecConfig(
            sample_rate=44100,
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 8],
            decoder_dim=1536,
            decoder_rates=[8, 4, 4, 2],
            n_codebooks=32,
            codebook_size=1024,
            codebook_dim=8,
            quantizer_dropout=0.5
        )
    }
}


def snr_scale(clean: torch.Tensor, noise: torch.Tensor, snr: float) -> torch.Tensor:
    """Scale noise to achieve desired signal-to-noise ratio.
    
    Args:
        clean: Clean signal tensor
        noise: Noise signal tensor (same shape as clean)
        snr: Desired SNR in dB
        
    Returns:
        Scaled noise tensor that achieves the target SNR when added to clean
        
    Raises:
        AssertionError: If clean and noise have different shapes
    """
    # Ensure both clean and noise have the same shape
    assert clean.shape == noise.shape, "Clean and noise must have the same shape."

    # Compute power (mean squared amplitude)
    power_signal = torch.mean(clean**2)
    power_noise = torch.mean(noise**2)

    # Prevent division by zero
    epsilon = 1e-10
    power_noise = torch.clamp(power_noise, min=epsilon)

    # Calculate desired noise power based on SNR
    desired_noise_power = power_signal / (10 ** (snr / 10))

    # Scale noise to achieve the desired noise power
    scale = torch.sqrt(desired_noise_power / power_noise)
    scaled_noise = scale * noise

    return scaled_noise


def time_scale(signal: torch.Tensor, scale: float = 2.0, 
               rngnp: Optional[np.random.Generator] = None, seed: int = 42) -> torch.Tensor:
    """Apply random time scaling to audio signal.
    
    Stretches or compresses the signal in time by a random factor,
    then resamples to original length.
    
    Args:
        signal: Input audio tensor
        scale: Maximum scaling factor (actual scale is random in [1/scale, scale])
        rngnp: NumPy random generator for reproducibility
        seed: Random seed if rngnp not provided
        
    Returns:
        Time-scaled signal with same shape as input
    """
    if rngnp is None:
        rngnp = np.random.default_rng(seed=seed)
    scaling = np.power(scale, rngnp.uniform(-1, 1))
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    ## trim or zero pad to torche original size
    if scaled_signal.shape[-1] > signal.shape[-1]:
        nframes_offset = (scaled_signal.shape[-1] - signal.shape[-1]) // 2
        scaled_signal = scaled_signal[..., nframes_offset : nframes_offset + signal.shape[-1]]
    else:
        nframes_diff = signal.shape[-1] - scaled_signal.shape[-1]
        pad_left = int(np.random.uniform() * nframes_diff)
        pad_right = nframes_diff - pad_left
        scaled_signal = F.pad(input=scaled_signal, pad=(pad_left, pad_right), mode="constant", value=0)
    return scaled_signal


def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Generate mel-scale frequencies.
    
    Args:
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        
    Returns:
        Array of mel frequencies in Hz
    """
    def _hz_to_mel(f: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + f / 700)

    def _mel_to_hz(m: float) -> float:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10 ** (m / 2595) - 1)

    low = _hz_to_mel(fmin)
    high = _hz_to_mel(fmax)

    mels = np.linspace(low, high, n_mels)

    return _mel_to_hz(mels)


def now_as_str() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_device(sample: Any, device: Union[str, torch.device]) -> Any:
    """Move a sample to the specified device.
    
    Args:
        sample: Sample to move (can be nested structure)
        device: Target device ('cuda', 'cpu', or torch.device)
        
    Returns:
        Sample with all tensors moved to device
    """
    def _move_to_device(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_device, sample)


def prepare_sample(samples: Any, cuda_enabled: bool = True) -> Any:
    """Prepare samples for model input.
    
    Args:
        samples: Input samples
        cuda_enabled: Whether to move samples to CUDA
        
    Returns:
        Prepared samples on appropriate device
    """
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    # TODO fp16 support

    return samples


def prepare_sample_dist(samples: Any, device: Union[str, torch.device]) -> Any:
    """Prepare samples for distributed training.
    
    Args:
        samples: Input samples
        device: Target device
        
    Returns:
        Prepared samples on specified device
    """
    samples = move_to_device(samples, device)

    # TODO fp16 support

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def prepare_one_sample(wav_path: str, wav_processor: Optional[Any] = None, 
                      cuda_enabled: bool = True) -> dict:
    """Prepare a single audio sample for inference.

    Args:
        wav_path: Path to the audio file
        wav_processor: Optional function to process the audio
        cuda_enabled: Whether to move the sample to GPU
        
    Returns:
        Dictionary containing processed audio sample
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 10s

    # spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
    print("audio shape", audio.shape)

    audio_t = torch.tensor(audio).unsqueeze(0)
    audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SAMPLE_RATE)
    print("audio shape after resample", audio_t.shape)

    samples = {
        "raw_wav": audio_t,
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        "audio_chunk_sizes": [1],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_one_sample_waveform(audio, cuda_enabled=True, sr=16000):
    print("shape", audio.shape)
    if len(audio.shape) == 2:  # stereo to mono
        print("converting stereo to mono?")
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 30s

    samples = {
        "raw_wav": torch.tensor(audio).unsqueeze(0).type(torch.DoubleTensor),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_sample_waveforms(audio_paths, cuda_enabled=True, sr=TARGET_SAMPLE_RATE, max_length_seconds=10):
    batch_len = sr  # minimum length of audio
    audios = []
    for audio_path in audio_paths:
        audio, loaded_sr = sf.read(audio_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        audio = audio[: loaded_sr * 10]
        audio = resampy.resample(audio, loaded_sr, sr)
        audio = torch.from_numpy(audio)

        if len(audio) < sr * max_length_seconds:
            pad_size = sr * max_length_seconds - len(audio)
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio = torch.clamp(audio, -1.0, 1.0)
        if len(audio) > batch_len:
            batch_len = len(audio)
        audios.append(audio)
    padding_mask = torch.zeros((len(audios), batch_len), dtype=torch.bool)
    for i in range(len(audios)):
        if len(audios[i]) < batch_len:
            pad_len = batch_len - len(audios[i])
            sil = torch.zeros(pad_len, dtype=torch.float32)
            audios[i] = torch.cat((audios[i], sil), dim=0)
            padding_mask[i, len(audios[i]) :] = True
    audios = torch.stack(audios, dim=0)

    samples = {
        "raw_wav": audios,
        "padding_mask": padding_mask,
        "audio_chunk_sizes": [len(audio_paths)],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def generate_sample_batches(
    audio_path,
    cuda_enabled: bool = True,
    sr: int = TARGET_SAMPLE_RATE,
    chunk_len: int = 10,
    hop_len: int = 5,
    batch_size: int = 4,
):
    audio, loaded_sr = sf.read(audio_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    audio = torchaudio.functional.resample(torch.from_numpy(audio), loaded_sr, sr)
    hop_len = hop_len * sr
    chunk_len = max(len(audio), chunk_len * sr)
    chunks = []

    for i in range(0, len(audio), hop_len):
        chunk = audio[i : i + chunk_len]
        if len(chunk) < chunk_len:
            break
        chunks.append(chunk)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        padding_mask = torch.zeros((len(batch), sr * chunk_len), dtype=torch.bool)
        batch = torch.stack(batch, dim=0)
        samples = {
            "raw_wav": batch,
            "padding_mask": padding_mask,
            "audio_chunk_sizes": [1 for _ in range(len(batch))],
        }
        if cuda_enabled:
            samples = move_to_device(samples, "cuda")
        yield samples


def prepare_samples_for_detection(samples, prompt, label):
    prompts = [prompt for i in range(len(samples["raw_wav"]))]
    labels = [label for i in range(len(samples["raw_wav"]))]
    task = ["detection" for i in range(len(samples["raw_wav"]))]
    samples["prompt"] = prompts
    samples["text"] = labels
    samples["task"] = task
    return samples


def universal_torch_load(
    f: str | os.PathLike,
    *,
    cache_mode: Literal["none", "use", "force"] = "none",
    **kwargs,
) -> Any:
    """
    Wrapper function for torch.load that can handle GCS paths.

    This function provides a convenient way to load PyTorch objects from both local and
    Google Cloud Storage (GCS) paths. For GCS paths, it can optionally caches the
    downloaded files locally to avoid repeated downloads.

    The cache location is determined by:
    1. The ESP_CACHE_HOME environment variable if set
    2. Otherwise defaults to ~/.cache/esp/

    Args:
        f: File-like object, string or PathLike object.
           Can be a local path or a GCS path (starting with 'gs://').
        cache_mode (str, optional): Cache mode for GCS files. Options are:
            "none": No caching (use bucket directly)
            "use": Use cache if available, download if not
            "force": Force redownload even if cache exists
            Defaults to "none".
        **kwargs: Additional keyword arguments passed to torch.load().

    Returns:
        The object loaded from the file using torch.load.

    Raises:
        IsADirectoryError: If the GCS path points to a directory instead of a file.
        FileNotFoundError: If the local file does not exist.
    """

    if is_gcs_path(f):
        gs_path = GSPath(str(f))
        if gs_path.is_dir():
            raise IsADirectoryError(f"Cannot load a directory: {f}")

        if cache_mode in ["use", "force"]:
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / gs_path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / gs_path.name

            if not cache_path.exists() or cache_mode == "force":
                logger.info(
                    f"{'Force downloading' if cache_mode == 'force' else 'Cache file does not exist, downloading'} to {cache_path}..."
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                gs_path.download_to(cache_path)
            else:
                logger.debug(f"Found {cache_path}, using local cache.")
            f = cache_path
        else:
            f = gs_path
    else:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"File does not exist: {f}")

    with open(f, "rb") as opened_file:
        return torch.load(opened_file, **kwargs)


def download_model(model_tag: str, model_dir: Optional[Path] = None) -> Path:
    """Download a pretrained model if not already cached

    Args:
        model_tag: Tag of the model to download
        model_dir: Directory to save the model (default: ~/.cache/higgs-audio-v2-unofficial)
        
    Returns:
        Path to the downloaded model file
    """
    if model_tag not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model tag: {model_tag}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "higgs-audio-v2-unofficial"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{model_tag}.pt"
    
    if model_path.exists():
        return model_path
    
    # Download model if URL is available
    url = MODEL_REGISTRY[model_tag]["url"]
    if url is None:
        raise ValueError(f"No pretrained model available for {model_tag}. "
                        "Please train a model or provide a checkpoint path.")
    
    # TODO: Implement actual download logic when models are available
    # For now, just raise an error
    raise NotImplementedError("Model downloading not yet implemented")


def load_model(
    model_path_or_tag: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    load_state_dict: bool = True,
    tag: Optional[str] = None
) -> HiggsAudioTokenizer:
    """Load a tokenizer model from a checkpoint or tag
    
    Args:
        model_path_or_tag: Path to checkpoint file or model tag
        device: Device to load the model on
        load_state_dict: Whether to load the state dict
        tag: Override tag for configuration (if using custom checkpoint)
        
    Returns:
        Loaded HiggsAudioTokenizer model
    """
    model_path_or_tag = str(model_path_or_tag)
    
    # Check if it's a tag
    if model_path_or_tag in MODEL_REGISTRY:
        tag = model_path_or_tag
        if load_state_dict:
            model_path = download_model(tag)
        else:
            model_path = None
    else:
        # It's a path
        model_path = Path(model_path_or_tag)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to infer tag from filename if not provided
        if tag is None:
            for possible_tag in MODEL_REGISTRY:
                if possible_tag in model_path.stem:
                    tag = possible_tag
                    break
    
    # Get configuration
    if tag is not None and tag in MODEL_REGISTRY:
        config = MODEL_REGISTRY[tag]["config"]
    else:
        # Try to load config from checkpoint
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location="cpu")
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                warnings.warn("No config found in checkpoint, using default 24kHz config")
                config = MODEL_REGISTRY["24khz"]["config"]
        else:
            config = MODEL_REGISTRY["24khz"]["config"]
    
    # Create model with appropriate parameters from AudioCodecConfig
    # Note: This function expects semantic_model_path to be available in config
    if not hasattr(config, 'semantic_model_path'):
        raise ValueError("Config must have semantic_model_path for HiggsAudioTokenizer")
    
    # Validate required config attributes
    if not hasattr(config, 'D'):
        raise ConfigError("Config must have 'D' attribute")
    if not hasattr(config, 'bins') and not hasattr(config, 'codebook_size'):
        raise ConfigError("Config must have either 'bins' or 'codebook_size' attribute")
    if not hasattr(config, 'n_q') and not hasattr(config, 'n_codebooks'):
        raise ConfigError("Config must have either 'n_q' or 'n_codebooks' attribute")
    
    model = HiggsAudioTokenizer(
        n_filters=config.encoder_dim,
        D=config.D,
        ratios=config.encoder_rates,
        sample_rate=config.sample_rate,
        bins=config.bins if hasattr(config, 'bins') else config.codebook_size,
        n_q=config.n_q if hasattr(config, 'n_q') else config.n_codebooks,
        codebook_dim=config.codebook_dim if hasattr(config, 'codebook_dim') else None,
        semantic_model_path=config.semantic_model_path,
        device=str(device)
    )
    
    # Load state dict if requested
    if load_state_dict and model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
        
        # Remove any prefix from state dict keys if needed
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model


def save_model(
    model: HiggsAudioTokenizer,
    save_path: Union[str, Path],
    config: Optional[AudioCodecConfig] = None,
    metadata: Optional[Dict] = None
):
    """Save a tokenizer model to a checkpoint file
    
    Args:
        model: HiggsAudioTokenizer model to save
        save_path: Path to save the checkpoint
        config: Configuration to save (uses model.config if None)
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config or model.config,
    }
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, save_path)


def list_models() -> Dict[str, str]:
    """List available model tags and their descriptions
    
    Returns:
        Dictionary mapping tags to descriptions
    """
    return {tag: info["description"] for tag, info in MODEL_REGISTRY.items()}


def get_model_info(tag: str) -> Dict:
    """Get information about a specific model tag
    
    Args:
        tag: Model tag
        
    Returns:
        Dictionary with model information
    """
    if tag not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model tag: {tag}")
    
    info = MODEL_REGISTRY[tag].copy()
    config = info["config"]
    
    # Add computed information
    info["sample_rate"] = config.sample_rate
    info["downsample_rate"] = config.downsample_rate
    info["frame_rate"] = config.frame_rate
    info["n_codebooks"] = config.n_codebooks
    info["codebook_size"] = config.codebook_size
    info["total_codebook_size"] = config.total_codebook_size
    
    return info


# Convenience function for quick model loading
def load_24khz_model(device: Union[str, torch.device] = "cuda") -> HiggsAudioTokenizer:
    """Load the default 24kHz model
    
    Args:
        device: Device to load the model on
        
    Returns:
        Loaded 24kHz HiggsAudioTokenizer model
    """
    return load_model("24khz", device=device, load_state_dict=False)


def load_discriminator(
    discriminator_type: str = "mpd_msd_mrd",
    period_scales: Optional[List[int]] = None,
    scale_downsample_rates: Optional[List[int]] = None,
    fft_sizes: Optional[List[int]] = None,
    bands: Optional[List[List[float]]] = None,
    device: Union[str, torch.device] = "cuda"
) -> nn.Module:
    """Load discriminator model for GAN training.
    
    Args:
        discriminator_type: Type of discriminator ("mpd_msd_mrd", "mpd_msd", "mpd")
        period_scales: Period scales for MPD (default: [2, 3, 5, 7, 11])
        scale_downsample_rates: Downsample rates for MSD (default: [4, 4, 4, 4])
        device: Device to load the model on
        
    Returns:
        Discriminator model
    """
    try:
        from .models.discriminator import Discriminator, MPD, MRD, MSD
    except ImportError:
        raise ImportError(
            "Failed to import discriminator models. "
            "Please ensure the discriminator module is available."
        )
    
    # Default values
    if period_scales is None:
        period_scales = [2, 3, 5, 7, 11]
    if scale_downsample_rates is None:
        scale_downsample_rates = [4, 4, 4, 4]
    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512]
    if bands is None:
        bands = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    
    # Create discriminator based on type
    if discriminator_type == "mpd_msd_mrd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=scale_downsample_rates,
            fft_sizes=fft_sizes,
            sample_rate=24000,
            bands=bands
        )
    elif discriminator_type == "mpd_msd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=scale_downsample_rates,
            fft_sizes=[],  # No MRD
            sample_rate=24000,
            bands=bands
        )
    elif discriminator_type == "mpd":
        discriminator = Discriminator(
            periods=period_scales,
            rates=[],  # No MSD
            fft_sizes=[],  # No MRD
            sample_rate=24000,
            bands=bands
        )
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
    
    return discriminator.to(device)


def create_model_from_config(
    config: Union[ModelConfig, Config],
    device: Union[str, torch.device] = "cuda"
) -> HiggsAudioTokenizer:
    """Create a model from a Pydantic configuration.
    
    Args:
        config: ModelConfig or full Config object
        device: Device to create the model on
        
    Returns:
        HiggsAudioTokenizer model
    """
    if isinstance(config, Config):
        model_config = config.model
    else:
        model_config = config
    
    # Create HiggsAudioTokenizer with appropriate parameters
    model = HiggsAudioTokenizer(
        n_filters=model_config.encoder_dim,
        D=model_config.D if hasattr(model_config, 'D') else 128,
        ratios=model_config.encoder_rates,
        sample_rate=model_config.sample_rate,
        bins=model_config.bins if hasattr(model_config, 'bins') else model_config.codebook_size,
        n_q=model_config.n_q if hasattr(model_config, 'n_q') else model_config.n_codebooks,
        codebook_dim=model_config.codebook_dim if hasattr(model_config, 'codebook_dim') else None,
        semantic_model_path=model_config.semantic_model_path,
        device=str(device)
    )
    return model.to(device)


def load_models(
    config: Config,
    checkpoint_path: Optional[Path] = None,
    device: Union[str, torch.device] = "cuda"
) -> Tuple[HiggsAudioTokenizer, Optional[nn.Module]]:
    """Load both generator and discriminator models.
    
    Args:
        config: Full configuration object
        checkpoint_path: Optional path to checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (generator, discriminator) models
    """
    # Create generator
    generator = create_model_from_config(config.model, device)
    
    # Create discriminator if needed
    discriminator = None
    if config.model.use_discriminator:
        discriminator = load_discriminator(
            discriminator_type=config.model.discriminator_type,
            period_scales=config.model.discriminator_period_scales,
            scale_downsample_rates=config.model.discriminator_scale_downsample_rates,
            fft_sizes=config.model.discriminator_fft_sizes,
            bands=config.model.discriminator_bands,
            device=device
        )
    
    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load generator state
        if "model_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["model_state_dict"])
        elif "generator_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["generator_state_dict"])
        elif "state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["state_dict"])
        
        # Load discriminator state if available
        if discriminator is not None and "discriminator_state_dict" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        if "iteration" in checkpoint:
            print(f"Checkpoint iteration: {checkpoint['iteration']}")
    
    return generator, discriminator


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    use_ema: bool = True
) -> HiggsAudioTokenizer:
    """Load model specifically for inference.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
        use_ema: Whether to use EMA weights if available
        
    Returns:
        Model ready for inference
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get config
    if "config" in checkpoint:
        # Handle both old dataclass and new Pydantic configs
        if isinstance(checkpoint["config"], dict):
            # It's a Pydantic config saved as dict
            if "model" in checkpoint["config"]:
                # Full config
                config = Config(**checkpoint["config"])
                model = create_model_from_config(config.model, device)
            else:
                # Just model config
                model_config = ModelConfig(**checkpoint["config"])
                model = create_model_from_config(model_config, device)
        else:
            # Legacy dataclass config
            config = checkpoint["config"]
            if not hasattr(config, 'semantic_model_path'):
                raise ValueError("Checkpoint config must have semantic_model_path for HiggsAudioTokenizer")
            
            # Validate required config attributes
            if not hasattr(config, 'D'):
                raise ConfigError("Checkpoint config must have 'D' attribute")
            if not hasattr(config, 'bins') and not hasattr(config, 'codebook_size'):
                raise ConfigError("Checkpoint config must have either 'bins' or 'codebook_size' attribute")
            if not hasattr(config, 'n_q') and not hasattr(config, 'n_codebooks'):
                raise ConfigError("Checkpoint config must have either 'n_q' or 'n_codebooks' attribute")
            
            model = HiggsAudioTokenizer(
                n_filters=config.encoder_dim,
                D=config.D,
                ratios=config.encoder_rates,
                sample_rate=config.sample_rate,
                bins=config.bins if hasattr(config, 'bins') else config.codebook_size,
                n_q=config.n_q if hasattr(config, 'n_q') else config.n_codebooks,
                codebook_dim=config.codebook_dim if hasattr(config, 'codebook_dim') else None,
                semantic_model_path=config.semantic_model_path,
                device=str(device)
            )
    else:
        warnings.warn("No config in checkpoint, using default 24kHz config")
        model = load_24khz_model(device)
    
    # Load weights
    if use_ema and "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded EMA weights")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume checkpoint is state dict itself
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(
    generator: HiggsAudioTokenizer,
    discriminator: Optional[nn.Module] = None
) -> None:
    """Print model summary.
    
    Args:
        generator: Generator model
        discriminator: Optional discriminator model
    """
    g_params = count_parameters(generator)
    print(f"\nGenerator Parameters: {g_params / 1e6:.2f}M")
    
    if discriminator is not None:
        d_params = count_parameters(discriminator)
        print(f"Discriminator Parameters: {d_params / 1e6:.2f}M")
        print(f"Total Parameters: {(g_params + d_params) / 1e6:.2f}M")
    
    print(f"\nModel Configuration:")
    if hasattr(generator, 'sample_rate'):
        print(f"  Sample Rate: {generator.sample_rate} Hz")
    if hasattr(generator, 'downsample_rate'):
        print(f"  Downsample Rate: {generator.downsample_rate}x")
    elif hasattr(generator, 'encoder_rates'):
        # Calculate downsample rate from encoder rates
        downsample = 1
        for rate in generator.encoder_rates:
            downsample *= rate
        print(f"  Downsample Rate: {downsample}x")
    if hasattr(generator, 'frame_rate'):
        print(f"  Frame Rate: {generator.frame_rate:.1f} Hz")
    if hasattr(generator, 'n_codebooks'):
        print(f"  Codebooks: {generator.n_codebooks}")
    if hasattr(generator, 'codebook_size'):
        print(f"  Codebook Size: {generator.codebook_size}")
    if hasattr(generator, 'n_codebooks') and hasattr(generator, 'codebook_size'):
        print(f"  Total Vocabulary: {generator.n_codebooks * generator.codebook_size}")


def ensure_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype.
    
    Args:
        dtype_str: String representation of dtype
        
    Returns:
        torch.dtype object
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    
    return dtype_map[dtype_str]


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    """Create a DataLoader with optional distributed sampler.
    
    Args:
        dataset: The dataset to load
        config: Config with batch size and data loading settings
        is_train: Whether this is for training (affects shuffling)
        use_distributed: Whether to use distributed sampler
        
    Returns:
        DataLoader instance (wrapped in IterLoader for training)
    """
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, num_replicas=get_world_size(), rank=get_rank())
    else:
        sampler = None
    
    # Check if dataset has a collater
    collate_fn = dataset.collater if hasattr(dataset, 'collater') else None
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=False,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        collate_fn=collate_fn,
        drop_last=is_train,
    )
    
    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)
    
    return loader