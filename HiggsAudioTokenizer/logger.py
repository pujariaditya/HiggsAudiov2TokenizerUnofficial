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

"""Logging utilities for HiggsAudiov2 (Unofficial Implementation).

Provides unified logging infrastructure for model training, supporting:
- Console logging with customizable verbosity
- TensorBoard integration for scalar and audio visualization
- Weights & Biases (W&B) support for experiment tracking
- Multi-GPU distributed training compatibility
- Metric tracking and averaging

Based on logging patterns from various audio modeling frameworks.
"""

import os
import time
import datetime
import json
import logging
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .dist_utils import is_main_process, is_dist_avail_and_initialized, get_world_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier(device_ids=[t.device.index])
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """Logs metrics during training with support for distributed synchronization."""
    
    def __init__(self, delimiter="\t", name=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.name = name

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None, start_step=None, wandb_logger=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                if is_main_process():
                    if logger is not None:
                        assert start_step is not None, "start_step is needed to compute global_step!"
                        for name, meter in self.meters.items():
                            logger.add_scalar("{}".format(name), float(str(meter)), global_step=start_step + i)
                        # Log to wandb if logger is provided
                        if wandb_logger is not None:
                            wandb_logger.log({name: float(str(meter)) for name, meter in self.meters.items()}, step=start_step + i)
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))

    def get_all_metrics(self):
        """Get all metrics as a dictionary."""
        return {k: v.global_avg for k, v in self.meters.items()}


class CodecMetricLogger(MetricLogger):
    """Extended metric logger for audio codec training with GAN losses."""
    
    def __init__(self, delimiter="\t", name="codec"):
        super().__init__(delimiter, name)
        # Track different loss components
        self.loss_groups = {
            'reconstruction': ['l1/loss', 'stft/loss', 'mel/loss'],
            'vq': ['vq/commitment_loss', 'vq/codebook_loss'],
            'adversarial': ['adv/gen_loss', 'adv/feat_loss', 'adv/disc_loss'],
            'total': ['loss', 'g_loss', 'd_loss']
        }
    
    def get_grouped_metrics(self):
        """Get metrics grouped by loss type."""
        grouped = {}
        all_metrics = self.get_all_metrics()
        
        for group_name, metric_names in self.loss_groups.items():
            grouped[group_name] = {}
            for name in metric_names:
                if name in all_metrics:
                    grouped[group_name][name] = all_metrics[name]
        
        # Add any ungrouped metrics
        grouped_names = set()
        for names in self.loss_groups.values():
            grouped_names.update(names)
        
        grouped['other'] = {k: v for k, v in all_metrics.items() 
                           if k not in grouped_names}
        
        return grouped
    
    def __str__(self):
        """Format metrics in a readable multi-line format."""
        lines = []
        
        # Group metrics by type
        generator_metrics = ['mel_loss', 'vq_commitment_loss', 'semantic_loss', 'g_total_loss']
        adversarial_metrics = ['adv_gen_loss', 'adv_feat_loss']
        discriminator_main = ['d_total_loss']
        # Dynamically find discriminator losses
        discriminator_individual = []
        for name in self.meters:
            if name.startswith('d_') and name.endswith('_loss') and name != 'd_total_loss':
                discriminator_individual.append(name)
        discriminator_individual.sort(key=lambda x: int(x.split('_')[1]))  # Sort by discriminator number
        lr_metrics = ['lr_g', 'lr_d']
        
        # Generator losses
        gen_parts = []
        for name in generator_metrics:
            if name in self.meters:
                meter = self.meters[name]
                gen_parts.append(f"{name}: {meter.value:.4f} ({meter.global_avg:.4f})")
        if gen_parts:
            lines.append("  Generator: " + " | ".join(gen_parts))
        
        # Adversarial losses
        adv_parts = []
        for name in adversarial_metrics:
            if name in self.meters:
                meter = self.meters[name]
                adv_parts.append(f"{name}: {meter.value:.4f} ({meter.global_avg:.4f})")
        if adv_parts:
            lines.append("  Adversarial: " + " | ".join(adv_parts))
        
        # Discriminator total loss
        disc_parts = []
        for name in discriminator_main:
            if name in self.meters:
                meter = self.meters[name]
                disc_parts.append(f"{name}: {meter.value:.4f} ({meter.global_avg:.4f})")
        if disc_parts:
            lines.append("  Discriminator: " + " | ".join(disc_parts))
        
        # Individual discriminator losses (split into multiple lines)
        disc_individual = []
        for name in discriminator_individual:
            if name in self.meters:
                meter = self.meters[name]
                # Extract discriminator number for display
                disc_num = name.split('_')[1]
                disc_individual.append(f"d_{disc_num}: {meter.value:.4f} ({meter.global_avg:.4f})")
        
        if disc_individual:
            # Split into lines of 4 discriminators each
            for i in range(0, len(disc_individual), 4):
                batch = disc_individual[i:i+4]
                lines.append("    " + " | ".join(batch))
        
        # Learning rates
        lr_parts = []
        for name in lr_metrics:
            if name in self.meters:
                meter = self.meters[name]
                short_name = name.replace('lr_', '')
                lr_parts.append(f"{short_name}: {meter.value:.4e}")
        if lr_parts:
            lines.append("  Learning Rate: " + " | ".join(lr_parts))
        
        # Any other metrics not covered
        covered_metrics = set(generator_metrics + adversarial_metrics + 
                            discriminator_main + discriminator_individual + lr_metrics)
        other_parts = []
        for name, meter in self.meters.items():
            if name not in covered_metrics:
                other_parts.append(f"{name}: {meter.value:.4f} ({meter.global_avg:.4f})")
        if other_parts:
            lines.append("  Other: " + " | ".join(other_parts))
        
        return "\n" + "\n".join(lines) if lines else "No metrics recorded"


class TensorBoardLogger(object):
    """TensorBoard logger for audio codec training."""
    
    def __init__(self, log_dir, comment=""):
        self.log_dir = Path(log_dir)
        self.writer = None
        if is_main_process():
            self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
    
    def add_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, step):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_audio(self, tag, audio, step, sample_rate):
        if self.writer is not None:
            # Ensure audio is in the right format
            if audio.dim() == 3:
                audio = audio.squeeze(0)  # Remove batch dimension
            if audio.dim() == 2 and audio.size(0) > 1:
                audio = audio[0]  # Take first channel if multi-channel
            self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)
    
    def add_image(self, tag, img, step):
        if self.writer is not None:
            self.writer.add_image(tag, img, step)
    
    def add_spectrogram(self, tag, audio, step, 
                       sample_rate=24000, n_fft=2048):
        """Add spectrogram visualization of audio."""
        if self.writer is not None:
            # Compute spectrogram
            spec = torch.stft(
                audio.squeeze(),
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft).to(audio.device),
                return_complex=True
            )
            spec_mag = spec.abs()
            spec_db = 20 * torch.log10(spec_mag + 1e-8)
            
            # Normalize for visualization
            spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
            
            # Add channel dimension for image
            self.writer.add_image(tag, spec_db.unsqueeze(0), step)
    
    def add_histogram(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def add_metrics(self, metrics, step, prefix=""):
        """Add multiple metrics at once."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.add_scalar(tag, value, step)
    
    def add_codec_samples(self, original, reconstructed,
                         step, sample_rate=24000, prefix="samples"):
        """Add original and reconstructed audio samples with spectrograms."""
        if self.writer is not None:
            # Add audio
            self.add_audio(f"{prefix}/original", original, step, sample_rate)
            self.add_audio(f"{prefix}/reconstructed", reconstructed, step, sample_rate)
            
            # Add spectrograms
            self.add_spectrogram(f"{prefix}/spec_original", original, step, sample_rate)
            self.add_spectrogram(f"{prefix}/spec_reconstructed", reconstructed, step, sample_rate)
    
    def flush(self):
        if self.writer is not None:
            self.writer.flush()
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


class WandBLogger(object):
    """Weights & Biases logger for audio codec training."""
    
    def __init__(self, project, name=None, 
                 config=None):
        self.enabled = is_main_process()
        if self.enabled:
            import wandb
            self.wandb = wandb
            # Force finish any existing run and start fresh
            if wandb.run is not None:
                wandb.finish()
            self.run = wandb.init(project=project, name=name, config=config, resume="allow")
        else:
            self.wandb = None
            self.run = None
    
    def log(self, metrics, step):
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def log_audio(self, key, audio, sample_rate, step):
        if self.enabled:
            self.wandb.log({
                key: self.wandb.Audio(audio, sample_rate=sample_rate)
            }, step=step)
    
    def log_codec_comparison(self, original, reconstructed,
                           step, sample_rate=24000):
        """Log audio comparison to WandB."""
        if self.enabled:
            # Convert to numpy
            orig_np = original.squeeze().cpu().numpy()
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Log audios
            self.log_audio("samples/original", orig_np, sample_rate, step)
            self.log_audio("samples/reconstructed", recon_np, sample_rate, step)
    
    def finish(self):
        if self.enabled:
            self.wandb.finish()


class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed as attributes."""
    
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super(AttrDict, self).__setattr__(key, value)


def setup_logger(output_dir, name="higgs_audio_tokenizer",
                 use_tensorboard=True, use_wandb=False,
                 wandb_project=None):
    """
    Setup logging infrastructure.
    
    Args:
        output_dir: Directory for logs
        name: Experiment name
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: WandB project name
        
    Returns:
        Dictionary with logger instances
    """
    # Setup Python logging
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = {
        'metric_logger': CodecMetricLogger(),
        'output_dir': output_dir
    }
    
    if use_tensorboard and is_main_process():
        tb_dir = output_dir / 'tensorboard'
        tb_dir.mkdir(exist_ok=True)
        loggers['tensorboard'] = TensorBoardLogger(tb_dir, comment=name)
    
    if use_wandb and is_main_process():
        loggers['wandb'] = WandBLogger(
            project=wandb_project or "higgs-audio-v2-unofficial",
            name=name
        )
    
    return AttrDict(loggers)