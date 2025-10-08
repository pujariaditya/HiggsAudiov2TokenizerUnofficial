# Higgs Audio V2 Tokenizer (Unofficial Implementation)

## ⚠️ Unofficial Implementation

This is an unofficial implementation of the Higgs Audio V2 Tokenizer architecture with semantic features from HuBERT. The official Higgs Audio repository can be found at [github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio), but it does not include training code. This implementation provides a complete training pipeline and integrates elements from the Descript Audio Codec (DAC) architecture.

## Attribution

This implementation is based on:
- **Higgs Audio**: [github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) - Semantic-acoustic audio tokenization with 960x downsampling (25 Hz frame rate)
- **Descript Audio Codec**: [github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)
- **Original xCodec**: [github.com/zhenye234/xcodec](https://github.com/zhenye234/xcodec)

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- CUDA-capable GPU (required for training)
- HuBERT base model for semantic features

## Installation

### Using `uv` (recommended)

Clone the repository and install the dependencies:

```bash
git clone https://github.com/pujariaditya/HiggsAudiov2TokenizerUnofficial.git
cd HiggsAudiov2TokenizerUnofficial
uv sync
```

### Without `uv`

If you're not using `uv`, you can install the package with pip:

```bash
pip install -e .
```

## Prerequisites: Base Model Setup

Download the required HuBERT model for semantic features:

```bash
# Create pretrained directory
mkdir -p pretrained

# Download HuBERT base model (required for semantic features)  
huggingface-cli download facebook/hubert-base-ls960 --local-dir pretrained/hubert-base-ls960
```

## Data Preparation

Use the automated script to prepare LibriSpeech dataset:

```bash
# Basic usage - processes all LibriSpeech subsets
uv run python scripts/prepare_dataset.py --dataset-path /path/to/LibriSpeech

# Process specific subsets only
uv run python scripts/prepare_dataset.py \
    --dataset-path /path/to/LibriSpeech \
    --subsets train-clean-100 dev-clean test-clean

# Custom output directory and parallel workers
uv run python scripts/prepare_dataset.py \
    --dataset-path /path/to/LibriSpeech \
    --output-dir ./data \
    --max-workers 16
```

**Command-line options:**
- `--dataset-path`: Path to LibriSpeech root directory (default: `./Dataset/librispeech/LibriSpeech`)
- `--output-dir`: Output directory for JSONL files (default: `./data`)
- `--subsets`: Specific subsets to process (e.g., `train-clean-100 dev-clean`)
- `--max-workers`: Number of parallel workers (default: auto-detected)

The script automatically creates three JSONL files:
- `data/train.jsonl` - Training data (train-clean-100, train-clean-360, train-other-500)
- `data/valid.jsonl` - Validation data (dev-clean, dev-other)
- `data/test.jsonl` - Test data (test-clean, test-other)

**JSONL format:** Each line contains:
```json
{"id": "librispeech_1272-128104-0000", "audio_path": "/path/to/audio.flac", "duration": 5.23}
```

## Training

### Single GPU Training

```bash
uv run audiotokenizer train --cfg-path configs/default.yml
```

### Multi-GPU Training with torchrun

```bash
# 2 GPUs on single node
uv run torchrun --nproc_per_node=2 --nnodes=1 cli.py train --cfg-path configs/default.yml

# 4 GPUs on single node
uv run torchrun --nproc_per_node=4 --nnodes=1 cli.py train --cfg-path configs/default.yml

# 8 GPUs on single node
uv run torchrun --nproc_per_node=8 --nnodes=1 cli.py train --cfg-path configs/default.yml
```

### Resume Training from Checkpoint

```bash
uv run audiotokenizer train --cfg-path configs/default.yml --options run.resume_from=outputs/higgs_audio_v2_24khz/checkpoint_latest.pth
```

### Monitor Training

Training progress is logged to:
- **WandB**: Automatically logs to project "higgs-audio-v2-tokenizer"
- **TensorBoard**: Run `tensorboard --logdir outputs/higgs_audio_v2_24khz/tensorboard`
- **Console**: Live metrics displayed every 50 iterations
- **Log file**: `outputs/higgs_audio_v2_24khz/train.log`

## Architecture

The Higgs Audio V2 Tokenizer combines acoustic and semantic features:

- **Sample Rate**: 24kHz
- **Downsampling**: 960x (8×5×4×2×3)
- **Frame Rate**: 25 Hz
- **Codebooks**: 8 RVQ codebooks with 1024 entries each
- **Semantic Model**: HuBERT base for semantic features
- **Feature Fusion**: Concatenation of acoustic and semantic embeddings

Key components:
- **Encoder**: DAC-style convolutional encoder with Snake activation
- **Semantic Encoder**: Processes HuBERT features
- **Quantizer**: 8-layer Residual Vector Quantization
- **Decoder**: Transposed convolutions for audio reconstruction
- **Discriminator**: Multi-Period, Multi-Scale, and Multi-Resolution discriminators

## Configuration

Key training parameters in `configs/default.yml`:

```yaml
model:
  sample_rate: 24000
  n_codebooks: 8          # Number of RVQ layers
  codebook_size: 1024     # Entries per codebook
  encoder_rates: [8, 5, 4, 2, 3]  # 960x downsampling
  semantic_model_path: "pretrained/hubert-base-ls960"

run:
  batch_size_train: 72    # Per GPU batch size
  batch_size_eval: 100    # Validation batch size
  max_iterations: 400000
  validation_interval: 1000
  checkpoint_interval: 5000

loss:
  mel_loss_weight: 100.0   # Primary reconstruction loss
  semantic_loss_weight: 10.0  # HuBERT feature matching
  adv_gen_loss_weight: 1.0    # Generator adversarial loss
  adv_feat_loss_weight: 2.0   # Feature matching loss
```

## Acknowledgments

This unofficial implementation builds upon:
- xCodec architecture by zhenye234
- Descript Audio Codec by Descript Inc.
- HuBERT model by Facebook/Meta