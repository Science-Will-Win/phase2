# Ministral 3 Fine-Tuning Framework

A fine-tuning framework for Mistral AI's Ministral 3 3B model.

## Requirements

- Python >= 3.11
- CUDA GPU (recommended: 24GB VRAM)
- PyTorch with CUDA

## Installation

### Automatic Setup (Recommended)

```bash
python environment.py
```

### Manual Setup

```bash
# Using uv (fast)
uv venv --python 3.11 .venv
uv pip install -e .

# Or using pip
python -m venv .venv
pip install -e .
```

### Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

## Training Pipeline: SFT → GDPO

### Why GDPO?

**SFT (Supervised Fine-Tuning)** teaches the model to imitate correct answers, but it has a critical limitation: the model only learns from "what is correct" without understanding "what is wrong."

**GDPO (Group reward-Decoupled Policy Optimization)** addresses this by:

| Problem with SFT | How GDPO Solves It |
|------------------|-------------------|
| Only learns from correct examples | Learns from both good and bad responses within a group |
| No quality differentiation | Assigns reward scores (format, accuracy, length) to each response |
| Mimics without understanding | Reinforces better responses, suppresses worse ones |

### 2-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: SFT (Supervised Fine-Tuning)                         │
│  - Purpose: Learn basic instruction-following ability          │
│  - Loss: Cross-Entropy                                         │
│  - Output: A model that can follow instructions                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: GDPO (Policy Optimization)                           │
│  - Purpose: Optimize response quality                          │
│  - Loss: GDPO (multi-reward based)                             │
│  - Output: A model with better format, accuracy, and reasoning │
└─────────────────────────────────────────────────────────────────┘
```

### Training Commands

**Stage 1: SFT**
```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/train.json --loss_type cross_entropy --epochs 3 --freeze_until_layer 24
```

**Stage 2: GDPO** (continue from SFT model)
```bash
python training.py --model_type ministral_3_3b_instruct --model_path model/train/sft-model-folder --data_path data/train.json --loss_type gdpo --epochs 2 --gdpo_group_size 4 --freeze_until_layer 24
```

### GDPO Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gdpo_group_size` | 4 | Number of responses to generate per input |
| `--gdpo_kl_coef` | 0.01 | KL divergence penalty coefficient |
| `--gdpo_use_conditioned_rewards` | False | Condition easier rewards on harder ones |
| `--gdpo_condition_threshold` | 1.0 | Threshold for conditioned rewards |

## GPU Configuration

All scripts support `--local` and `--gpu` arguments:

| Environment | Flag | GPU Default | DataParallel |
|-------------|------|-------------|--------------|
| Server | (none) | `6,7` | Auto (if 2+ GPUs) |
| Local | `--local` | `0` | Auto (if 2+ GPUs) |

The `--gpu` argument accepts any comma-separated GPU IDs. DataParallel is automatically enabled when 2 or more GPUs are specified.

```bash
# Run on server (default: GPU 6,7)
python training.py --model_type ministral_3_3b_instruct

# Run locally (default: GPU 0)
python training.py --local --model_type ministral_3_3b_instruct

# Specify GPUs explicitly (2 GPUs)
python training.py --gpu 2,3 --model_type ministral_3_3b_instruct

# Use 4 GPUs
python training.py --gpu 0,1,2,3 --model_type ministral_3_3b_instruct

# Use all 8 GPUs
python training.py --gpu 0,1,2,3,4,5,6,7 --model_type ministral_3_3b_instruct
```

## Usage

### Training

Basic training:

```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/sample.json --epochs 3 --batch_size 1 --learning_rate 2e-5
```

With layer freezing (memory efficient):

```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/sample.json --epochs 3 --freeze_until_layer 24
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_type` | - | Model type (required) |
| `--data_path` | - | Training data path |
| `--epochs` | 1 | Number of epochs |
| `--batch_size` | 2 | Batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--loss_type` | cross_entropy | Loss function (`cross_entropy`, `gdpo`, `heteroscedastic_gdpo`) |
| `--freeze_until_layer` | - | Freeze layers 0 to N (-1: freeze all) |
| `--local` | False | Use local paths |
| `--gpu` | 6,7 or 0 | GPU IDs to use |
| `--debug` | False | Enable debug output |

For detailed help:
```bash
python training.py --help
python training.py --freeze_until_layer --help  # Detailed help for specific argument
```

### Inference

Run with Web UI:

```bash
python inference.py --model ministral_3_3b_instruct
```

Run with CLI:

```bash
python inference.py --model ministral_3_3b_instruct --cli
```

Use a trained model:

```bash
python inference.py --model train/your-trained-model-folder
```

#### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | - | Model path or type (required) |
| `--max_length` | 32768 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_k` | 50 | Top-k sampling |
| `--cli` | False | CLI mode (instead of Web UI) |
| `--no_refusal` | False | Disable refusal mechanism |
| `--local` | False | Use local paths |
| `--gpu` | 6,7 or 0 | GPU IDs to use |

### RLHF Data Collection

Collect preference data with UI:

```bash
python rlhf_collect.py --data_path data/sample.json --model ministral_3_3b_instruct
```

## Memory Requirements

| Configuration | Approx. VRAM |
|---------------|--------------|
| Full fine-tune (BF16) | ~20GB |
| Freeze 20 layers | ~14GB |
| Freeze 24 layers | ~10GB |
| Inference only | ~7GB |

## Project Structure

```
pre-aiffel/
├── training.py          # Training script
├── inference.py         # Inference script (Web UI / CLI)
├── rlhf_collect.py      # RLHF data collection
├── GDPO.py              # GDPO loss function
├── model.py             # Model loader
├── architectures/       # Model architecture definitions
├── data/                # Training data
├── model/               # Model weights
├── result/              # Training logs (CSV)
└── inference_log/       # Inference conversation logs (JSON)
```

## License

This project is for educational and research purposes. Model weights are subject to Mistral AI's Apache 2.0 license.
