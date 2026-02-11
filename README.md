# Aigen-R0

A reasoning-enhanced LLM fine-tuned from Mistral AI's Ministral 3 3B using GDPO (Group reward-Decoupled Policy Optimization).

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

Basic SFT with Cross-Entropy:
```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/train.json --loss_type cross_entropy --epochs 3 --freeze_until_layer 24
```

SFT with Learned Uncertainty (Kendall & Gal, 2017):
```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/train.json --loss_type heteroscedastic_uncertainty --epochs 3 --freeze_until_layer 24
```

SFT with Non-Learnable Uncertainty (sigma from logits.std()):
```bash
python training.py --model_type ministral_3_3b_instruct --data_path data/train.json --loss_type non_learnable_heteroscedastic_uncertainty --epochs 3 --freeze_until_layer 24
```

**Heteroscedastic Loss Types:**

| Loss Type | Sigma Source | Description |
|-----------|--------------|-------------|
| `heteroscedastic_uncertainty` | Learned (model output) | Model learns per-token uncertainty |
| `non_learnable_heteroscedastic_uncertainty` | `logits.std()` | Uses logit standard deviation as uncertainty |

**Full Training Example:**

Below is a comprehensive example showing commonly used training options:

```bash
python training.py --local \
    --model_type <your_model_type> \
    --loss_type <loss_type> \
    --data_path <your_data.json> \
    --epochs 100 \
    --batch_size 32 \
    --freeze_until_layer 13 \
    --val_ratio 0.3 \
    --early_stopping_patience 5 \
    --heteroscedastic_T 32 \
    --track_token_errors \
    --debug \
    --save_strategy no
```

Where:
- `<your_model_type>`: Model type (e.g., `ministral_3_3b_instruct`)
- `<loss_type>`: Loss function (`cross_entropy`, `heteroscedastic_uncertainty`, `non_learnable_heteroscedastic_uncertainty`)
- `<your_data.json>`: Training data path (e.g., `data/train.json`)

Key options:
- `--freeze_until_layer 13`: Freeze first 13 layers (train ~60% of model)
- `--val_ratio 0.3`: 30% validation split
- `--early_stopping_patience 5`: Stop if no improvement for 5 epochs
- `--heteroscedastic_T 32`: 32 Monte Carlo samples (for heteroscedastic loss types)
- `--track_token_errors`: Track per-token prediction errors
- `--save_strategy no`: Don't save intermediate checkpoints

**Heteroscedastic Uncertainty Loss:**

The model learns per-token uncertainty σ alongside predictions. The loss uses Monte Carlo sampling:

```
x̂_t = f + σ · ε_t,  where ε_t ~ N(0, I)

L = -log( (1/T) · Σ_t softmax(x̂_t)[c] )
```

Where:
- `f`: logits (model output)
- `σ = exp(log_variance / 2)`: learned standard deviation
- `ε_t`: random noise sampled from standard normal distribution
- `T`: number of Monte Carlo samples (default: 3)
- `c`: ground truth token index

This enables the model to express confidence in its predictions by learning input-dependent uncertainty.

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
| `--gdpo_sequential` | False | Use sequential processing for lower memory (slower but fits on smaller GPUs) |

**Memory Optimization:**
- By default, GDPO generates all G samples in parallel, which is fast but uses more GPU memory
- With `--gdpo_sequential`, samples are generated one at a time and stored on CPU, reducing peak GPU memory by ~G times
- Use sequential mode when encountering OOM errors during GDPO training

**Hierarchical Reward Thresholds:**

| Threshold | Default | Description |
|-----------|---------|-------------|
| `--gdpo_accuracy_threshold` | 1.0 | Accuracy threshold (binary: 1=correct, 0=incorrect) |
| `--gdpo_uncertainty_threshold` | 0.6 | Uncertainty threshold for penalty |
| `--gdpo_tool_correctness_threshold` | 1.5 | Tool correctness threshold (~75% match required) |

The hierarchical reward system works as follows:
1. **Uncertainty** (hardest): If fail, all lower rewards are zeroed
2. **Accuracy** (hard): If fail, Tool Correctness and Easy rewards are zeroed
3. **Tool Correctness** (medium): If fail, Easy rewards are zeroed
4. **Easy rewards** (Format, Length, Tool Format): Granted only if all above pass

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
| `--loss_type` | cross_entropy | Loss function: `cross_entropy` (SFT), `gdpo`, `heteroscedastic_gdpo` |
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
aigen-r0/
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

Copyright 2026 Aigen-R0 Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This project is built upon Ministral 3 3B by Mistral AI, also licensed under Apache 2.0.
