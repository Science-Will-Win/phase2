"""
Detailed Help System for Command Line Arguments

Usage: python script.py --arg_name --help
Example: python inference.py --refusal_threshold --help
"""

import sys

DETAILED_HELP = {
    # ===================
    # Common Arguments
    # ===================
    "model_type": """--model_type: Specify the model architecture to use

Usage:
  --model_type ministral_3_3b_instruct     Default model (3B parameters)

Note:
  - Must match a Python filename in the architectures directory
  - Use --model_path to specify custom weight location
""",

    "data_path": """--data_path: Data file path

Usage:
  --data_path sample.json             Relative path from data/ folder
  --data_path /absolute/path.json     Absolute path

Data format:
  - JSON file
  - Each item requires "instruction" or "text" field
""",

    "max_length": """--max_length: Maximum number of tokens to generate

Usage:
  --max_length 512      Short response
  --max_length 2048     Medium response
  --max_length 32768    Long response (default)

Note:
  - Must be within model's max_position_embeddings (262144)
  - Higher values increase memory usage
""",

    "temperature": """--temperature: Sampling temperature

Usage:
  --temperature 0.3     Conservative, consistent output
  --temperature 0.7     Balanced (default)
  --temperature 1.0     Creative
  --temperature 1.5+    Highly diverse but unstable

Characteristics:
  - Lower: Prefers high-probability tokens, more repetitive
  - Higher: More diverse but increased error likelihood
""",

    "temperatures": """--temperatures: Multiple temperature settings (for RLHF)

Usage:
  --temperatures 0.5,0.7,1.0,1.2      Generate 4 responses (default)
  --temperatures 0.3,0.5              Generate 2 responses

Separate responses are generated for each temperature for comparison
""",

    # ===================
    # Refusal Mechanism
    # ===================
    "refusal_threshold": """--refusal_threshold: Uncertainty threshold

When the standard deviation of logits exceeds this value during token generation,
the token is rejected and temperature is lowered.

Usage:
  --refusal_threshold 1.5    Strict (frequent rejections)
  --refusal_threshold 3.0    Default
  --refusal_threshold 5.0    Lenient (rare rejections)

Behavior:
  uncertainty > threshold → reject → temp *= temp_decay → retry
""",

    "refusal_max_retries": """--refusal_max_retries: Maximum retries per token

Usage:
  --refusal_max_retries 3    Default
  --refusal_max_retries 5    More retries

Temperature decreases with each retry (temp_decay applied)
""",

    "refusal_temp_decay": """--refusal_temp_decay: Temperature decay rate on retry

Usage:
  --refusal_temp_decay 0.8    Default (20% decrease)
  --refusal_temp_decay 0.5    Rapid decrease

Example (temp=1.0, decay=0.8):
  retry 1: 1.0 × 0.8 = 0.8
  retry 2: 0.8 × 0.8 = 0.64
  retry 3: 0.64 × 0.8 = 0.512
""",

    "refusal_min_temp": """--refusal_min_temp: Minimum temperature lower bound

Usage:
  --refusal_min_temp 0.4    Default
  --refusal_min_temp 0.1    Allow lower bound

Temperature will not drop below this value
""",

    "refusal_recovery_tokens": """--refusal_recovery_tokens: Tokens needed for temperature recovery

Usage:
  --refusal_recovery_tokens 3    Default
  --refusal_recovery_tokens 5    Slower recovery

Number of tokens required to return to original temperature after rejection
""",

    "refusal_recovery_method": """--refusal_recovery_method: Temperature recovery curve

Available:
  linear        Constant rate recovery
  exponential   Fast at first, slower later (default)
  ease_out      Fast at first, smooth at end
  ease_in_out   Smooth at both start and end
  step          Instant recovery at last token

Usage:
  --refusal_recovery_method exponential
""",

    "no_refusal": """--no_refusal: Disable refusal mechanism

Usage:
  --no_refusal

Uses standard sampling without uncertainty-based rejection
""",

    # ===================
    # Training Arguments
    # ===================
    "loss_type": """--loss_type: Loss function type

Available:
  cross_entropy
    - Standard Cross Entropy Loss
    - Basic language model training

  non_learnable_heteroscedastic_uncertainty
    - Heteroscedastic uncertainty (Kendall & Gal, 2017)
    - σ = logits.std() (computed from logits, not learned)
    - Works with any standard model architecture
    - Good for exploring uncertainty without special model heads

  heteroscedastic_uncertainty
    - Learned heteroscedastic uncertainty (Kendall & Gal, 2017)
    - σ = exp(log_variance / 2) (model learns log_variance)
    - Requires special architecture with variance prediction head
    - More expressive uncertainty modeling

  gdpo
    - Group reward-Decoupled Policy Optimization
    - 5 objectives (tool rewards enabled by default):
      Format, Length, Tool Format, Accuracy, Tool Correctness
    - Add Temperature objective with --gdpo_use_temperature_contrastive
    - Disable tool rewards with --gdpo_enable_tool_reward 0 (3 objectives)

  heteroscedastic_gdpo
    - GDPO + Uncertainty Reward (6 objectives with tool rewards):
      Format, Length, Tool Format, Accuracy, Uncertainty, Tool Correctness
    - Integrates uncertainty as reward (not separate loss)
    - Add Temperature objective with --gdpo_use_temperature_contrastive
    - Disable tool rewards for 4 objectives

Usage:
  --loss_type cross_entropy
  --loss_type non_learnable_heteroscedastic_uncertainty
  --loss_type heteroscedastic_uncertainty
  --loss_type gdpo
  --loss_type heteroscedastic_gdpo
  --loss_type heteroscedastic_gdpo --gdpo_use_temperature_contrastive

Related arguments:
  GDPO common:
    --gdpo_group_size, --gdpo_kl_coef, --gdpo_temperature
    --gdpo_reward_weight_format, --gdpo_reward_weight_length, --gdpo_reward_weight_accuracy
  
  Temperature Contrastive:
    --gdpo_use_temperature_contrastive
    --gdpo_low_temperature, --gdpo_high_temperature
    --gdpo_reward_weight_temperature
  
  Uncertainty (heteroscedastic_gdpo):
    --gdpo_uncertainty_threshold
    --gdpo_reward_weight_uncertainty
  
  Reasoning Judge:
    --gdpo_enable_reasoning_judge
    --gdpo_reward_weight_reasoning_quality
""",

    "epochs": """--epochs: Number of training epochs

Usage:
  --epochs 1    Fast training (default)
  --epochs 3    Typical fine-tuning
  --epochs 10   Small datasets

Note: Watch for overfitting
""",

    "batch_size": """--batch_size: Batch size

Usage:
  --batch_size 1    Memory saving
  --batch_size 2    Default
  --batch_size 4    Faster training (with sufficient memory)

Adjust based on GPU memory
""",

    "learning_rate": """--learning_rate: Learning rate

Usage:
  --learning_rate 2e-5    Default (recommended for fine-tuning)
  --learning_rate 1e-5    Stable
  --learning_rate 5e-5    Faster training

Too high causes divergence, too low slows learning
""",

    "gdpo_group_size": """--gdpo_group_size: GDPO group size

Usage:
  --gdpo_group_size 4    Default

Number of responses to generate per prompt
Learns preference based on rewards within the group
""",

    "gdpo_kl_coef": """--gdpo_kl_coef: KL divergence penalty coefficient

Usage:
  --gdpo_kl_coef 0.01    Default
  --gdpo_kl_coef 0.1     Strong regularization

Controls deviation from the original model
""",

    "gdpo_use_conditioned_rewards": """--gdpo_use_conditioned_rewards: Enable conditioned rewards

Usage:
  --gdpo_use_conditioned_rewards

Behavior:
  Easy rewards (Format, Length) are conditioned on difficult rewards.
  Easy rewards are only given when ALL difficult conditions are met.

  4-Level Hierarchy (when tool rewards enabled):
  1. Uncertainty (hardest): If fail, all lower rewards are zeroed
  2. Accuracy: If fail, Tool Correctness and Easy rewards are zeroed
  3. Tool Correctness: If fail, Easy rewards are zeroed
  4. Easy rewards (Format, Length, Tool Format): Granted only if all above pass

  For standard GDPO (without uncertainty), only accuracy is checked.

Related:
  --gdpo_accuracy_threshold: Accuracy threshold
  --gdpo_uncertainty_threshold: Uncertainty threshold
  --gdpo_tool_correctness_threshold: Tool correctness threshold
""",

    "gdpo_accuracy_threshold": """--gdpo_accuracy_threshold: Accuracy threshold for conditioned rewards

Usage:
  --gdpo_accuracy_threshold 1.0    Default (binary: 1=correct, 0=incorrect)
  --gdpo_accuracy_threshold 0.5    More lenient

When --gdpo_use_conditioned_rewards is enabled, easy rewards (Format, Length)
are only given if accuracy meets or exceeds this threshold.

Works in conjunction with --gdpo_uncertainty_threshold when using heteroscedastic_gdpo.
""",

    "heteroscedastic_T": """--heteroscedastic_T: Monte Carlo sample count

Usage:
  --heteroscedastic_T 3    Default (memory efficient)
  --heteroscedastic_T 10   More accurate uncertainty estimation

Higher is more accurate but increases memory/time
""",

    "random_seed": """--random_seed: Random seed

Usage:
  --random_seed -1     Random (default)
  --random_seed 42     Reproducible results

-1 for different results each time, positive for reproducibility
""",

    "debug": """--debug: Debug mode

Usage:
  --debug

Enables detailed logging:
  - Per-token generation info
  - Uncertainty values
  - Temperature changes
""",

    "top_k": """--top_k: Top-K sampling

Usage:
  --top_k 50    Default
  --top_k 10    More restrictive
  --top_k 100   More diverse

Samples only from top K probability tokens
""",

    # ===================
    # GDPO Temperature Contrastive
    # ===================
    "gdpo_use_temperature_contrastive": """--gdpo_use_temperature_contrastive: Enable Temperature Contrastive sampling

Usage:
  --gdpo_use_temperature_contrastive

Behavior:
  When enabled:
  - Generate G samples with low temperature (low_temp) → Chosen (positive, +1)
  - Generate G samples with high temperature (high_temp) → Rejected (negative, -1)
  - Train with 2G total samples

  When disabled (default):
  - Generate G samples with single temperature (gdpo_temperature)

Note:
  - Memory usage doubles (2G samples)
  - Automatic negative sampling, no manual labeling needed
""",

    "gdpo_low_temperature": """--gdpo_low_temperature: Low temperature for chosen samples

Usage:
  --gdpo_low_temperature 0.3    Default
  --gdpo_low_temperature 0.1    Very deterministic
  --gdpo_low_temperature 0.5    Slightly diverse

Characteristics:
  - Low temperature → Prefers high probability tokens → Consistent output
  - These outputs are treated as 'chosen/positive'

Only used when --gdpo_use_temperature_contrastive is enabled
""",

    "gdpo_high_temperature": """--gdpo_high_temperature: High temperature for rejected samples

Usage:
  --gdpo_high_temperature 1.2    Default
  --gdpo_high_temperature 1.0    Normal
  --gdpo_high_temperature 1.5    Very diverse

Characteristics:
  - High temperature → Flattens probability distribution → Diverse but error-prone
  - These outputs are treated as 'rejected/negative'

Only used when --gdpo_use_temperature_contrastive is enabled
""",

    "gdpo_reward_weight_temperature": """--gdpo_reward_weight_temperature: Temperature reward weight

Usage:
  --gdpo_reward_weight_temperature 1.0    Default
  --gdpo_reward_weight_temperature 0.5    Reduce temperature contrast effect

Weight for temperature reward when --gdpo_use_temperature_contrastive is enabled
""",

    # ===================
    # GDPO Uncertainty Reward
    # ===================
    "gdpo_uncertainty_threshold": """--gdpo_uncertainty_threshold: Uncertainty penalty threshold

Usage:
  --gdpo_uncertainty_threshold 0.6    Default
  --gdpo_uncertainty_threshold 0.5    Stricter
  --gdpo_uncertainty_threshold 0.8    More lenient

Behavior:
  soft_scaled_uncertainty >= threshold → Apply negative penalty
  soft_scaled_uncertainty < threshold → No penalty (0)

Soft Scaling: u* = u / (1 + |u|), range (0, 1)

Used as 4th reward objective in heteroscedastic_gdpo
""",

    "gdpo_reward_weight_uncertainty": """--gdpo_reward_weight_uncertainty: Uncertainty reward weight

Usage:
  --gdpo_reward_weight_uncertainty 1.0    Default
  --gdpo_reward_weight_uncertainty 0.5    Reduce uncertainty effect
  --gdpo_reward_weight_uncertainty 2.0    Increase uncertainty effect

Weight for uncertainty reward in advantage calculation for heteroscedastic_gdpo
""",

    # ===================
    # GDPO Memory & Additional Options
    # ===================
    "gdpo_sequential": """--gdpo_sequential: Enable sequential processing for memory optimization

Usage:
  --gdpo_sequential

Behavior:
  - Default (parallel): Generate all G samples at once (fast, high memory)
  - Sequential: Generate samples one at a time, store on CPU (slow, low memory)

Memory comparison (B=2, G=4, seq_len=2048):
  - Parallel: ~8.5GB peak GPU memory
  - Sequential: ~1.1GB peak GPU memory

Use when encountering OOM errors during GDPO training.
""",

    "gdpo_tool_correctness_threshold": """--gdpo_tool_correctness_threshold: Tool correctness threshold

Usage:
  --gdpo_tool_correctness_threshold 1.5    Default (~75% match required)
  --gdpo_tool_correctness_threshold 2.0    More lenient
  --gdpo_tool_correctness_threshold 1.0    Stricter

Tool correctness score range: -3 to +3
Threshold determines when tool rewards are conditioned in hierarchical system.

Hierarchy (from hardest to easiest):
  1. Tool Correctness <- this threshold (hardest)
  2. Accuracy
  3. Uncertainty (if enabled)
  4. Easy rewards (Format, Length, Tool Format)

Note: Tool rewards are enabled by default in GDPO.
""",

    "gdpo_enable_tool_reward": """--gdpo_enable_tool_reward: Enable/disable tool reward system

Usage:
  (default)                     Tool rewards enabled (Tool Format + Tool Correctness)
  --gdpo_enable_tool_reward 0   Disable tool rewards

Tool Reward System (enabled by default per GDPO paper):
  - Tool Format (Easy): Binary [0, 1] - checks [TOOL_CALLS] structure validity
    - [TOOL_CALLS] tag presence
    - Valid JSON parsing
    - Required fields: "name", "arguments"
  
  - Tool Correctness (Hardest): Continuous [-3, 3] - semantic correctness
    - r_name: Tool name matching (Jaccard similarity)
    - r_param: Parameter name matching
    - r_value: Parameter value matching

Reward Indices when enabled:
  0: Format (Easy)
  1: Length (Easy)
  2: Tool Format (Easy)
  3: Accuracy (Hard)
  4: Uncertainty (Medium, if heteroscedastic)
  5: Tool Correctness (Hardest)

Conditional Hierarchy:
  Tool Correctness > Accuracy > Uncertainty > Easy (Format, Length, Tool Format)
""",

    "gdpo_uncertainty_full_sequence": """--gdpo_uncertainty_full_sequence: Measure uncertainty on full sequence

Usage:
  --gdpo_uncertainty_full_sequence

Behavior:
  - Default (disabled): Measure uncertainty only in [THINK]...[/THINK] section
  - Enabled: Measure uncertainty across the entire generated sequence

Reasoning-only is recommended for most cases as it focuses on the
model's reasoning process rather than output formatting.
""",

    "gdpo_target_length": """--gdpo_target_length: Target length for length penalty

Usage:
  --gdpo_target_length 1024    Default
  --gdpo_target_length 512     Prefer shorter responses
  --gdpo_target_length 2048    Prefer longer responses

Length reward penalizes deviation from target length.
""",

    # ===================
    # Training Arguments (newly added)
    # ===================
    "val_ratio": """--val_ratio: Validation set ratio

Usage:
  --val_ratio 0.3     30% validation, 70% training (default)
  --val_ratio 0.2     20% validation, 80% training
  --val_ratio 0.0     No validation set

Note:
  - Used for early stopping and evaluation
  - Requires --early_stopping_patience > 0 to enable early stopping
""",

    "early_stopping_patience": """--early_stopping_patience: Early stopping patience

Usage:
  --early_stopping_patience 5     Stop if no improvement for 5 epochs (default)
  --early_stopping_patience 0     Disable early stopping

Note:
  - Monitors eval_loss for improvement
  - Requires val_ratio > 0 to work
""",

    "stratify": """--stratify: Stratified train/val split by data property

Usage:
  --stratify type       Split by "type" field in data
  --stratify task       Split by "task" field in data

Note:
  - Maintains same ratio of categories in train and val sets
  - If not specified, random sampling is used
""",

    "track_token_errors": """--track_token_errors: Track per-token prediction errors

Usage:
  --track_token_errors     Enable token error tracking

Output:
  - token_errors.csv with (predicted, actual, count) statistics
  - Useful for analyzing model mistakes
""",

    "log_every_n_epochs": """--log_every_n_epochs: Control detailed training log frequency

Usage:
  --log_every_n_epochs 1     Log every epoch (default)
  --log_every_n_epochs 5     Log detailed records every 5 epochs

Behavior:
  - Detailed logs: predict/label/step-loss recorded every N epochs
  - Epoch summary: avg train_loss/accuracy logged EVERY epoch
  - Validation: eval_loss/accuracy logged every epoch (unchanged)

Use case:
  - Large datasets (500K+ samples): set to 5 to reduce CSV size ~80%
  - Epoch summary still provides training progress overview
""",

    "freeze_until_layer": """--freeze_until_layer: Freeze model layers up to specified layer

Usage:
  --freeze_until_layer 24     Freeze layers 0-24 (train only layers 25+)
  --freeze_until_layer 13     Freeze layers 0-13 (train ~60% of model)

Note:
  - Reduces memory usage and training time
  - Lower layers capture general features, higher layers capture task-specific
""",

    "save_strategy": """--save_strategy: Checkpoint saving strategy

Usage:
  --save_strategy epoch     Save after each epoch (default)
  --save_strategy steps     Save every N steps (use with --save_steps)
  --save_strategy no        Don't save intermediate checkpoints
""",

    "heteroscedastic_T": """--heteroscedastic_T: Monte Carlo samples for heteroscedastic loss

Usage:
  --heteroscedastic_T 3      3 samples (default, faster)
  --heteroscedastic_T 32     32 samples (more accurate uncertainty)

Note:
  - Higher values = better uncertainty estimation but slower
  - Only used with heteroscedastic loss types
""",

    "gdpo_enable_reasoning_judge": """--gdpo_enable_reasoning_judge: Enable LLM-based reasoning quality reward

Usage:
  --gdpo_enable_reasoning_judge

Behavior:
  When enabled, an LLM judge evaluates the quality of [THINK]...[/THINK] reasoning
  for each rollout response. The score (0.0-1.0) becomes a reward at Level 3 (Medium),
  same level as Uncertainty.

  Mode is determined by config.yaml (reasoning_judge section):
  - API Mode: Set api_key or base_url to use external LLM (GPT, vLLM server)
  - Local Mode: Leave both empty to load judge model locally (stays resident in GPU)

  Default judge model: biomni-r0-32b (local)

config.yaml setup:
  reasoning_judge:
    base_url: ""          # empty = Local, URL = API mode
    api_key: ""           # empty = Local or keyless server
    model: "biomni-r0-32b"

Note:
  - Disabled by default (no impact on training unless explicitly enabled)
  - Only affects GDPO and heteroscedastic_gdpo loss types
  - Local mode keeps judge model in GPU memory throughout training
""",

    "gdpo_reward_weight_reasoning_quality": """--gdpo_reward_weight_reasoning_quality: Reasoning quality reward weight

Usage:
  --gdpo_reward_weight_reasoning_quality 1.0    Default
  --gdpo_reward_weight_reasoning_quality 0.5    Reduce reasoning quality effect
  --gdpo_reward_weight_reasoning_quality 2.0    Increase reasoning quality effect

Weight for reasoning quality reward in GDPO advantage calculation.
Only effective when --gdpo_enable_reasoning_judge is enabled.
""",

    # ===================
    # LoRA Arguments
    # ===================
    "use_lora": """--use_lora: Enable LoRA (Low-Rank Adaptation) training

Usage:
  --use_lora    Wrap the base model with LoRA adapters before training

Behavior:
  - Base model weights are frozen (not updated)
  - Only LoRA adapter weights (A, B matrices) are trained
  - log_variance_head is kept fully trainable via modules_to_save
  - ~99.7% of parameters frozen → much lower memory and faster training

Recommended for:
  - Training on a single GPU with limited VRAM
  - Starting from a base model (not a partially frozen SFT model)
  - When you want to retain base model capabilities

Note:
  - Use --lora_rank, --lora_alpha, --lora_target_modules to configure
  - Do NOT combine with --freeze_until_layer (LoRA already freezes the base)
  - Learning rate should be ~10x higher than full fine-tuning (e.g. 2e-4)
""",

    "lora_rank": """--lora_rank: LoRA rank (r)

Usage:
  --lora_rank 4     Very small, minimal expressivity
  --lora_rank 8     Memory-efficient (recommended for local GPU)
  --lora_rank 16    Default, good balance
  --lora_rank 32    Higher expressivity, more memory
  --lora_rank 64    Maximum expressivity

Memory impact:
  - Each LoRA layer adds 2 matrices: A (hidden × r) and B (r × hidden)
  - Trainable params ∝ r
  - Optimizer state ∝ r (AdamW stores m and v for each param)

Scaling law:
  - Effective rank needed depends on task complexity
  - r=8 is usually sufficient for domain adaptation
  - r=16~32 for more complex behavioral changes
""",

    "lora_alpha": """--lora_alpha: LoRA scaling factor (alpha)

Usage:
  --lora_alpha 16    Match rank (scaling=1.0)
  --lora_alpha 32    Default (scaling=2.0 when rank=16)
  --lora_alpha 64    Stronger scaling

Formula:
  effective_lr = learning_rate × (alpha / rank)

Convention:
  - Common practice: alpha = 2 × rank
  - alpha=32 with rank=16 → scaling=2.0 (amplifies LoRA updates)
  - alpha=rank → scaling=1.0 (neutral)

Note:
  - Higher alpha = stronger LoRA influence
  - If loss diverges, try lowering alpha or learning_rate
""",

    "lora_dropout": """--lora_dropout: LoRA dropout rate

Usage:
  --lora_dropout 0.0     No dropout (deterministic, faster)
  --lora_dropout 0.05    Default (light regularization)
  --lora_dropout 0.1     Stronger regularization

Behavior:
  - Applied to LoRA A matrix output during training
  - Disabled during inference (eval mode)
  - Helps prevent overfitting on small datasets

Recommendation:
  - Small datasets (< 1000 samples): 0.05~0.1
  - Large datasets: 0.0~0.05
""",

    "lora_target_modules": """--lora_target_modules: Modules to apply LoRA adapters to

Usage:
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    Default: all attention + MLP projections (7 module types × 26 layers = 182 modules)

  --lora_target_modules "q_proj,v_proj"
    Attention-only, minimal (common in original LoRA paper)

  --lora_target_modules "q_proj,k_proj,v_proj,o_proj"
    Attention-only, full

Module types in Ministral 3B:
  Attention:  q_proj (3072×3072), k_proj (3072×1024), v_proj (3072×1024), o_proj (3072×3072)
  MLP:        gate_proj (3072×8192), up_proj (3072×8192), down_proj (8192×3072)

Memory vs expressivity:
  "q_proj,v_proj"          → ~5M trainable params  (minimal)
  "q_proj,k,v,o_proj"      → ~9M trainable params  (attention only)
  all 7 (default)          → ~11M trainable params (full)

Note: Comma-separated, no spaces
""",

    "lora_bias": """--lora_bias: Whether to train bias parameters in LoRA layers

Usage:
  --lora_bias none        Default: don't train any biases
  --lora_bias lora_only   Train biases in LoRA layers only
  --lora_bias all         Train all biases in the model

Recommendation:
  - none: Default, safest and most memory-efficient
  - lora_only: Slight expressivity boost with minimal overhead
  - all: Rarely needed, increases trainable params significantly
""",

    "heteroscedastic_sequential": """--heteroscedastic_sequential: Use sequential MC sampling

Usage:
  (default)                        Parallel processing (faster, more memory)
  --heteroscedastic_sequential     Sequential processing (slower, less memory)

Behavior:
  - Parallel (default): All T samples computed simultaneously
    - Fast (vectorized operations)
    - Memory: O(T * batch * seq_len * vocab_size)
  
  - Sequential: T samples computed one at a time
    - Slower (T iterations of loop)
    - Memory: O(batch * seq_len * vocab_size)

Memory comparison (T=32, vocab=128K, seq=2048, bf16):
  - Parallel: ~32GB peak GPU memory
  - Sequential: ~1GB peak GPU memory

Note:
  - Use sequential if running out of GPU memory with high --heteroscedastic_T
  - Results are mathematically identical (only performance differs)
  - Affects: non_learnable_heteroscedastic_uncertainty, heteroscedastic_uncertainty, heteroscedastic_gdpo
""",
}


def check_detailed_help():
    """
    Check for --arg_name --help pattern before argparse runs.
    Call this at the start of your script, before argparse.
    
    Usage:
        from utils.detailed_help import check_detailed_help
        check_detailed_help()
        
        # Then your normal argparse code
        parser = argparse.ArgumentParser(...)
    """
    if len(sys.argv) == 3 and sys.argv[2] == '--help' and sys.argv[1].startswith('--'):
        arg_name = sys.argv[1].lstrip('-').replace('-', '_')
        
        if arg_name in DETAILED_HELP:
            print(DETAILED_HELP[arg_name])
            sys.exit(0)
        else:
            print(f"No detailed help available for '--{arg_name.replace('_', '-')}'")
            print()
            print("Available detailed help topics:")
            for key in sorted(DETAILED_HELP.keys()):
                print(f"  --{key.replace('_', '-')}")
            sys.exit(1)


def get_available_help_topics():
    """Return list of arguments that have detailed help."""
    return sorted(DETAILED_HELP.keys())
