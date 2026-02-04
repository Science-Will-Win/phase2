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
    "model": """--model: Specify the model to use

Usage:
  --model ministral_3_3b_instruct     Default model
  --model train/my_finetuned          Fine-tuned model (searched in model/train/)

Model path search order:
  1. model/{model_name}/
  2. model/train/{model_name}/
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

  heteroscedastic_cross_entropy
    - Includes uncertainty learning (Kendall & Gal, 2017)
    - Model learns its own uncertainty

  gdpo
    - Group reward-Decoupled Policy Optimization
    - 3 objectives: Format, Length, Accuracy
    - Add 4th objective (Temperature) with --gdpo_use_temperature_contrastive

  heteroscedastic_gdpo
    - GDPO + Uncertainty Reward (4 objectives)
    - Format, Length, Accuracy, Uncertainty
    - Integrates uncertainty as reward (not separate loss)
    - Add 5th objective (Temperature) with --gdpo_use_temperature_contrastive

Usage:
  --loss_type cross_entropy
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

  Difficult rewards and conditions:
  1. Accuracy >= condition_threshold (always checked)
  2. Uncertainty < uncertainty_threshold (only when heteroscedastic_gdpo)

  AND Logic (both must be satisfied):
  | Accuracy | Uncertainty | Easy Rewards |
  |----------|-------------|--------------|
  | High     | Low         | Given        |
  | High     | High        | NOT Given    |
  | Low      | Low         | NOT Given    |
  | Low      | High        | NOT Given    |

  For standard GDPO (without uncertainty), only accuracy is checked.

Related:
  --gdpo_condition_threshold: Accuracy threshold
  --gdpo_uncertainty_threshold: Uncertainty threshold
""",

    "gdpo_condition_threshold": """--gdpo_condition_threshold: Accuracy threshold for conditioned rewards

Usage:
  --gdpo_condition_threshold 1.0    Default (requires perfect accuracy)
  --gdpo_condition_threshold 0.5    More lenient

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
