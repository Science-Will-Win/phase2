import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Optional, Tuple, Dict, Any, List

from training_logger import LossResult
from architectures.ministral_3_3b_instruct import Ministral3TokenConfig

# Import will be done lazily to avoid circular import
# from loss import compute_heteroscedastic_log_probs


# =============================================================================
# GDPO Base Class - Common functionality for all GDPO variants
# =============================================================================

class GDPOBase:
    """
    Base class for all GDPO variants.
    Provides common functionality for generation, rewards, and advantage calculation.
    
    Supports optional temperature contrastive sampling where low temperature
    samples are treated as positive and high temperature samples as negative.
    """
    
    def __init__(self, trainer: Any) -> None:
        """
        Initialize GDPO base with trainer configuration.
        
        Args:
            trainer: Trainer object containing gdpo_config, tokenizer, etc.
        """
        self.trainer = trainer
        self.tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        self.gdpo_config = getattr(trainer, "gdpo_config", {})
        self.ref_model = getattr(trainer, "ref_model", None)
        self.debug = getattr(trainer, "debug", False)
        
        # Common config
        self.G = self.gdpo_config.get("group_size", 4)
        self.max_new_tokens = self.gdpo_config.get("max_new_tokens", 128)
        self.kl_coef = self.gdpo_config.get("kl_coef", 0.01)
        
        # Temperature Contrastive config
        self.use_temp_contrastive = self.gdpo_config.get("use_temperature_contrastive", False)
        self.low_temp = self.gdpo_config.get("low_temperature", 0.3)
        self.high_temp = self.gdpo_config.get("high_temperature", 1.2)
        self.default_temp = self.gdpo_config.get("temperature", 1.0)
        
        # Reward config
        self.use_conditioned_rewards = self.gdpo_config.get("use_conditioned_rewards", False)
        self.condition_threshold = self.gdpo_config.get("condition_threshold", 1.0)
        self.target_length = self.gdpo_config.get("target_length", 1024)
    
    def generate_samples(
        self, 
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Generate samples with optional temperature contrastive.
        
        Args:
            model: The language model
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
        
        Returns:
            sequences: Generated sequences (B * effective_G, seq_len)
            temperature_rewards: Temperature labels or None (B * effective_G,)
            effective_G: Actual group size (G or 2G)
        """
        B = input_ids.shape[0]
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": self.G,
            "use_cache": False,
        }
        
        with torch.no_grad():
            if self.use_temp_contrastive:
                # Low temp generation (chosen/positive)
                gen_kwargs["temperature"] = self.low_temp
                low_sequences = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                )
                
                # High temp generation (rejected/negative)
                gen_kwargs["temperature"] = self.high_temp
                high_sequences = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                )
                
                sequences = torch.cat([low_sequences, high_sequences], dim=0)
                temperature_rewards = torch.cat([
                    torch.ones(B * self.G),      # Low temp → +1
                    -torch.ones(B * self.G)      # High temp → -1
                ])
                effective_G = 2 * self.G
            else:
                gen_kwargs["temperature"] = self.default_temp
                sequences = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                )
                temperature_rewards = None
                effective_G = self.G
        
        return sequences, temperature_rewards, effective_G
    
    def prepare_references(
        self, 
        inputs: Dict[str, torch.Tensor], 
        effective_G: int
    ) -> Optional[List[str]]:
        """
        Prepare expanded references for reward calculation.
        
        Args:
            inputs: Input dict containing optional 'labels'
            effective_G: Effective group size (G or 2G)
        
        Returns:
            expanded_refs: List of reference strings expanded to match batch size, or None
        """
        expanded_refs = None
        
        if "labels" in inputs:
            valid_labels = inputs["labels"].clone()
            valid_labels[valid_labels == -100] = self.tokenizer.pad_token_id
            references = self.tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
            
            expanded_refs = []
            for ref in references:
                expanded_refs.extend([ref] * effective_G)
        
        return expanded_refs
    
    def get_reward_weights(self, num_objectives: int) -> List[float]:
        """
        Get reward weights as a list.
        
        Args:
            num_objectives: Number of objectives
        
        Returns:
            List of weights for each objective
        """
        weights_config = self.gdpo_config.get("reward_weights", {})
        
        weights = [
            weights_config.get("format", 1.0),
            weights_config.get("length", 1.0),
            weights_config.get("accuracy", 1.0),
        ]
        
        # Add more weights based on num_objectives
        if num_objectives >= 4:
            weights.append(weights_config.get("uncertainty", 1.0))
        if num_objectives >= 5:
            weights.append(weights_config.get("temperature", 1.0))
        
        return weights[:num_objectives]
    
    def compute_advantages(
        self, 
        rewards: torch.Tensor, 
        weights: List[float], 
        device: torch.device
    ) -> torch.Tensor:
        """
        GDPO Advantage calculation with decoupled normalization.
        
        Args:
            rewards: Reward tensor (B, G, n)
            weights: List of n weights
            device: Torch device
        
        Returns:
            final_advantages_flat: Flattened advantages (B*G,)
        """
        eps = 1e-8
        B, G, n = rewards.shape
        
        # Per-objective normalization
        means = rewards.mean(dim=1, keepdim=True)
        stds = rewards.std(dim=1, keepdim=True)
        advantages_k = (rewards - means) / (stds + eps)
        
        # Weighted sum
        weights_tensor = torch.tensor(weights, device=device, dtype=rewards.dtype)
        advantages_sum = (advantages_k * weights_tensor).sum(dim=2)
        
        # Batch normalization
        bn_mean = advantages_sum.mean()
        bn_std = advantages_sum.std()
        final_advantages = (advantages_sum - bn_mean) / (bn_std + eps)
        
        return final_advantages.view(-1)
    
    def compute_log_probs(
        self, 
        model: torch.nn.Module, 
        sequences: torch.Tensor, 
        input_len: int
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for policy gradient.
        
        Args:
            model: The language model
            sequences: Generated sequences
            input_len: Length of input (to mask)
        
        Returns:
            outputs: Model outputs
            logits: Raw logits
            shift_logits: Shifted logits for loss
            shift_labels: Shifted labels
            token_log_probs: Per-token log probabilities
            valid_mask: Mask for valid (non-padding) tokens
        """
        train_attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        
        labels = sequences.clone()
        labels[:, :input_len] = -100
        
        outputs = model(input_ids=sequences, attention_mask=train_attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != -100).float()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_log_probs = -token_nll.view(sequences.shape[0], -1)
        
        return outputs, logits, shift_logits, shift_labels, token_log_probs, valid_mask
    
    def compute_kl_penalty(
        self, 
        ref_model: Optional[torch.nn.Module], 
        sequences: torch.Tensor, 
        token_log_probs: torch.Tensor, 
        shift_labels: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Compute KL divergence penalty.
        
        Args:
            ref_model: Reference model (or None)
            sequences: Generated sequences
            token_log_probs: Log probs from policy model
            shift_labels: Shifted labels
            valid_mask: Valid token mask
        
        Returns:
            kl_penalty: KL penalty value (scalar or 0.0)
            kl: Per-token KL values (or None)
        """
        if ref_model is None:
            return 0.0, None
        
        eps = 1e-8
        train_attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=sequences, attention_mask=train_attention_mask)
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[..., :-1, :].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            ref_token_nll = loss_fct(ref_shift_logits.view(-1, ref_shift_logits.size(-1)), shift_labels.view(-1))
            ref_token_log_probs = -ref_token_nll.view(sequences.shape[0], -1)
            
            kl = token_log_probs - ref_token_log_probs
            kl_penalty = self.kl_coef * (kl * valid_mask).sum() / (valid_mask.sum() + eps)
        
        return kl_penalty, kl
    
    def build_reward_config(self) -> Dict[str, Any]:
        """Build reward configuration dict."""
        return {
            "use_conditioned_rewards": self.use_conditioned_rewards,
            "condition_threshold": self.condition_threshold,
            "target_length": self.target_length,
        }


def soft_scale(x):
    """
    Soft scaling: a* = x / (1 + |x|)
    Maps values to (-1, 1) range smoothly.
    
    For positive x (like sigma): maps to (0, 1)
    
    Args:
        x: Input value (tensor or scalar)
        
    Returns:
        Soft scaled value in (-1, 1) range
    """
    if isinstance(x, torch.Tensor):
        return x / (1 + torch.abs(x))
    return x / (1 + abs(x))


def condition_reward(easy_reward, hard_reward, threshold=1.0):
    """
    Condition easier reward on harder reward (Paper Eq. 8).
    
    The easier reward is only given if the harder reward meets or exceeds the threshold.
    This forces the model to prioritize the harder objective first.
    
    Args:
        easy_reward: The easier reward value (e.g., format, length)
        hard_reward: The harder reward value (e.g., accuracy)
        threshold: Minimum hard_reward value required to receive easy_reward
        
    Returns:
        easy_reward if hard_reward >= threshold, else 0.0
    """
    if hard_reward >= threshold:
        return easy_reward
    return 0.0


def compute_rewards(sequences, tokenizer, references=None, reward_config=None, 
                    token_config=None, num_objectives=3,
                    uncertainty_scores=None,
                    temperature_rewards=None):
    """
    Computes rewards based on GDPO paper objectives:
    1. Format Reward: Checks for [THINK]...[/THINK]<SPECIAL_36>...<SPECIAL_37> structure.
    2. Length Penalty: Penalizes excessive length (Length Explosion).
    3. Accuracy Reward: Checks against ground truth sequences (if provided).
    4. Uncertainty Reward: Penalizes high uncertainty (optional, when uncertainty_scores provided).
    5. Temperature Reward: Positive for low temp, negative for high temp (optional).
    
    Args:
        sequences: Generated token sequences
        tokenizer: Tokenizer for decoding
        references: Ground truth references for accuracy check
        reward_config: Configuration dict with keys:
            - use_conditioned_rewards (bool): Whether to condition easier rewards on accuracy
            - condition_threshold (float): Threshold for accuracy to receive other rewards
            - target_length (int): Target length for length penalty calculation
            - uncertainty_threshold (float): Threshold for uncertainty penalty (default 0.6)
        token_config: Token configuration class (default: Ministral3TokenConfig)
        num_objectives: Number of reward objectives (default 3)
        uncertainty_scores: Optional (Batch*Group,) tensor of soft-scaled uncertainty scores
        temperature_rewards: Optional (Batch*Group,) tensor of temperature labels (+1 for low, -1 for high)
    
    Returns: (Batch * Group, num_objectives)
    """
    # Auto-adjust num_objectives based on provided scores
    if uncertainty_scores is not None:
        num_objectives = max(num_objectives, 4)
    if temperature_rewards is not None:
        temp_obj_idx = 4 if uncertainty_scores is not None else 3
        num_objectives = max(num_objectives, temp_obj_idx + 1)
    
    batch_size = sequences.shape[0]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    
    rewards = torch.zeros(batch_size, num_objectives)
    
    # Set token config defaults
    if token_config is None:
        token_config = Ministral3TokenConfig
    
    # Validate required THINK tokens
    if not token_config.THINK_START or not token_config.THINK_END:
        raise ValueError(
            f"GDPO requires THINK tokens. "
            f"THINK_START={token_config.THINK_START}, THINK_END={token_config.THINK_END}"
        )
    
    # Parse reward_config
    if reward_config is None:
        reward_config = {}
    use_conditioned_rewards = reward_config.get("use_conditioned_rewards", False)
    condition_threshold = reward_config.get("condition_threshold", 1.0)
    target_len = reward_config.get("target_length", 1024)
    
    # Generate regex pattern for format reward
    think_pattern = rf"{re.escape(token_config.THINK_START)}\s*\S+.*?{re.escape(token_config.THINK_END)}"
    
    if token_config.ANSWER_START and token_config.ANSWER_END:
        answer_pattern = rf"\s*{re.escape(token_config.ANSWER_START)}\s*\S+.*?{re.escape(token_config.ANSWER_END)}"
        format_pattern = think_pattern + answer_pattern
    else:
        format_pattern = think_pattern
    
    format_regex = re.compile(format_pattern, re.DOTALL)
    
    for i, text in enumerate(texts):
        # 1. Format Reward
        # Strict adherence to the tag structure
        if format_regex.search(text):
            format_score = 1.0
        else:
            format_score = 0.0
            
        # 2. Length Penalty
        # Paper suggests penalizing very long sequences to prevent "Length Explosion".
        # We can use a soft penalty or a hard cap.
        # Let's use a soft penalty: reward decreases as length exceeds a threshold.
        seq_len = len(sequences[i])
        if seq_len > target_len:
            # Decay reward beyond target length
            length_score = max(0.0, 1.0 - ((seq_len - target_len) / target_len))
        else:
            length_score = 1.0
        
        # 3. Accuracy Reward (Hard reward - computed first as it may condition others)
        # If references (ground truth texts) are provided, we check for containment or exact match.
        # For now, we assume simple containment of the reference answer in the generated <answer> block.
        acc_score = 0.0
        if references and i < len(references):
            ref_text = references[i]  # This might need alignment with Batch*Group
            # In a real scenario, we'd parse the <answer> part and compare.
            # Simplified: Check if reference is inside generation.
            if ref_text in text:
                acc_score = 1.0
        
        # 4. Apply Conditioned Rewards (Paper Eq. 8)
        # Easier rewards (format, length) are conditioned on harder rewards (accuracy AND uncertainty)
        if use_conditioned_rewards:
            # Check accuracy condition (difficult reward 1)
            acc_condition_met = (acc_score >= condition_threshold)
            
            # Check uncertainty condition (difficult reward 2) - only if uncertainty provided
            uncertainty_condition_met = True  # default: no uncertainty check
            if uncertainty_scores is not None and i < len(uncertainty_scores):
                u = uncertainty_scores[i]
                if isinstance(u, torch.Tensor):
                    u = u.item()
                uncertainty_threshold = reward_config.get("uncertainty_threshold", 0.6)
                uncertainty_condition_met = (u < uncertainty_threshold)
            
            # AND condition: both difficult rewards must meet their thresholds
            difficult_condition_met = acc_condition_met and uncertainty_condition_met
            
            if not difficult_condition_met:
                format_score = 0.0
                length_score = 0.0
        
        rewards[i, 0] = format_score
        rewards[i, 1] = length_score
        rewards[i, 2] = acc_score
        
        # 5. Uncertainty Reward (when uncertainty_scores provided)
        if uncertainty_scores is not None:
            uncertainty_threshold = reward_config.get("uncertainty_threshold", 0.6)
            if i < len(uncertainty_scores):
                u = uncertainty_scores[i]
                if isinstance(u, torch.Tensor):
                    u = u.item()
                # Penalize if uncertainty exceeds threshold
                if u >= uncertainty_threshold:
                    rewards[i, 3] = -u  # Penalty (negative)
                else:
                    rewards[i, 3] = 0.0
        
        # 6. Temperature Contrastive Reward (when temperature_rewards provided)
        if temperature_rewards is not None:
            temp_obj_idx = 4 if uncertainty_scores is not None else 3
            if i < len(temperature_rewards):
                t = temperature_rewards[i]
                if isinstance(t, torch.Tensor):
                    t = t.item()
                rewards[i, temp_obj_idx] = t
        
    return rewards


# =============================================================================
# GDPO Loss Classes
# =============================================================================

class GDPOLoss(GDPOBase):
    """
    Standard GDPO with 3 objectives (Format, Length, Accuracy).
    Optionally supports temperature contrastive (adds 4th objective).
    """
    
    def __call__(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """
        Compute GDPO loss.
        
        Args:
            model: Policy model
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels'
            ref_model: Reference model for KL penalty (optional)
        
        Returns:
            LossResult with total_loss, components, and outputs
        """
        if ref_model is None:
            ref_model = self.ref_model
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        
        # 1. Generate samples
        sequences, temp_rewards, effective_G = self.generate_samples(model, input_ids, attention_mask)
        
        # 2. Prepare references
        expanded_refs = self.prepare_references(inputs, effective_G)
        
        # 3. Determine num_objectives
        n = 4 if temp_rewards is not None else 3
        
        # 4. Compute rewards
        reward_config = self.build_reward_config()
        
        rewards_flat = compute_rewards(
            sequences, self.tokenizer,
            references=expanded_refs,
            reward_config=reward_config,
            token_config=Ministral3TokenConfig,
            num_objectives=n,
            temperature_rewards=temp_rewards
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 5. Compute advantages
        weights = self.get_reward_weights(n)
        final_advantages_flat = self.compute_advantages(rewards, weights, model.device)
        
        # 6. Forward pass & log probs
        input_len = input_ids.shape[1]
        outputs, logits, shift_logits, shift_labels, token_log_probs, valid_mask = \
            self.compute_log_probs(model, sequences, input_len)
        
        # 7. KL penalty
        kl_penalty, kl = self.compute_kl_penalty(
            ref_model, sequences, token_log_probs, shift_labels, valid_mask
        )
        
        # 8. Policy gradient loss
        seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
        pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
        
        total_loss = pg_loss + kl_penalty
        
        # 9. Logging
        reward_means = rewards.mean(dim=(0, 1))
        kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
        
        components = {
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "reward_accuracy": reward_means[2].item(),
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[3].item()
        
        # Memory cleanup
        del sequences, rewards_flat, rewards
        del logits, shift_logits, shift_labels, token_log_probs
        if ref_model and kl is not None:
            del kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return LossResult(total_loss=total_loss, components=components, outputs=outputs)


class HeteroscedasticGDPOLoss(GDPOBase):
    """
    GDPO with Uncertainty Reward (4 objectives: Format, Length, Accuracy, Uncertainty).
    Optionally supports temperature contrastive (adds 5th objective).
    """
    
    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        self.uncertainty_threshold = self.gdpo_config.get("uncertainty_threshold", 0.6)
    
    def compute_uncertainty(
        self, 
        shift_logits: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty scores from logits std.
        
        Args:
            shift_logits: Shifted logits
            valid_mask: Valid token mask
        
        Returns:
            sigma_per_sample: Raw sigma per sample
            uncertainty_scores: Soft-scaled uncertainty scores
        """
        eps = 1e-8
        sigma_per_token = shift_logits.std(dim=-1)
        sigma_per_sample = (sigma_per_token * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + eps)
        uncertainty_scores = soft_scale(sigma_per_sample)
        return sigma_per_sample, uncertainty_scores
    
    def __call__(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """
        Compute Heteroscedastic GDPO loss with uncertainty reward.
        
        Args:
            model: Policy model
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels'
            ref_model: Reference model for KL penalty (optional)
        
        Returns:
            LossResult with total_loss, components, and outputs
        """
        if ref_model is None:
            ref_model = self.ref_model
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        
        # 1. Generate samples
        sequences, temp_rewards, effective_G = self.generate_samples(model, input_ids, attention_mask)
        
        # 2. Prepare references
        expanded_refs = self.prepare_references(inputs, effective_G)
        
        # 3. Forward pass FIRST (to compute uncertainty)
        input_len = input_ids.shape[1]
        outputs, logits, shift_logits, shift_labels, token_log_probs, valid_mask = \
            self.compute_log_probs(model, sequences, input_len)
        
        # 4. Compute uncertainty
        sigma_per_sample, uncertainty_scores = self.compute_uncertainty(shift_logits, valid_mask)
        
        # 5. Determine num_objectives
        n = 5 if temp_rewards is not None else 4
        
        # 6. Compute rewards
        reward_config = self.build_reward_config()
        reward_config["uncertainty_threshold"] = self.uncertainty_threshold
        
        rewards_flat = compute_rewards(
            sequences, self.tokenizer,
            references=expanded_refs,
            reward_config=reward_config,
            token_config=Ministral3TokenConfig,
            num_objectives=n,
            uncertainty_scores=uncertainty_scores.detach().cpu(),
            temperature_rewards=temp_rewards
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 7. Compute advantages
        weights = self.get_reward_weights(n)
        final_advantages_flat = self.compute_advantages(rewards, weights, model.device)
        
        # 8. KL penalty
        kl_penalty, kl = self.compute_kl_penalty(
            ref_model, sequences, token_log_probs, shift_labels, valid_mask
        )
        
        # 9. Policy gradient loss
        seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
        pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
        
        total_loss = pg_loss + kl_penalty
        
        # 10. Logging
        reward_means = rewards.mean(dim=(0, 1))
        kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
        
        components = {
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "reward_accuracy": reward_means[2].item(),
            "reward_uncertainty": reward_means[3].item(),
            "uncertainty_mean": uncertainty_scores.mean().item(),
            "uncertainty_std": uncertainty_scores.std().item(),
            "sigma_mean": sigma_per_sample.mean().item(),
            "sigma_std": sigma_per_sample.std().item(),
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[4].item()
        
        # Memory cleanup
        del sequences, rewards_flat, rewards
        del logits, shift_logits, shift_labels, token_log_probs
        del sigma_per_sample, uncertainty_scores
        if ref_model and kl is not None:
            del kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return LossResult(total_loss=total_loss, components=components, outputs=outputs)


# =============================================================================
# Legacy Function Wrappers (for backward compatibility with caching)
# =============================================================================

def compute_gdpo_loss(model, inputs, trainer, ref_model=None):
    """
    Legacy wrapper for GDPOLoss class.
    
    Caches the loss instance on trainer to avoid repeated instantiation.
    Implements Group reward-Decoupled Normalization Policy Optimization (GDPO).
    
    Args:
        model: Policy model
        inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels'
        trainer: Trainer object with gdpo_config
        ref_model: Reference model for KL penalty (optional)
    
    Returns:
        LossResult with total_loss, components, and outputs
    """
    if not hasattr(trainer, '_gdpo_loss_instance'):
        trainer._gdpo_loss_instance = GDPOLoss(trainer)
    return trainer._gdpo_loss_instance(model, inputs, ref_model)


def compute_heteroscedastic_gdpo_loss(model, inputs, trainer, ref_model=None):
    """
    Legacy wrapper for HeteroscedasticGDPOLoss class.
    
    Caches the loss instance on trainer to avoid repeated instantiation.
    GDPO with Uncertainty Reward (4 objectives: Format, Length, Accuracy, Uncertainty).
    
    Args:
        model: Policy model
        inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels'
        trainer: Trainer object with gdpo_config
        ref_model: Reference model for KL penalty (optional)
    
    Returns:
        LossResult with total_loss, components, and outputs
    """
    if not hasattr(trainer, '_hetero_gdpo_loss_instance'):
        trainer._hetero_gdpo_loss_instance = HeteroscedasticGDPOLoss(trainer)
    return trainer._hetero_gdpo_loss_instance(model, inputs, ref_model)
