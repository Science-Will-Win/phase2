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
        
        # Tool Reward config
        self.enable_tool_reward = self.gdpo_config.get("enable_tool_reward", False)
        self.tool_correctness_threshold = self.gdpo_config.get("tool_correctness_threshold", 0.0)
    
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
    
    def prepare_gt_tool_calls(
        self, 
        inputs: Dict[str, Any], 
        effective_G: int
    ) -> Optional[List[str]]:
        """
        Prepare expanded ground truth tool calls for reward calculation.
        
        Args:
            inputs: Input dict containing optional 'gt_tool_calls' (list of strings)
            effective_G: Effective group size (G or 2G)
        
        Returns:
            expanded_gt_tools: List of GT tool call strings expanded to match batch size, or None
        """
        if not self.enable_tool_reward:
            return None
        
        gt_tool_calls = inputs.get("gt_tool_calls", None)
        if gt_tool_calls is None:
            return None
        
        expanded_gt_tools = []
        for gt in gt_tool_calls:
            expanded_gt_tools.extend([gt] * effective_G)
        
        return expanded_gt_tools
    
    def get_reward_weights(self, num_objectives: int, has_uncertainty: bool = False, 
                           has_tool_reward: bool = False) -> List[float]:
        """
        Get reward weights as a list.
        
        Objective indices depend on which optional rewards are enabled:
        - 0: Format (Easy)
        - 1: Length (Easy)
        - 2: Accuracy (Hard)
        - 3: Uncertainty (Hardest) - if enabled
        - 4/3: Tool Correctness (Medium) - if enabled
        - 5/4: Tool Format (Easy) - if enabled
        - Last: Temperature - if enabled
        
        Args:
            num_objectives: Number of objectives
            has_uncertainty: Whether uncertainty reward is enabled
            has_tool_reward: Whether tool rewards are enabled
        
        Returns:
            List of weights for each objective
        """
        weights_config = self.gdpo_config.get("reward_weights", {})
        
        # Base weights: Format, Length, Accuracy
        weights = [
            weights_config.get("format", 1.0),
            weights_config.get("length", 1.0),
            weights_config.get("accuracy", 1.0),
        ]
        
        # Add Uncertainty weight (index 3)
        if has_uncertainty:
            weights.append(weights_config.get("uncertainty", 1.0))
        
        # Add Tool weights (indices 4,5 or 3,4)
        if has_tool_reward:
            weights.append(weights_config.get("tool_correctness", 1.0))
            weights.append(weights_config.get("tool_format", 1.0))
        
        # Temperature weight is added at the end if needed
        if len(weights) < num_objectives:
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
            "tool_correctness_threshold": self.tool_correctness_threshold,
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


# =============================================================================
# Tool Use Reward Functions
# =============================================================================

# Regex patterns for tool call parsing (from tools/adapters/mistral.py)
TOOL_CALL_PATTERN_WITH_ARGS = r'\[TOOL_CALLS\](\w+)\[ARGS\](\{.*?\})(?=\[TOOL_CALLS\]|$|\s*$)'
TOOL_CALL_PATTERN_NO_ARGS = r'\[TOOL_CALLS\](\w+)(\{.*?\})(?=\[TOOL_CALLS\]|$|\s*$)'
# Simple pattern for format checking (no capture groups)
TOOL_CALL_PATTERN_SIMPLE = r'\[TOOL_CALLS\]\w+(?:\[ARGS\])?\{.*?\}'


def parse_tool_calls_for_reward(text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from text for reward computation.
    
    Parses [TOOL_CALLS]tool_name[ARGS]{...} format used by Mistral models.
    
    Args:
        text: LLM output text containing tool calls
        
    Returns:
        List of tool calls: [{"name": str, "arguments": dict}, ...]
    """
    import json
    
    tool_calls = []
    
    # Try pattern with [ARGS] first
    matches = re.findall(TOOL_CALL_PATTERN_WITH_ARGS, text, re.DOTALL)
    
    # If no matches, try pattern without [ARGS]
    if not matches:
        matches = re.findall(TOOL_CALL_PATTERN_NO_ARGS, text, re.DOTALL)
    
    for name, args_str in matches:
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            # Store empty dict if JSON parsing fails
            arguments = {}
        
        tool_calls.append({
            "name": name,
            "arguments": arguments
        })
    
    return tool_calls


def compute_tool_format_reward(text: str, token_config) -> float:
    """
    Tool Format Reward - checks [TOOL_CALLS] appears after [/THINK].
    
    Note: [THINK]...[/THINK] structure is verified by Format Reward (Index 0).
    This only checks tool call placement and format.
    
    Args:
        text: LLM output text
        token_config: Token configuration class with THINK_END
        
    Returns:
        1.0 if tool call appears after [/THINK], 0.0 otherwise
    """
    think_end = token_config.THINK_END
    
    if not think_end:
        # If no think tokens configured, just check for tool call
        if '[TOOL_CALLS]' in text:
            return 1.0
        return 0.0
    
    # Find position of [/THINK]
    think_end_pos = text.find(think_end)
    if think_end_pos == -1:
        return 0.0  # No [/THINK] found
    
    # Check for [TOOL_CALLS]tool_name{...} after [/THINK]
    after_think = text[think_end_pos + len(think_end):]
    
    if re.search(TOOL_CALL_PATTERN_SIMPLE, after_think, re.DOTALL):
        return 1.0
    return 0.0


def compute_tool_correctness_reward(
    predicted_calls: List[Dict[str, Any]], 
    gt_calls: List[Dict[str, Any]]
) -> float:
    """
    Tool Correctness Reward (R_correct ∈ [-3, 3]) - GDPO Appendix C.
    
    Evaluates predicted tool calls against ground-truth calls using three components:
    1. r_name: Tool name matching (Jaccard similarity)
    2. r_param: Parameter name matching (Jaccard similarity per matched tool)
    3. r_value: Parameter value matching (exact match count)
    
    This is a "Medium" level reward in the 4-level hierarchy.
    
    Args:
        predicted_calls: List of predicted tool calls [{"name": str, "arguments": dict}, ...]
        gt_calls: List of ground truth tool calls [{"name": str, "arguments": dict}, ...]
        
    Returns:
        R_correct ∈ [-3, 3]: 6 * (R_max / S_max) - 3
    """
    if not gt_calls:
        # No ground truth provided, return neutral score
        return 0.0
    
    if not predicted_calls:
        # No predictions, return minimum score
        return -3.0
    
    # Extract tool names
    gt_names = set(call.get("name", "") for call in gt_calls)
    pred_names = set(call.get("name", "") for call in predicted_calls)
    
    # 1. Tool Name Matching: r_name = |N_G ∩ N_P| / |N_G ∪ N_P|
    intersection = gt_names & pred_names
    union = gt_names | pred_names
    r_name = len(intersection) / len(union) if union else 0.0
    
    # 2. Parameter Name Matching and 3. Parameter Value Matching
    # Match predicted calls to ground truth calls by tool name
    r_param = 0.0
    r_value = 0.0
    
    # Create lookup for predicted calls by name
    pred_by_name = {}
    for call in predicted_calls:
        name = call.get("name", "")
        if name not in pred_by_name:
            pred_by_name[name] = []
        pred_by_name[name].append(call)
    
    for gt_call in gt_calls:
        gt_name = gt_call.get("name", "")
        gt_args = gt_call.get("arguments", {})
        gt_keys = set(gt_args.keys())
        
        # Find matching predicted call
        if gt_name in pred_by_name and pred_by_name[gt_name]:
            pred_call = pred_by_name[gt_name].pop(0)  # Match first available
            pred_args = pred_call.get("arguments", {})
            pred_keys = set(pred_args.keys())
            
            # Parameter name matching (Jaccard)
            key_intersection = gt_keys & pred_keys
            key_union = gt_keys | pred_keys
            if key_union:
                r_param += len(key_intersection) / len(key_union)
            
            # Parameter value matching (exact match)
            for key in gt_keys:
                gt_val = gt_args.get(key)
                pred_val = pred_args.get(key)
                if gt_val == pred_val:
                    r_value += 1
        # else: No matching prediction for this GT call, r_param and r_value stay 0
    
    # 4. Total Match Score: r_match = r_name + r_param + r_value
    r_match = r_name + r_param + r_value
    
    # 5. S_max = 1 + |G| + Σ|keys(G_j)|
    total_gt_keys = sum(len(call.get("arguments", {}).keys()) for call in gt_calls)
    s_max = 1 + len(gt_calls) + total_gt_keys
    
    # Final: R_correct = 6 * (R_max / S_max) - 3
    r_correct = 6.0 * (r_match / s_max) - 3.0 if s_max > 0 else -3.0
    
    # Clamp to [-3, 3]
    return max(-3.0, min(3.0, r_correct))


def compute_rewards(sequences, tokenizer, references=None, reward_config=None, 
                    token_config=None, num_objectives=3,
                    uncertainty_scores=None,
                    temperature_rewards=None,
                    gt_tool_calls=None):
    """
    Computes rewards based on GDPO paper objectives with 4-level hierarchy.
    
    Reward Objectives:
    - Index 0: Format Reward [0, 1] - Easy level
    - Index 1: Length Penalty [0, 1] - Easy level
    - Index 2: Accuracy Reward [0, 1] - Hard level
    - Index 3: Uncertainty Reward [-1, 0] - Hardest level (optional)
    - Index 4: Tool Correctness [-3, 3] - Medium level (optional, when gt_tool_calls provided)
    - Index 5: Tool Format [0, 1] - Easy level (optional, when gt_tool_calls provided)
    - Index 6: Temperature Reward [-1, 1] - Optional
    
    4-Level Hierarchy (when uncertainty_scores and gt_tool_calls provided):
    - Level 1 (Hardest): Uncertainty - If fail, all below zeroed
    - Level 2 (Hard): Accuracy - If fail, Tool Correct and Easy zeroed
    - Level 3 (Medium): Tool Correctness - If fail, Easy zeroed
    - Level 4 (Easy): Format, Length, Tool Format
    
    Args:
        sequences: Generated token sequences
        tokenizer: Tokenizer for decoding
        references: Ground truth references for accuracy check
        reward_config: Configuration dict with keys:
            - condition_threshold (float): Threshold for accuracy to grant lower rewards (default 1.0)
            - target_length (int): Target length for length penalty calculation
            - uncertainty_threshold (float): Threshold for uncertainty (default 0.6)
            - tool_correctness_threshold (float): Threshold for Tool Correctness (default 0.0)
        token_config: Token configuration class (default: Ministral3TokenConfig)
        num_objectives: Number of reward objectives (default 3)
        uncertainty_scores: Optional (Batch*Group,) tensor of soft-scaled uncertainty scores
        temperature_rewards: Optional (Batch*Group,) tensor of temperature labels (+1 for low, -1 for high)
        gt_tool_calls: Optional list of ground truth tool call strings in [TOOL_CALLS]... format
    
    Returns: (Batch * Group, num_objectives)
    """
    # Auto-adjust num_objectives based on provided scores
    # Base: 3 (Format, Length, Accuracy)
    # +1 for Uncertainty (index 3)
    # +2 for Tool rewards (index 4: Tool Correctness, index 5: Tool Format)
    # +1 for Temperature (last index)
    if uncertainty_scores is not None:
        num_objectives = max(num_objectives, 4)
    if gt_tool_calls is not None:
        # Tool Correctness at index 4, Tool Format at index 5
        tool_correct_idx = 4 if uncertainty_scores is not None else 3
        num_objectives = max(num_objectives, tool_correct_idx + 2)  # +2 for both tool rewards
    if temperature_rewards is not None:
        # Temperature goes at the end
        temp_obj_idx = num_objectives
        num_objectives = temp_obj_idx + 1
    
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
    condition_threshold = reward_config.get("condition_threshold", 1.0)
    target_len = reward_config.get("target_length", 1024)
    tool_correctness_threshold = reward_config.get("tool_correctness_threshold", 0.0)
    
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
        acc_score = 0.0
        if references and i < len(references):
            ref_text = references[i]
            # Simplified: Check if reference is inside generation.
            if ref_text in text:
                acc_score = 1.0
        
        # 4. Tool Rewards (when gt_tool_calls provided)
        tool_correct_score = 0.0
        tool_format_score = 0.0
        
        if gt_tool_calls is not None and i < len(gt_tool_calls):
            # Parse predicted tool calls from generated text
            pred_calls = parse_tool_calls_for_reward(text)
            
            # Parse ground truth tool calls
            gt_text = gt_tool_calls[i]
            gt_calls = parse_tool_calls_for_reward(gt_text) if gt_text else []
            
            # Compute Tool Correctness (Medium level)
            tool_correct_score = compute_tool_correctness_reward(pred_calls, gt_calls)
            
            # Compute Tool Format (Easy level)
            tool_format_score = compute_tool_format_reward(text, token_config)
        
        # 5. Hierarchical Conditional Rewards (4-Level Cascade)
        # Level 0: All pass, Level 1: Uncertainty fail, Level 2: Accuracy fail, Level 3: Tool Correct fail
        uncertainty_reward = 0.0
        failed_level = 0  # 0 = all pass
        
        # Determine failed level
        if uncertainty_scores is not None and i < len(uncertainty_scores):
            u = uncertainty_scores[i]
            if isinstance(u, torch.Tensor):
                u = u.item()
            uncertainty_threshold = reward_config.get("uncertainty_threshold", 0.6)
            
            if u >= uncertainty_threshold:
                failed_level = 1  # Uncertainty fail
                uncertainty_reward = -1.0
            else:
                uncertainty_reward = -u
                if acc_score < condition_threshold:
                    failed_level = 2  # Accuracy fail
                elif gt_tool_calls is not None and tool_correct_score < tool_correctness_threshold:
                    failed_level = 3  # Tool Correct fail
        else:
            # No uncertainty - check accuracy and tool correct
            if acc_score < condition_threshold:
                failed_level = 2
            elif gt_tool_calls is not None and tool_correct_score < tool_correctness_threshold:
                failed_level = 3
        
        # Apply hierarchical zeroing based on failed level
        if failed_level >= 1:  # Uncertainty fail → zero all below
            acc_score = 0.0
            tool_correct_score = 0.0
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        elif failed_level >= 2:  # Accuracy fail → zero Tool Correct and Easy
            tool_correct_score = 0.0
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        elif failed_level >= 3:  # Tool Correct fail → zero Easy only
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        
        # Assign rewards to tensor
        rewards[i, 0] = format_score
        rewards[i, 1] = length_score
        rewards[i, 2] = acc_score
        
        # 6. Uncertainty Reward assignment (index 3)
        if uncertainty_scores is not None:
            rewards[i, 3] = uncertainty_reward
        
        # 7. Tool Rewards assignment
        if gt_tool_calls is not None:
            tool_correct_idx = 4 if uncertainty_scores is not None else 3
            rewards[i, tool_correct_idx] = tool_correct_score
            rewards[i, tool_correct_idx + 1] = tool_format_score
        
        # 8. Temperature Contrastive Reward (at the end)
        if temperature_rewards is not None:
            # Calculate temp index based on what's enabled
            temp_obj_idx = 3  # Base index after Format, Length, Accuracy
            if uncertainty_scores is not None:
                temp_obj_idx += 1  # +1 for Uncertainty
            if gt_tool_calls is not None:
                temp_obj_idx += 2  # +2 for Tool Correctness and Tool Format
            
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
    Standard GDPO with 3+ objectives (Format, Length, Accuracy).
    Optionally supports:
    - Tool rewards (Tool Correctness, Tool Format) - adds 2 objectives
    - Temperature contrastive - adds 1 objective
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
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels', 'gt_tool_calls'
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
        
        # 2. Prepare references and GT tool calls
        expanded_refs = self.prepare_references(inputs, effective_G)
        expanded_gt_tools = self.prepare_gt_tool_calls(inputs, effective_G)
        
        # 3. Determine num_objectives
        # Base: 3 (Format, Length, Accuracy)
        n = 3
        has_tool_reward = expanded_gt_tools is not None
        if has_tool_reward:
            n += 2  # Tool Correctness + Tool Format
        if temp_rewards is not None:
            n += 1  # Temperature
        
        # 4. Compute rewards
        reward_config = self.build_reward_config()
        
        rewards_flat = compute_rewards(
            sequences, self.tokenizer,
            references=expanded_refs,
            reward_config=reward_config,
            token_config=Ministral3TokenConfig,
            num_objectives=n,
            temperature_rewards=temp_rewards,
            gt_tool_calls=expanded_gt_tools
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 5. Compute advantages
        weights = self.get_reward_weights(n, has_uncertainty=False, has_tool_reward=has_tool_reward)
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
        
        # Add tool reward logging if enabled
        next_idx = 3
        if has_tool_reward:
            components["reward_tool_correctness"] = reward_means[next_idx].item()
            components["reward_tool_format"] = reward_means[next_idx + 1].item()
            next_idx += 2
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[next_idx].item()
        
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
    GDPO with Uncertainty Reward (4+ objectives: Format, Length, Accuracy, Uncertainty).
    Optionally supports:
    - Tool rewards (Tool Correctness, Tool Format) - adds 2 objectives
    - Temperature contrastive - adds 1 objective
    
    4-Level Hierarchy when tool rewards enabled:
    - Level 1 (Hardest): Uncertainty
    - Level 2 (Hard): Accuracy
    - Level 3 (Medium): Tool Correctness
    - Level 4 (Easy): Format, Length, Tool Format
    """
    
    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        self.uncertainty_threshold = self.gdpo_config.get("uncertainty_threshold", 0.6)
        # Default False = reasoning only, True = full sequence
        self.uncertainty_full_sequence = self.gdpo_config.get("uncertainty_full_sequence", False)
    
    def get_reasoning_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Create mask for tokens inside [THINK]...[/THINK].
        
        Args:
            sequences: Token IDs (batch_size, seq_len)
        
        Returns:
            reasoning_mask: Float tensor (batch_size, seq_len) with 1s for reasoning tokens
        """
        batch_size, seq_len = sequences.shape
        mask = torch.zeros_like(sequences, dtype=torch.float32)
        
        # Get token IDs for [THINK] and [/THINK]
        think_start_id = self.tokenizer.convert_tokens_to_ids("[THINK]")
        think_end_id = self.tokenizer.convert_tokens_to_ids("[/THINK]")
        
        for i in range(batch_size):
            in_think = False
            for j in range(seq_len):
                token_id = sequences[i, j].item()
                
                if token_id == think_start_id:
                    in_think = True
                    mask[i, j] = 1.0  # Include [THINK] token
                elif token_id == think_end_id:
                    mask[i, j] = 1.0  # Include [/THINK] token
                    in_think = False
                elif in_think:
                    mask[i, j] = 1.0
        
        return mask.to(sequences.device)
    
    def compute_uncertainty(
        self, 
        shift_logits: torch.Tensor, 
        valid_mask: torch.Tensor,
        reasoning_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty scores from logits std.
        If reasoning_mask provided, only compute on [THINK]...[/THINK] section.
        
        Args:
            shift_logits: Shifted logits
            valid_mask: Valid token mask
            reasoning_mask: Optional mask for reasoning tokens (1 inside [THINK]...[/THINK])
        
        Returns:
            sigma_per_sample: Raw sigma per sample
            uncertainty_scores: Soft-scaled uncertainty scores
        """
        eps = 1e-8
        sigma_per_token = shift_logits.std(dim=-1)
        
        # Combine masks: valid tokens AND reasoning tokens (if provided)
        if reasoning_mask is not None:
            # Shift reasoning_mask to match shift_logits (remove first token)
            shifted_reasoning_mask = reasoning_mask[:, 1:]
            combined_mask = valid_mask * shifted_reasoning_mask
        else:
            combined_mask = valid_mask
        
        sigma_per_sample = (sigma_per_token * combined_mask).sum(dim=1) / (combined_mask.sum(dim=1) + eps)
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
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels', 'gt_tool_calls'
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
        
        # 2. Prepare references and GT tool calls
        expanded_refs = self.prepare_references(inputs, effective_G)
        expanded_gt_tools = self.prepare_gt_tool_calls(inputs, effective_G)
        
        # 3. Forward pass FIRST (to compute uncertainty)
        input_len = input_ids.shape[1]
        outputs, logits, shift_logits, shift_labels, token_log_probs, valid_mask = \
            self.compute_log_probs(model, sequences, input_len)
        
        # 4. Compute uncertainty (reasoning-only by default, full sequence if configured)
        if self.uncertainty_full_sequence:
            # Full sequence uncertainty
            sigma_per_sample, uncertainty_scores = self.compute_uncertainty(shift_logits, valid_mask)
        else:
            # Reasoning-only uncertainty (default)
            reasoning_mask = self.get_reasoning_mask(sequences)
            sigma_per_sample, uncertainty_scores = self.compute_uncertainty(
                shift_logits, valid_mask, reasoning_mask
            )
        
        # 5. Determine num_objectives
        # Base: 4 (Format, Length, Accuracy, Uncertainty)
        n = 4
        has_tool_reward = expanded_gt_tools is not None
        if has_tool_reward:
            n += 2  # Tool Correctness + Tool Format
        if temp_rewards is not None:
            n += 1  # Temperature
        
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
            temperature_rewards=temp_rewards,
            gt_tool_calls=expanded_gt_tools
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 7. Compute advantages
        weights = self.get_reward_weights(n, has_uncertainty=True, has_tool_reward=has_tool_reward)
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
        
        # Add tool reward logging if enabled
        next_idx = 4  # After Uncertainty
        if has_tool_reward:
            components["reward_tool_correctness"] = reward_means[next_idx].item()
            components["reward_tool_format"] = reward_means[next_idx + 1].item()
            next_idx += 2
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[next_idx].item()
        
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
