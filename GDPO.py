import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Optional, Tuple, Dict, Any, List

from training_logger import LossResult
from architectures.ministral_3_3b_instruct import Ministral3TokenConfig

# Import for Monte Carlo uncertainty computation
from heteroscedastic_utils import (
    compute_heteroscedastic_log_probs,
    compute_learned_heteroscedastic_log_probs 
)

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
        self.accuracy_threshold = self.gdpo_config.get("accuracy_threshold", 1.0)
        self.target_length = self.gdpo_config.get("target_length", 1024)
        
        # Tool Reward config (enabled by default per GDPO paper)
        self.enable_tool_reward = self.gdpo_config.get("enable_tool_reward", True)
        self.tool_correctness_threshold = self.gdpo_config.get("tool_correctness_threshold", 1.5)
        
        # Reasoning Judge config
        self.enable_reasoning_judge = self.gdpo_config.get("enable_reasoning_judge", False)
        self.reasoning_judge = None
        self.reasoning_quality_threshold = 0.5
        
        if self.enable_reasoning_judge:
            from utils.reasoning_judge import ReasoningJudge
            self.reasoning_judge = ReasoningJudge()
        
        # Memory Optimization config
        self.sequential = self.gdpo_config.get("sequential", False)
    
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
                
                # Pad to same length before concatenation (sequences may have different lengths)
                max_len = max(low_sequences.shape[1], high_sequences.shape[1])
                pad_id = self.tokenizer.pad_token_id
                if low_sequences.shape[1] < max_len:
                    low_sequences = F.pad(low_sequences, (0, max_len - low_sequences.shape[1]), value=pad_id)
                if high_sequences.shape[1] < max_len:
                    high_sequences = F.pad(high_sequences, (0, max_len - high_sequences.shape[1]), value=pad_id)
                
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
    
    def generate_samples_sequential(
        self, 
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Generate samples one at a time for memory efficiency, storing on CPU.
        
        Args:
            model: The language model
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
        
        Returns:
            sequences_list: List of sequences on CPU (each B, seq_len)
            temperature_rewards: Temperature labels or None (B * effective_G,)
            effective_G: Actual group size (G or 2G)
        """
        B = input_ids.shape[0]
        all_sequences = []
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,  # Generate one at a time
            "use_cache": False,
        }
        
        with torch.no_grad():
            if self.use_temp_contrastive:
                # Low temp generation (chosen/positive) - one at a time
                gen_kwargs["temperature"] = self.low_temp
                for _ in range(self.G):
                    seq = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                    )
                    all_sequences.append(seq.cpu())
                    del seq
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # High temp generation (rejected/negative) - one at a time
                gen_kwargs["temperature"] = self.high_temp
                for _ in range(self.G):
                    seq = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                    )
                    all_sequences.append(seq.cpu())
                    del seq
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                temperature_rewards = torch.cat([
                    torch.ones(B * self.G),      # Low temp → +1
                    -torch.ones(B * self.G)      # High temp → -1
                ])
                effective_G = 2 * self.G
            else:
                gen_kwargs["temperature"] = self.default_temp
                for _ in range(self.G):
                    seq = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                    )
                    all_sequences.append(seq.cpu())
                    del seq
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                temperature_rewards = None
                effective_G = self.G
        
        return all_sequences, temperature_rewards, effective_G
    
    def pad_sequences_list(
        self,
        sequences_list: List[torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        """
        Pad and concatenate a list of sequences to uniform length.
        
        Args:
            sequences_list: List of sequence tensors (each B, varying_seq_len)
            device: Target device
        
        Returns:
            padded_sequences: Padded tensor (B * G, max_seq_len)
        """
        max_len = max(seq.shape[1] for seq in sequences_list)
        padded = []
        
        for seq in sequences_list:
            if seq.shape[1] < max_len:
                padding = torch.full(
                    (seq.shape[0], max_len - seq.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=seq.dtype
                )
                seq = torch.cat([seq, padding], dim=1)
            padded.append(seq)
        
        return torch.cat(padded, dim=0).to(device)
    
    def compute_single_log_probs(
        self, 
        model: torch.nn.Module, 
        seq: torch.Tensor, 
        input_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for a single sequence batch.
        Immediately discards logits to save memory.
        
        Args:
            model: The language model
            seq: Single sequence batch (B, seq_len)
            input_len: Length of input (to mask)
        
        Returns:
            token_log_probs: Per-token log probabilities (B, seq_len-1)
            valid_mask: Mask for valid (non-padding) tokens (B, seq_len-1)
        """
        train_attention_mask = (seq != self.tokenizer.pad_token_id).long()
        
        labels = seq.clone()
        labels[:, :input_len] = -100
        
        outputs = model(input_ids=seq, attention_mask=train_attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != -100).float()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_log_probs = -token_nll.view(seq.shape[0], -1)
        
        # Immediately delete large tensors
        del logits, shift_logits, outputs
        
        return token_log_probs, valid_mask, shift_labels
    
    def compute_single_kl(
        self,
        ref_model: torch.nn.Module,
        seq: torch.Tensor,
        token_log_probs: torch.Tensor,
        shift_labels: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence for a single sequence batch.
        
        Args:
            ref_model: Reference model
            seq: Single sequence batch (B, seq_len)
            token_log_probs: Policy log probs (B, seq_len-1)
            shift_labels: Shifted labels (B, seq_len-1)
            valid_mask: Valid token mask (B, seq_len-1)
        
        Returns:
            kl_per_sample: KL divergence per sample (B,)
        """
        eps = 1e-8
        train_attention_mask = (seq != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=seq, attention_mask=train_attention_mask)
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[..., :-1, :].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            ref_token_nll = loss_fct(ref_shift_logits.view(-1, ref_shift_logits.size(-1)), shift_labels.view(-1))
            ref_token_log_probs = -ref_token_nll.view(seq.shape[0], -1)
            
            kl = token_log_probs - ref_token_log_probs
            kl_per_sample = (kl * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + eps)
            
            del ref_logits, ref_shift_logits, ref_outputs
        
        return kl_per_sample
    
    def prepare_references(
        self,
        inputs: Dict[str, torch.Tensor],
        effective_G: int
    ) -> Optional[List[str]]:
        """
        Prepare expanded references for reward calculation.

        Priority:
        1. Ref model generation (if self.ref_model is not None) — greedy decode
        2. Labels-based decode (기존 방식, SFT 데이터용)

        Args:
            inputs: Input dict containing 'input_ids', 'attention_mask', optional 'labels'
            effective_G: Effective group size (G or 2G)

        Returns:
            expanded_refs: List of reference strings expanded to match batch size, or None
        """
        # Priority 1: Ref model 생성 (GDPO post-training용)
        if self.ref_model is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get(
                "attention_mask",
                (input_ids != self.tokenizer.pad_token_id).long()
            )

            with torch.no_grad():
                ref_sequences = self.ref_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy for deterministic reference
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode generated part only (prompt 부분 제외)
            prompt_len = input_ids.shape[1]
            ref_new_tokens = ref_sequences[:, prompt_len:]
            references = self.tokenizer.batch_decode(
                ref_new_tokens, skip_special_tokens=False
            )

            if self.debug:
                print(f"[DEBUG] Ref model generated {len(references)} references")
                if references:
                    print(f"[DEBUG] Ref[0] (first 100 chars): {references[0][:100]}")

            expanded_refs = []
            for ref in references:
                expanded_refs.extend([ref] * effective_G)
            return expanded_refs

        # Priority 2: Labels 기반 (기존 SFT 데이터 방식)
        if "labels" in inputs:
            valid_labels = inputs["labels"].clone()
            valid_labels[valid_labels == -100] = self.tokenizer.pad_token_id
            references = self.tokenizer.batch_decode(valid_labels, skip_special_tokens=True)

            expanded_refs = []
            for ref in references:
                expanded_refs.extend([ref] * effective_G)
            return expanded_refs

        return None
    
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
                           has_tool_reward: bool = False,
                           has_reasoning_judge: bool = False) -> List[float]:
        """
        Get reward weights as a list.
        
        Objective indices when has_tool_reward=True:
        - 0: Format (Easy)
        - 1: Length (Easy)
        - 2: Tool Format (Easy)
        - 3: Accuracy (Hard)
        - 4: Uncertainty (Medium) - if enabled
        - next: Reasoning Quality (Medium) - if enabled
        - next: Tool Correctness (Hardest)
        - Last: Temperature - if enabled
        
        Objective indices when has_tool_reward=False:
        - 0: Format (Easy)
        - 1: Length (Easy)
        - 2: Accuracy (Hard)
        - 3: Uncertainty (Medium) - if enabled
        - next: Reasoning Quality (Medium) - if enabled
        - Last: Temperature - if enabled
        
        Args:
            num_objectives: Number of objectives
            has_uncertainty: Whether uncertainty reward is enabled
            has_tool_reward: Whether tool rewards are enabled
            has_reasoning_judge: Whether reasoning quality reward is enabled
        
        Returns:
            List of weights for each objective
        """
        weights_config = self.gdpo_config.get("reward_weights", {})
        
        if has_tool_reward:
            weights = [
                weights_config.get("format", 1.0),
                weights_config.get("length", 1.0),
                weights_config.get("tool_format", 1.0),
                weights_config.get("accuracy", 1.0),
            ]
            
            if has_uncertainty:
                weights.append(weights_config.get("uncertainty", 1.0))
            
            if has_reasoning_judge:
                weights.append(weights_config.get("reasoning_quality", 1.0))
            
            weights.append(weights_config.get("tool_correctness", 1.0))
        else:
            weights = [
                weights_config.get("format", 1.0),
                weights_config.get("length", 1.0),
                weights_config.get("accuracy", 1.0),
            ]
            
            if has_uncertainty:
                weights.append(weights_config.get("uncertainty", 1.0))
            
            if has_reasoning_judge:
                weights.append(weights_config.get("reasoning_quality", 1.0))
        
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
    
    def judge_rollout_reasoning(
        self,
        sequences: torch.Tensor,
        input_ids: torch.Tensor,
        references: Optional[List[str]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Judge reasoning quality for generated rollout sequences.
        
        Args:
            sequences: Generated sequences (B*G, seq_len)
            input_ids: Original input IDs (B, seq_len) for question extraction
            references: Optional reference answers
        
        Returns:
            Tensor of scores (B*G,) or None if judge is disabled
        """
        if self.reasoning_judge is None:
            return None
        
        token_config = self.gdpo_config.get("token_config", Ministral3TokenConfig)
        think_start = token_config.THINK_START or "[THINK]"
        think_end = token_config.THINK_END or "[/THINK]"
        
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)

        B = input_ids.shape[0]
        questions_raw = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        effective_G = len(texts) // B
        questions = []
        for q in questions_raw:
            questions.extend([q] * effective_G)
        
        refs = references if references else [None] * len(texts)
        
        scores = self.reasoning_judge.judge_batch(
            questions, texts, refs, think_start, think_end
        )
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def cleanup_judge(self):
        """Release judge resources. Call at training end."""
        if self.reasoning_judge is not None:
            self.reasoning_judge.cleanup()
    
    def build_reward_config(self) -> Dict[str, Any]:
        """Build reward configuration dict."""
        return {
            "use_conditioned_rewards": self.use_conditioned_rewards,
            "accuracy_threshold": self.accuracy_threshold,
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


def compute_tool_format_reward(text: str) -> float:
    """
    Tool Format Reward (Binary 0/1) - GDPO Paper.
    
    Checks structural validity of tool calls:
    - [TOOL_CALLS] tag presence
    - Valid JSON parsing
    - Required fields: "name", "arguments"
    
    Args:
        text: LLM output text to check for valid tool call format
        
    Returns:
        1.0 if format is valid, 0.0 otherwise
    """
    if "[TOOL_CALLS]" not in text:
        return 0.0
    
    calls = parse_tool_calls_for_reward(text)
    if not calls:
        return 0.0
    
    for call in calls:
        if "name" not in call or "arguments" not in call:
            return 0.0
    
    return 1.0


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
    
    This is the "Hardest" level reward in the 4-level hierarchy.
    
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
                    reasoning_quality_scores=None,
                    temperature_rewards=None,
                    gt_tool_calls=None,
                    enable_tool_reward=True):
    """
    Computes rewards based on GDPO paper objectives with 4-level hierarchy.
    
    Reward Objectives (when enable_tool_reward=True):
    - Index 0: Format Reward [0, 1] - Easy level
    - Index 1: Length Penalty [0, 1] - Easy level
    - Index 2: Tool Format [0, 1] - Easy level (structural validity)
    - Index 3: Accuracy Reward [0, 1] - Hard level
    - Index 4: Uncertainty Reward [-1, 0] - Medium level (optional)
    - next: Reasoning Quality [0, 1] - Medium level (optional)
    - next: Tool Correctness [-3, 3] - Hardest level
    - Last: Temperature Reward [-1, 1] - Optional
    
    4-Level Hierarchy:
    - Level 1 (Hardest): Tool Correctness - If fail, zero Acc, Uncertainty, Reasoning, Easy
    - Level 2 (Hard): Accuracy - If fail, zero Uncertainty, Reasoning, and Easy
    - Level 3 (Medium): Uncertainty / Reasoning Quality - If either fail, zero Easy only
    - Level 4 (Easy): Format, Length, Tool Format
    
    Args:
        sequences: Generated token sequences
        tokenizer: Tokenizer for decoding
        references: Ground truth references for accuracy check
        reward_config: Configuration dict with keys:
            - accuracy_threshold (float): Threshold for accuracy to grant lower rewards (default 1.0)
            - target_length (int): Target length for length penalty calculation
            - uncertainty_threshold (float): Threshold for uncertainty (default 0.6)
            - tool_correctness_threshold (float): Threshold for Tool Correctness (default 1.5, ~75% match)
            - reasoning_quality_threshold (float): Threshold for reasoning quality (default 0.5)
        token_config: Token configuration class (default: Ministral3TokenConfig)
        num_objectives: Number of reward objectives (default 3)
        uncertainty_scores: Optional (Batch*Group,) tensor of uncertainty scores (1 - prob_correct)
        reasoning_quality_scores: Optional (Batch*Group,) list/tensor of reasoning quality scores [0, 1]
        temperature_rewards: Optional (Batch*Group,) tensor of temperature labels (+1 for low, -1 for high)
        gt_tool_calls: Optional list of ground truth tool call strings in [TOOL_CALLS]... format
        enable_tool_reward: Whether tool rewards (Format + Correctness) are enabled (default True)
    
    Returns: (Batch * Group, num_objectives)
    """
    # Auto-adjust num_objectives based on enabled features
    if enable_tool_reward:
        num_objectives = 5  # Format, Length, Tool Format, Accuracy, Tool Correctness
        if uncertainty_scores is not None:
            num_objectives += 1
        if reasoning_quality_scores is not None:
            num_objectives += 1
    else:
        num_objectives = 3  # Format, Length, Accuracy
        if uncertainty_scores is not None:
            num_objectives += 1
        if reasoning_quality_scores is not None:
            num_objectives += 1
    
    if temperature_rewards is not None:
        num_objectives += 1
    
    batch_size = sequences.shape[0]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=False)

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
    accuracy_threshold = reward_config.get("accuracy_threshold", 1.0)
    target_len = reward_config.get("target_length", 1024)
    tool_correctness_threshold = reward_config.get("tool_correctness_threshold", 1.5)
    
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
        
        # 3. Tool Format Reward (Easy level - structural validity)
        tool_format_score = 0.0
        if enable_tool_reward:
            tool_format_score = compute_tool_format_reward(text)
        
        # 4. Accuracy Reward (Hard reward - computed first as it may condition others)
        # If references (ground truth texts) are provided, we check for containment or exact match.
        acc_score = 0.0
        if references and i < len(references):
            ref_text = references[i]
            # Simplified: Check if reference is inside generation.
            if ref_text in text:
                acc_score = 1.0
        
        # 5. Tool Correctness Reward (Hardest level - semantic correctness)
        tool_correct_score = 0.0
        if enable_tool_reward:
            if gt_tool_calls is not None and i < len(gt_tool_calls):
                # Parse predicted tool calls from generated text
                pred_calls = parse_tool_calls_for_reward(text)
                
                # Parse ground truth tool calls
                gt_text = gt_tool_calls[i]
                gt_calls = parse_tool_calls_for_reward(gt_text) if gt_text else []
                
                # Compute Tool Correctness
                tool_correct_score = compute_tool_correctness_reward(pred_calls, gt_calls)
        
        # 6. Hierarchical Conditional Rewards (4-Level Cascade)
        # Level 1 (Hardest): Tool Correctness - fail → zero Acc, Medium, Easy
        # Level 2 (Hard): Accuracy - fail → zero Medium and Easy
        # Level 3 (Medium): Uncertainty / Reasoning Quality - either fail → zero Easy
        # Level 4 (Easy): Format, Length, Tool Format
        uncertainty_reward = 0.0
        reasoning_quality_reward = 0.0
        failed_level = 0  # 0 = all pass
        
        # Compute uncertainty reward value
        if uncertainty_scores is not None and i < len(uncertainty_scores):
            u = uncertainty_scores[i]
            if isinstance(u, torch.Tensor):
                u = u.item()
            uncertainty_threshold = reward_config.get("uncertainty_threshold", 0.6)
            
            if u >= uncertainty_threshold:
                uncertainty_reward = -1.0
            else:
                uncertainty_reward = -u
        
        # Compute reasoning quality reward value
        rq_threshold = reward_config.get("reasoning_quality_threshold", 0.5)
        if reasoning_quality_scores is not None and i < len(reasoning_quality_scores):
            rq = reasoning_quality_scores[i]
            if isinstance(rq, torch.Tensor):
                rq = rq.item()
            reasoning_quality_reward = rq
        
        # Determine failed level (check in order: Tool > Acc > Uncertainty/ReasoningQuality)
        if enable_tool_reward and tool_correct_score < tool_correctness_threshold:
            failed_level = 1
        elif acc_score < accuracy_threshold:
            failed_level = 2
        elif (uncertainty_scores is not None and uncertainty_reward == -1.0) or \
             (reasoning_quality_scores is not None and reasoning_quality_reward < rq_threshold):
            failed_level = 3
        
        # Apply hierarchical zeroing
        if failed_level == 1:
            # Tool Correctness 실패: acc, medium, easy 전부 zero
            acc_score = 0.0
            uncertainty_reward = 0.0
            reasoning_quality_reward = 0.0
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        elif failed_level == 2:
            # Accuracy 실패: medium, easy zero
            uncertainty_reward = 0.0
            reasoning_quality_reward = 0.0
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        elif failed_level == 3:
            # Medium 실패: easy만 zero
            format_score = 0.0
            length_score = 0.0
            tool_format_score = 0.0
        
        # Assign rewards using dynamic index tracking
        rewards[i, 0] = format_score
        rewards[i, 1] = length_score
        
        if enable_tool_reward:
            rewards[i, 2] = tool_format_score
            rewards[i, 3] = acc_score
            idx = 4
            
            if uncertainty_scores is not None:
                rewards[i, idx] = uncertainty_reward
                idx += 1
            
            if reasoning_quality_scores is not None:
                rewards[i, idx] = reasoning_quality_reward
                idx += 1
            
            rewards[i, idx] = tool_correct_score
            idx += 1
            
            if temperature_rewards is not None and i < len(temperature_rewards):
                t = temperature_rewards[i]
                if isinstance(t, torch.Tensor):
                    t = t.item()
                rewards[i, idx] = t
        else:
            rewards[i, 2] = acc_score
            idx = 3
            
            if uncertainty_scores is not None:
                rewards[i, idx] = uncertainty_reward
                idx += 1
            
            if reasoning_quality_scores is not None:
                rewards[i, idx] = reasoning_quality_reward
                idx += 1
            
            if temperature_rewards is not None and i < len(temperature_rewards):
                t = temperature_rewards[i]
                if isinstance(t, torch.Tensor):
                    t = t.item()
                rewards[i, idx] = t
        
    return rewards


# =============================================================================
# GDPO Loss Classes
# =============================================================================

class GDPOLoss(GDPOBase):
    """
    Standard GDPO with 3+ objectives (Format, Length, Accuracy).
    Optionally supports:
    - Tool Correctness reward - adds 1 objective
    - Temperature contrastive - adds 1 objective
    - Sequential mode for memory efficiency
    """
    
    def __call__(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """
        Compute GDPO loss. Automatically selects parallel or sequential mode.
        
        Args:
            model: Policy model
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels', 'gt_tool_calls'
            ref_model: Reference model for KL penalty (optional)
        
        Returns:
            LossResult with total_loss, components, and outputs
        """
        if self.sequential:
            return self._call_sequential(model, inputs, ref_model)
        else:
            return self._call_parallel(model, inputs, ref_model)
    
    def _call_parallel(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """Parallel mode: Generate all samples at once (faster, more memory)."""
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
        
        # 3. Reasoning judge
        has_judge = self.enable_reasoning_judge
        rq_scores = self.judge_rollout_reasoning(sequences, input_ids, expanded_refs) if has_judge else None
        
        # 4. Determine num_objectives
        has_tool_reward = self.enable_tool_reward
        if has_tool_reward:
            n = 5
        else:
            n = 3
        if has_judge:
            n += 1
        if temp_rewards is not None:
            n += 1
        
        # 5. Compute rewards
        reward_config = self.build_reward_config()
        
        rewards_flat = compute_rewards(
            sequences, self.tokenizer,
            references=expanded_refs,
            reward_config=reward_config,
            token_config=Ministral3TokenConfig,
            num_objectives=n,
            reasoning_quality_scores=rq_scores,
            temperature_rewards=temp_rewards,
            gt_tool_calls=expanded_gt_tools,
            enable_tool_reward=has_tool_reward
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 6. Compute advantages
        weights = self.get_reward_weights(n, has_uncertainty=False, has_tool_reward=has_tool_reward,
                                          has_reasoning_judge=has_judge)
        final_advantages_flat = self.compute_advantages(rewards, weights, model.device)
        
        # 7. Forward pass & log probs
        input_len = input_ids.shape[1]
        outputs, logits, shift_logits, shift_labels, token_log_probs, valid_mask = \
            self.compute_log_probs(model, sequences, input_len)
        
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
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        
        if has_tool_reward:
            components["reward_tool_format"] = reward_means[2].item()
            components["reward_accuracy"] = reward_means[3].item()
            idx = 4
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
            components["reward_tool_correctness"] = reward_means[idx].item()
            idx += 1
        else:
            components["reward_accuracy"] = reward_means[2].item()
            idx = 3
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[idx].item()
        
        # Memory cleanup
        del sequences, rewards_flat, rewards
        del logits, shift_logits, shift_labels, token_log_probs
        if ref_model and kl is not None:
            del kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return LossResult(total_loss=total_loss, components=components, outputs=outputs)
    
    def _call_sequential(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """Sequential mode: Generate samples one at a time (slower, less memory)."""
        if ref_model is None:
            ref_model = self.ref_model
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Generate samples sequentially (stored on CPU)
        sequences_list, temp_rewards, effective_G = self.generate_samples_sequential(
            model, input_ids, attention_mask
        )
        
        # 2. Prepare references and GT tool calls
        expanded_refs = self.prepare_references(inputs, effective_G)
        expanded_gt_tools = self.prepare_gt_tool_calls(inputs, effective_G)
        
        # 3. Reasoning judge (on padded sequences)
        has_judge = self.enable_reasoning_judge
        rq_scores = None
        if has_judge:
            sequences_for_judge = self.pad_sequences_list(sequences_list, device)
            rq_scores = self.judge_rollout_reasoning(sequences_for_judge, input_ids, expanded_refs)
            del sequences_for_judge
        
        # 4. Determine num_objectives
        has_tool_reward = self.enable_tool_reward
        if has_tool_reward:
            n = 5
        else:
            n = 3
        if has_judge:
            n += 1
        if temp_rewards is not None:
            n += 1
        
        # 5. Compute rewards sequentially
        reward_config = self.build_reward_config()
        all_rewards = []
        
        for i, seq_cpu in enumerate(sequences_list):
            seq_gpu = seq_cpu.to(device)
            batch_temp_rewards = None
            if temp_rewards is not None:
                start_idx = i * B
                end_idx = (i + 1) * B
                batch_temp_rewards = temp_rewards[start_idx:end_idx]
            
            batch_gt_tools = None
            if expanded_gt_tools is not None:
                start_idx = i * B
                end_idx = (i + 1) * B
                batch_gt_tools = expanded_gt_tools[start_idx:end_idx]
            
            batch_refs = None
            if expanded_refs is not None:
                start_idx = i * B
                end_idx = (i + 1) * B
                batch_refs = expanded_refs[start_idx:end_idx]
            
            batch_rq = None
            if rq_scores is not None:
                start_idx = i * B
                end_idx = (i + 1) * B
                batch_rq = rq_scores[start_idx:end_idx]
            
            reward = compute_rewards(
                seq_gpu, self.tokenizer,
                references=batch_refs,
                reward_config=reward_config,
                token_config=Ministral3TokenConfig,
                num_objectives=n,
                reasoning_quality_scores=batch_rq,
                temperature_rewards=batch_temp_rewards,
                gt_tool_calls=batch_gt_tools,
                enable_tool_reward=has_tool_reward
            )
            all_rewards.append(reward.cpu())
            del seq_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack rewards and compute advantages
        rewards_flat = torch.cat(all_rewards, dim=0).to(device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        weights = self.get_reward_weights(n, has_uncertainty=False, has_tool_reward=has_tool_reward,
                                          has_reasoning_judge=has_judge)
        final_advantages_flat = self.compute_advantages(rewards, weights, device)
        
        # 6. Forward pass sequentially
        input_len = input_ids.shape[1]
        total_pg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl = torch.tensor(0.0, device=device)
        
        sequences = self.pad_sequences_list(sequences_list, device)
        
        all_log_probs = []
        all_valid_masks = []
        all_shift_labels = []
        
        for i, seq_cpu in enumerate(sequences_list):
            seq_gpu = seq_cpu.to(device)
            
            max_len = sequences.shape[1]
            if seq_gpu.shape[1] < max_len:
                padding = torch.full(
                    (seq_gpu.shape[0], max_len - seq_gpu.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=seq_gpu.dtype,
                    device=device
                )
                seq_gpu = torch.cat([seq_gpu, padding], dim=1)
            
            token_log_probs, valid_mask, shift_labels = self.compute_single_log_probs(
                model, seq_gpu, input_len
            )
            all_log_probs.append(token_log_probs)
            all_valid_masks.append(valid_mask)
            all_shift_labels.append(shift_labels)
            
            if ref_model is not None:
                kl_per_sample = self.compute_single_kl(
                    ref_model, seq_gpu, token_log_probs, shift_labels, valid_mask
                )
                total_kl = total_kl + kl_per_sample.sum()
            
            del seq_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        token_log_probs = torch.cat(all_log_probs, dim=0)
        valid_mask = torch.cat(all_valid_masks, dim=0)
        
        seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
        pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
        
        eps = 1e-8
        total_samples = B * effective_G
        kl_penalty = self.kl_coef * total_kl / (total_samples + eps) if ref_model else 0.0
        
        total_loss = pg_loss + kl_penalty
        
        # Logging
        reward_means = rewards.mean(dim=(0, 1))
        kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
        
        components = {
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        
        if has_tool_reward:
            components["reward_tool_format"] = reward_means[2].item()
            components["reward_accuracy"] = reward_means[3].item()
            idx = 4
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
            components["reward_tool_correctness"] = reward_means[idx].item()
            idx += 1
        else:
            components["reward_accuracy"] = reward_means[2].item()
            idx = 3
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[idx].item()
        
        # Extract first generated sample for predict logging before cleanup
        # del right before return has no VRAM benefit (locals freed on return anyway)
        pred_ids = sequences_list[0][:1, input_len:].clone()   # (1, gen_len)
        pred_mask = (pred_ids != self.tokenizer.pad_token_id)  # (1, gen_len)

        # Memory cleanup
        del sequences_list, sequences, rewards_flat, rewards
        del all_log_probs, all_valid_masks, all_shift_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return LossResult(
            total_loss=total_loss, components=components, outputs=None,
            predicted_ids=pred_ids, response_mask=pred_mask,
        )


class HeteroscedasticGDPOLoss(GDPOBase):
    """
    GDPO with Uncertainty Reward (4+ objectives: Format, Length, Accuracy, Uncertainty).
    Optionally supports:
    - Tool Correctness reward - adds 1 objective
    - Temperature contrastive - adds 1 objective
    
    4-Level Hierarchy when tool rewards enabled:
    - Level 1 (Hardest): Tool Correctness - If fail, zero all below
    - Level 2 (Hard): Accuracy - If fail, zero Uncertainty and Easy
    - Level 3 (Medium): Uncertainty - If fail, zero Easy only
    - Level 4 (Easy): Format, Length
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
        
        # Get token_config from gdpo_config (fallback to default Ministral3TokenConfig)
        token_config = self.gdpo_config.get("token_config", Ministral3TokenConfig)
        
        # Get token IDs for THINK_START and THINK_END
        think_start_id = self.tokenizer.convert_tokens_to_ids(token_config.THINK_START)
        think_end_id = self.tokenizer.convert_tokens_to_ids(token_config.THINK_END)
        
        for i in range(batch_size):
            in_think = False
            for j in range(seq_len):
                token_id = sequences[i, j].item()
                
                if token_id == think_start_id:
                    in_think = True
                    mask[i, j] = 1.0  # Include THINK_START token
                elif token_id == think_end_id:
                    mask[i, j] = 1.0  # Include THINK_END token
                    in_think = False
                elif in_think:
                    mask[i, j] = 1.0
        
        return mask.to(sequences.device)
    
    def compute_uncertainty(
        self, 
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        valid_mask: torch.Tensor,
        reasoning_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty scores using Monte Carlo sampling with reference labels.
        
        Uses the same formula as heteroscedastic loss:
            p_correct = (1/T) * Σ_t softmax(logits + σ*ε_t)[label]
            uncertainty = 1 - p_correct
        
        Args:
            shift_logits: Shifted logits (batch, seq_len, vocab_size)
            shift_labels: Shifted labels (batch, seq_len) - ground truth tokens
            valid_mask: Valid token mask (batch, seq_len)
            reasoning_mask: Optional mask for reasoning tokens (1 inside [THINK]...[/THINK])
        
        Returns:
            prob_correct_per_sample: Average correct probability per sample
            uncertainty_scores: 1 - prob_correct (higher = more uncertain)
        """
        eps = 1e-8
        T = self.gdpo_config.get("heteroscedastic_T", 3)
        sequential = self.gdpo_config.get("heteroscedastic_sequential", False)
        
        # Use Monte Carlo to compute probability of correct tokens
        # All samples are detached except the last (but for reward we don't need gradients)
        log_probs, _ = compute_heteroscedastic_log_probs(
            shift_logits, shift_labels, T=T, sequential=sequential, return_sigma=True
        )
        
        # Convert log probs to probabilities
        prob_correct = torch.exp(log_probs.clamp(max=0))  # Clamp to avoid exp overflow
        
        # Combine masks: valid tokens AND reasoning tokens (if provided)
        if reasoning_mask is not None:
            # Shift reasoning_mask to match shift_logits (remove first token)
            shifted_reasoning_mask = reasoning_mask[:, 1:]
            combined_mask = valid_mask * shifted_reasoning_mask
        else:
            combined_mask = valid_mask
        
        # Average probability per sample (only over masked tokens)
        prob_correct_per_sample = (prob_correct * combined_mask).sum(dim=1) / (combined_mask.sum(dim=1) + eps)
        
        # Uncertainty = 1 - probability of correct prediction
        # Range: [0, 1] where 1 = completely uncertain, 0 = completely confident
        uncertainty_scores = 1 - prob_correct_per_sample
        
        return prob_correct_per_sample, uncertainty_scores
    
    def __call__(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """
        Compute Heteroscedastic GDPO loss. Automatically selects parallel or sequential mode.
        
        Args:
            model: Policy model
            inputs: Input dict with 'input_ids', 'attention_mask', optional 'labels', 'gt_tool_calls'
            ref_model: Reference model for KL penalty (optional)
        
        Returns:
            LossResult with total_loss, components, and outputs
        """
        if self.sequential:
            return self._call_sequential(model, inputs, ref_model)
        else:
            return self._call_parallel(model, inputs, ref_model)
    
    def _call_parallel(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """Parallel mode: Generate all samples at once (faster, more memory)."""
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
        
        if not hasattr(outputs, 'log_variance') or outputs.log_variance is None:
            raise ValueError("Model outputs must include 'log_variance' for heteroscedastic loss.")
        shift_log_var=outputs.log_variance[..., :-1, :].contiguous()  # Align with shift_logits
        
        variance = torch.exp(shift_log_var.squeeze(-1))  # 학습된 불확실성 수치화

        reasoning_mask = self.get_reasoning_mask(sequences)
        if not self.uncertainty_full_sequence:
            shifted_reasoning_mask = reasoning_mask[:, 1:]
            combined_mask= valid_mask * shifted_reasoning_mask
        else:
            combined_mask = valid_mask
        
        eps=1e-8 #평균 분산 계산 ㄱㄱㄱ
        seq_variance = (variance * combined_mask).sum(dim=1) / (combined_mask.sum(dim=1) + eps)

        has_judge = self.enable_reasoning_judge
        rq_scores = None

        if has_judge:
            rq_scores = torch.zeros(sequences.shape[0], dtype=torch.float32, device=model.device) 
            certain_mask = seq_variance <= self.uncertainty_threshold
            certain_indices = certain_mask.nonzero(as_tuple=True)[0]
            
            if len (certain_indices) > 0:
                certain_seqs = sequences[certain_indices]
                certain_input_ids = input_ids[certain_indices % B]
                certain_refs=None
                if expanded_refs:
                    certain_refs=[expanded_refs[i.item()] for i in certain_indices]

                partial_rq_scores=self.judge_rollout_reasoning(certain_seqs, certain_input_ids, certain_refs)
                rq_scores[certain_indices] = partial_rq_scores.to(model.device)

        T = self.gdpo_config.get("heteroscedastic_T", 3)
        prob_correct_log, _ = compute_learned_heteroscedastic_log_probs(
            shift_logits, shift_log_var, shift_labels, T=T, sequential=False)
        prob_correct = torch.exp(prob_correct_log.clamp(max=0))
        prob_correct_per_sample = (prob_correct * combined_mask).sum(dim=1) / (combined_mask.sum(dim=1) + eps)
        uncertainty_scores = 1 - prob_correct_per_sample

        # 7. Compute rewards
        reward_config = self.build_reward_config()
        reward_config["uncertainty_threshold"] = self.uncertainty_threshold
        
        rewards_flat = compute_rewards(
            sequences, self.tokenizer,
            references=expanded_refs,
            reward_config=reward_config,
            token_config=Ministral3TokenConfig,
            num_objectives=n,
            uncertainty_scores=uncertainty_scores.detach().cpu(),
            reasoning_quality_scores=rq_scores,
            temperature_rewards=temp_rewards,
            gt_tool_calls=expanded_gt_tools,
            enable_tool_reward=has_tool_reward
        ).to(model.device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        # 8. Compute advantages
        weights = self.get_reward_weights(n, has_uncertainty=True, has_tool_reward=has_tool_reward,
                                          has_reasoning_judge=has_judge)
        final_advantages_flat = self.compute_advantages(rewards, weights, model.device)
        
        # 9. KL penalty
        kl_penalty, kl = self.compute_kl_penalty(
            ref_model, sequences, token_log_probs, shift_labels, valid_mask
        )
        
        # 10. Policy gradient loss
        seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
        pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
        
        total_loss = pg_loss + kl_penalty
        
        # 11. Logging
        reward_means = rewards.mean(dim=(0, 1))
        kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
        
        components = {
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "uncertainty_mean": uncertainty_scores.mean().item(),
            "uncertainty_std": uncertainty_scores.std().item(),
            "prob_correct_mean": prob_correct.mean().item(),
            "prob_correct_std": prob_correct.std().item(),
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        
        if has_tool_reward:
            components["reward_tool_format"] = reward_means[2].item()
            components["reward_accuracy"] = reward_means[3].item()
            components["reward_uncertainty"] = reward_means[4].item()
            idx = 5
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
            components["reward_tool_correctness"] = reward_means[idx].item()
            idx += 1
        else:
            components["reward_accuracy"] = reward_means[2].item()
            components["reward_uncertainty"] = reward_means[3].item()
            idx = 4
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[idx].item()
        
        # Memory cleanup
        del sequences, rewards_flat, rewards
        del logits, shift_logits, shift_labels, token_log_probs
        del prob_correct, uncertainty_scores
        if ref_model and kl is not None:
            del kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return LossResult(total_loss=total_loss, components=components, outputs=outputs)
    
    def _call_sequential(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        ref_model: Optional[torch.nn.Module] = None
    ) -> LossResult:
        """Sequential mode: Generate samples one at a time (slower, less memory)."""
        if ref_model is None:
            ref_model = self.ref_model
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Generate samples sequentially (stored on CPU)
        sequences_list, temp_rewards, effective_G = self.generate_samples_sequential(
            model, input_ids, attention_mask
        )
        
        # 2. Prepare references and GT tool calls
        expanded_refs = self.prepare_references(inputs, effective_G)
        expanded_gt_tools = self.prepare_gt_tool_calls(inputs, effective_G)
        
        # 2.5. Compute max_len and create padded sequences for consistent tensor sizes
        max_len = max(seq.shape[1] for seq in sequences_list)
        padded_sequences_list = []
        for seq_cpu in sequences_list:
            if seq_cpu.shape[1] < max_len:
                padding = torch.full(
                    (seq_cpu.shape[0], max_len - seq_cpu.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=seq_cpu.dtype
                )
                padded_seq = torch.cat([seq_cpu, padding], dim=1)
            else:
                padded_seq = seq_cpu
            padded_sequences_list.append(padded_seq)
        
        # 3. Forward pass sequentially to compute uncertainty and log probs
        input_len = input_ids.shape[1]
        all_log_probs = []
        all_valid_masks = []
        all_shift_labels = []
        all_prob_correct = []
        all_uncertainty = []
        
        for i, seq_cpu in enumerate(padded_sequences_list):
            seq_gpu = seq_cpu.to(device)
            
            # Compute log probs and get shift_logits for uncertainty
            train_attention_mask = (seq_gpu != self.tokenizer.pad_token_id).long()
            labels = seq_gpu.clone()
            labels[:, :input_len] = -100
            
            outputs = model(input_ids=seq_gpu, attention_mask=train_attention_mask)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100).float()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_log_probs = -token_nll.view(seq_gpu.shape[0], -1)
            
            # Compute uncertainty for this batch
            if self.uncertainty_full_sequence:
                prob_correct, uncertainty = self.compute_uncertainty(
                    shift_logits, shift_labels, valid_mask
                )
            else:
                reasoning_mask = self.get_reasoning_mask(seq_gpu)
                prob_correct, uncertainty = self.compute_uncertainty(
                    shift_logits, shift_labels, valid_mask, reasoning_mask
                )
            
            all_log_probs.append(token_log_probs.cpu())
            all_valid_masks.append(valid_mask.cpu())
            all_shift_labels.append(shift_labels.cpu())
            all_prob_correct.append(prob_correct.cpu())
            all_uncertainty.append(uncertainty.cpu())
            
            del seq_gpu, logits, shift_logits, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack uncertainty scores
        uncertainty_scores = torch.cat(all_uncertainty, dim=0).to(device)
        prob_correct_all = torch.cat(all_prob_correct, dim=0).to(device)
        
        # 4. Reasoning judge
        has_judge = self.enable_reasoning_judge
        rq_scores = None
        if has_judge:
            sequences_for_judge = self.pad_sequences_list(padded_sequences_list, device)
            rq_scores = self.judge_rollout_reasoning(sequences_for_judge, input_ids, expanded_refs)
            del sequences_for_judge
        
        # 5. Determine num_objectives
        has_tool_reward = self.enable_tool_reward
        if has_tool_reward:
            n = 6
        else:
            n = 4
        if has_judge:
            n += 1
        if temp_rewards is not None:
            n += 1
        
        # 6. Compute rewards sequentially
        reward_config = self.build_reward_config()
        reward_config["uncertainty_threshold"] = self.uncertainty_threshold
        all_rewards = []
        
        for i, seq_cpu in enumerate(padded_sequences_list):
            seq_gpu = seq_cpu.to(device)
            
            start_idx = i * B
            end_idx = (i + 1) * B
            batch_uncertainty = uncertainty_scores[start_idx:end_idx].detach().cpu()
            
            batch_temp_rewards = None
            if temp_rewards is not None:
                batch_temp_rewards = temp_rewards[start_idx:end_idx]
            
            batch_gt_tools = None
            if expanded_gt_tools is not None:
                batch_gt_tools = expanded_gt_tools[start_idx:end_idx]
            
            batch_refs = None
            if expanded_refs is not None:
                batch_refs = expanded_refs[start_idx:end_idx]
            
            batch_rq = None
            if rq_scores is not None:
                batch_rq = rq_scores[start_idx:end_idx]
            
            reward = compute_rewards(
                seq_gpu, self.tokenizer,
                references=batch_refs,
                reward_config=reward_config,
                token_config=Ministral3TokenConfig,
                num_objectives=n,
                uncertainty_scores=batch_uncertainty,
                reasoning_quality_scores=batch_rq,
                temperature_rewards=batch_temp_rewards,
                gt_tool_calls=batch_gt_tools,
                enable_tool_reward=has_tool_reward
            )
            all_rewards.append(reward.cpu())
            del seq_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack rewards and compute advantages
        rewards_flat = torch.cat(all_rewards, dim=0).to(device)
        rewards = rewards_flat.view(B, effective_G, n)
        
        weights = self.get_reward_weights(n, has_uncertainty=True, has_tool_reward=has_tool_reward,
                                          has_reasoning_judge=has_judge)
        final_advantages_flat = self.compute_advantages(rewards, weights, device)
        
        # 7. Compute KL sequentially
        total_kl = torch.tensor(0.0, device=device)
        
        if ref_model is not None:
            for i, seq_cpu in enumerate(padded_sequences_list):
                seq_gpu = seq_cpu.to(device)
                token_log_probs = all_log_probs[i].to(device)
                shift_labels = all_shift_labels[i].to(device)
                valid_mask = all_valid_masks[i].to(device)
                
                kl_per_sample = self.compute_single_kl(
                    ref_model, seq_gpu, token_log_probs, shift_labels, valid_mask
                )
                total_kl = total_kl + kl_per_sample.sum()
                
                del seq_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 8. Compute policy gradient
        token_log_probs = torch.cat([lp.to(device) for lp in all_log_probs], dim=0)
        valid_mask = torch.cat([vm.to(device) for vm in all_valid_masks], dim=0)
        
        seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
        pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
        
        eps = 1e-8
        total_samples = B * effective_G
        kl_penalty = self.kl_coef * total_kl / (total_samples + eps) if ref_model else 0.0
        
        total_loss = pg_loss + kl_penalty
        
        # Logging
        reward_means = rewards.mean(dim=(0, 1))
        kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
        
        components = {
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "uncertainty_mean": uncertainty_scores.mean().item(),
            "uncertainty_std": uncertainty_scores.std().item(),
            "prob_correct_mean": prob_correct_all.mean().item(),
            "prob_correct_std": prob_correct_all.std().item(),
            "advantage_mean": final_advantages_flat.mean().item(),
            "advantage_std": final_advantages_flat.std().item(),
        }
        
        if has_tool_reward:
            components["reward_tool_format"] = reward_means[2].item()
            components["reward_accuracy"] = reward_means[3].item()
            components["reward_uncertainty"] = reward_means[4].item()
            idx = 5
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
            components["reward_tool_correctness"] = reward_means[idx].item()
            idx += 1
        else:
            components["reward_accuracy"] = reward_means[2].item()
            components["reward_uncertainty"] = reward_means[3].item()
            idx = 4
            if has_judge:
                components["reward_reasoning_quality"] = reward_means[idx].item()
                idx += 1
        
        if temp_rewards is not None:
            components["reward_temperature"] = reward_means[idx].item()
        
        # Extract first generated sample for predict logging before cleanup
        # del right before return has no VRAM benefit (locals freed on return anyway)
        pred_ids = padded_sequences_list[0][:1, input_len:].clone()  # (1, gen_len)
        pred_mask = (pred_ids != self.tokenizer.pad_token_id)        # (1, gen_len)

        # Memory cleanup
        del sequences_list, padded_sequences_list, rewards_flat, rewards
        del all_log_probs, all_valid_masks, all_shift_labels
        del all_prob_correct, all_uncertainty, uncertainty_scores, prob_correct_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return LossResult(
            total_loss=total_loss, components=components, outputs=None,
            predicted_ids=pred_ids, response_mask=pred_mask,
        )


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
