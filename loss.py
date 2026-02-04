import torch
import torch.nn as nn
import torch.nn.functional as F

import GDPO
from training_logger import LossResult


def compute_heteroscedastic_log_probs(logits, labels, T=3, debug=False, return_sigma=False):
    """
    Compute log probabilities with heteroscedastic Monte Carlo sampling.
    Memory-efficient implementation.
    
    Formula: log_prob = log( (1/T) * Σ_t softmax(f + σ*ε_t)[c] )
    
    Where:
        f = logits (model output)
        σ = std(f) across vocab dimension (detached for memory efficiency)
        ε_t ~ N(0, 1)
        c = ground truth token index
    
    Args:
        logits: (batch, seq_len, vocab_size) - raw logits f_i
        labels: (batch, seq_len) - ground truth token ids (may contain -100)
        T: Number of Monte Carlo samples (default: 3 for memory efficiency)
        debug: If True, return debug_info dict along with log_probs
        return_sigma: If True, return sigma statistics (for logging)
    
    Returns:
        log_probs: (batch, seq_len) - Monte Carlo averaged log probabilities
                   (invalid positions have undefined values - caller handles masking)
        sigma_stats: (if return_sigma=True) dict with sigma statistics
        debug_info: (if debug=True) dict with sigma and avg_prob statistics
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # σ_i = std(f_i) across vocab dimension
    # Detach sigma - it's a statistic, gradients flow through logits directly
    sigma = logits.std(dim=-1, keepdim=True).detach()  # (batch, seq_len, 1)
    
    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)
    
    # Monte Carlo sampling: Σ_t softmax(f + σ*ε)[c]
    # Each vocab element gets independent noise (required for softmax to change)
    # Using logsumexp for memory efficiency (avoids storing full softmax output)
    prob_sum = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
    
    for t in range(T):
        # Generate noise: ε_t is per-vocab (independent noise for each class)
        # This is required because softmax(x + c) = softmax(x) for scalar c
        with torch.no_grad():
            epsilon_t = torch.randn(batch_size, seq_len, vocab_size,
                                    device=logits.device, dtype=logits.dtype)
        
        # x_hat = f + σ*ε_t (each vocab element gets different perturbation)
        # Gradients flow through logits
        x_hat = logits + sigma * epsilon_t
        del epsilon_t
        
        # Use logsumexp for memory efficiency (avoid storing full softmax output)
        # softmax(x)_c = exp(x_c) / sum(exp(x)) = exp(x_c - logsumexp(x))
        log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)  # (batch, seq, 1)
        x_c = x_hat.gather(dim=-1, index=labels_idx)  # (batch, seq, 1)
        del x_hat
        
        # prob_c = softmax(x_hat)_c
        prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)  # (batch, seq)
        del x_c, log_sum_exp
        
        # Accumulate
        prob_sum = prob_sum + prob_c
        del prob_c
        
        # Clear CUDA cache periodically
        if t < T - 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # (1/T) * Σ_t p_{i,t,c}
    avg_prob = prob_sum / T
    del prob_sum
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_stats = {
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std().item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
    }
    
    if debug:
        debug_info = {
            **sigma_stats,
            "avg_prob_mean": avg_prob.mean().item(),
            "avg_prob_std": avg_prob.std().item(),
        }
        return log_probs, debug_info
    
    if return_sigma:
        return log_probs, sigma_stats
    
    return log_probs

def get_loss_handler(loss_type):
    """
    Returns a loss computation function based on loss_type.
    The returned function should have signature:
    func(model, inputs, trainer) -> LossResult
    
    Available loss types:
    - "cross_entropy": Standard causal LM loss
    - "gdpo": Group reward-Decoupled Normalization Policy Optimization (3 objectives)
    - "heteroscedastic_gdpo": GDPO with uncertainty reward (4 objectives)
    - "heteroscedastic_cross_entropy": CE loss with heteroscedastic uncertainty
    - "heteroscedastic_uncertainty": Learned heteroscedastic uncertainty loss
    
    Note: "gdpo" and "heteroscedastic_gdpo" now use class-based implementation 
    with optional temperature contrastive support. Configure via trainer.gdpo_config:
    - use_temperature_contrastive: bool (default False)
    - low_temperature: float (default 0.3)
    - high_temperature: float (default 1.2)
    """
    if loss_type == "cross_entropy":
        return compute_cross_entropy_loss
    elif loss_type == "gdpo":
        return GDPO.compute_gdpo_loss
    elif loss_type == "heteroscedastic_cross_entropy":
        return compute_heteroscedastic_cross_entropy_loss
    elif loss_type == "heteroscedastic_gdpo":
        return GDPO.compute_heteroscedastic_gdpo_loss
    elif loss_type == "heteroscedastic_uncertainty":
        return heteroscedastic_uncertainty_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. "
                        f"Available: cross_entropy, gdpo, heteroscedastic_gdpo, "
                        f"heteroscedastic_cross_entropy, heteroscedastic_uncertainty")

def compute_cross_entropy_loss(model, inputs, trainer_context) -> LossResult:
    """
    Standard Causal LM Loss (Next Token Prediction).
    
    Returns:
        LossResult with components: {"cross_entropy": loss_value}
    """
    # Standard forward pass
    # labels are usually automatically included in inputs by DataCollator
    outputs = model(**inputs)
    
    # Default CausalLM models in HF return 'loss' in outputs if labels are present.
    if "loss" in outputs:
        loss = outputs["loss"]
        return LossResult(
            total_loss=loss,
            components={"cross_entropy": loss.item()},
            outputs=outputs
        )
    
    # If not, manual calculation (fallback)
    logits = outputs.get("logits")
    labels = inputs.get("labels")
    
    if logits is not None and labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return LossResult(
            total_loss=loss,
            components={"cross_entropy": loss.item()},
            outputs=outputs
        )
        
    raise ValueError("Could not compute loss: 'loss' not in outputs and 'labels' missing.")


def compute_learned_heteroscedastic_log_probs(logits, log_variance, labels, T=3):
    """
    Compute log probabilities with LEARNED heteroscedastic Monte Carlo sampling.
    
    Formula (Kendall & Gal, 2017 Eq. 12):
        x̂_{i,t} = f_i + σ_i * ε_t,  ε_t ~ N(0, I)
        L = log( (1/T) * Σ_t exp(x̂_{i,t,c} - log Σ_{c'} exp(x̂_{i,t,c'})) )
    
    Key difference from compute_heteroscedastic_log_probs:
        - σ is LEARNED from model output (log_variance), not computed from logits.std()
        - σ = exp(log_variance / 2)  where log_variance = s = log(σ²)
    
    Args:
        logits: (batch, seq_len, vocab_size) - raw logits f_i
        log_variance: (batch, seq_len, 1) - learned log(σ²) from model
        labels: (batch, seq_len) - ground truth token ids (may contain -100)
        T: Number of Monte Carlo samples
    
    Returns:
        log_probs: (batch, seq_len) - Monte Carlo averaged log probabilities
        sigma_stats: dict with sigma statistics for logging
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # σ_i = exp(s_i / 2) where s_i = log(σ²_i) is learned from model
    # This is the key difference: σ is LEARNED, not computed from logits
    sigma = torch.exp(log_variance / 2)  # (batch, seq_len, 1)
    
    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)
    
    # Monte Carlo sampling: Σ_t softmax(f + σ*ε)[c]
    prob_sum = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
    
    for t in range(T):
        # Generate noise: ε_t is per-vocab (independent noise for each class)
        with torch.no_grad():
            epsilon_t = torch.randn(batch_size, seq_len, vocab_size,
                                    device=logits.device, dtype=logits.dtype)
        
        # x_hat = f + σ*ε_t
        # Gradients flow through BOTH logits AND sigma (log_variance)
        x_hat = logits + sigma * epsilon_t
        del epsilon_t
        
        # Use logsumexp for memory efficiency
        log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)
        x_c = x_hat.gather(dim=-1, index=labels_idx)
        del x_hat
        
        # prob_c = softmax(x_hat)_c
        prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)
        del x_c, log_sum_exp
        
        prob_sum = prob_sum + prob_c
        del prob_c
        
        if t < T - 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # (1/T) * Σ_t p_{i,t,c}
    avg_prob = prob_sum / T
    del prob_sum
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_squeezed = sigma.squeeze(-1)
    variance = torch.exp(log_variance.squeeze(-1))  # σ²
    
    sigma_stats = {
        "sigma_mean": sigma_squeezed.mean().item(),
        "sigma_std": sigma_squeezed.std().item(),
        "sigma_min": sigma_squeezed.min().item(),
        "sigma_max": sigma_squeezed.max().item(),
        "variance_mean": variance.mean().item(),
        "log_variance_mean": log_variance.squeeze(-1).mean().item(),
    }
    
    return log_probs, sigma_stats


def heteroscedastic_uncertainty_loss(model, inputs, trainer_context) -> LossResult:
    """
    Learned Heteroscedastic Uncertainty Loss (Kendall & Gal, 2017 Eq. 12).
    
    Requires model to output both logits and log_variance (e.g., Mistral3HeteroscedasticForConditionalGeneration).
    
    Uses Monte Carlo sampling with LEARNED σ:
        x̂ = f + σ * ε,  ε ~ N(0, I)
        L = -log( (1/T) * Σ_t softmax(x̂_t)[c] )
    
    Where σ = exp(log_variance / 2) is learned by the model.
    
    Returns:
        LossResult with components: {"heteroscedastic_loss", "sigma_*", "variance_*", "cross_entropy"}
    """
    T = getattr(trainer_context, 'heteroscedastic_T', 3)
    debug = getattr(trainer_context, 'debug', False)
    
    # Extract labels before forward pass
    labels = inputs.get("labels")
    
    # Forward pass WITHOUT labels to prevent HF from computing internal loss
    forward_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
    outputs = model(**forward_inputs)
    
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    log_variance = outputs.log_variance  # (batch, seq_len, 1)
    
    if logits is None or labels is None:
        raise ValueError("Could not compute heteroscedastic uncertainty loss: logits or labels missing.")
    
    if log_variance is None:
        raise ValueError("Model does not output log_variance. Use a heteroscedastic model architecture.")
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_log_var = log_variance[..., :-1, :].contiguous()
    
    # Monte Carlo log probs with LEARNED σ
    log_probs, sigma_stats = compute_learned_heteroscedastic_log_probs(
        shift_logits, shift_log_var, shift_labels, T=T
    )
    
    # Masking
    valid_mask = (shift_labels != -100).float()
    
    # Heteroscedastic Loss = -Σ log_prob / num_valid
    loss = -(log_probs * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    
    # Standard Cross-Entropy Loss (for comparison/logging)
    ce_loss_fct = nn.CrossEntropyLoss(reduction='none')
    ce_token_loss = ce_loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    ce_valid_mask = valid_mask.view(-1)
    ce_loss = (ce_token_loss * ce_valid_mask).sum() / ce_valid_mask.sum().clamp(min=1)
    
    if debug:
        valid_count = valid_mask.sum().int().item()
        log_prob_mean = (log_probs * valid_mask).sum().item() / max(valid_count, 1)
        print(f"[DEBUG] Heteroscedastic Uncertainty Loss - T={T}")
        print(f"[DEBUG] Sigma - Mean: {sigma_stats['sigma_mean']:.4f}, "
              f"Std: {sigma_stats['sigma_std']:.4f}, "
              f"Range: [{sigma_stats['sigma_min']:.4f}, {sigma_stats['sigma_max']:.4f}]")
        print(f"[DEBUG] Variance Mean: {sigma_stats['variance_mean']:.4f}, "
              f"Log Variance Mean: {sigma_stats['log_variance_mean']:.4f}")
        print(f"[DEBUG] Log Prob Mean: {log_prob_mean:.4f}, Valid tokens: {valid_count}")
        print(f"[DEBUG] Cross-Entropy Loss: {ce_loss.item():.4f}")
        print(f"[DEBUG] Heteroscedastic Loss: {loss.item():.4f}")
    
    return LossResult(
        total_loss=loss,
        components={
            "heteroscedastic_loss": loss.item(),
            "cross_entropy": ce_loss.item(),
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
            "variance_mean": sigma_stats["variance_mean"],
            "log_variance_mean": sigma_stats["log_variance_mean"],
        },
        outputs=outputs
    )


def compute_heteroscedastic_cross_entropy_loss(model, inputs, trainer_context) -> LossResult:
    """
    Heteroscedastic Cross-Entropy Loss.
    
    Uses Monte Carlo sampling to account for uncertainty in logits.
    
    Formula:
        L = -Σ_i log( (1/T) * Σ_t softmax(f_i + σ_i*ε_t)[c_i] )
    
    Returns:
        LossResult with components: {"heteroscedastic_ce": loss_value, "sigma_*": variance stats}
    """
    T = getattr(trainer_context, 'heteroscedastic_T', 3)
    debug = getattr(trainer_context, 'debug', False)
    
    # Extract labels before forward pass
    labels = inputs.get("labels")  # (batch, seq_len)
    
    # Forward pass WITHOUT labels to prevent HF from computing internal loss
    # This saves ~260MB VRAM by avoiding duplicate loss computation
    forward_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
    outputs = model(**forward_inputs)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    if logits is None or labels is None:
        raise ValueError("Could not compute heteroscedastic loss: logits or labels missing.")
    
    # Heteroscedastic log_probs with sigma statistics
    if debug:
        log_probs, debug_info = compute_heteroscedastic_log_probs(logits, labels, T=T, debug=True)
        sigma_stats = {k: v for k, v in debug_info.items() if k.startswith("sigma_")}
        print(f"[DEBUG] Heteroscedastic CE - T={T}")
        print(f"[DEBUG] Sigma - Mean: {debug_info['sigma_mean']:.4f}, "
              f"Std: {debug_info['sigma_std']:.4f}, "
              f"Range: [{debug_info['sigma_min']:.4f}, {debug_info['sigma_max']:.4f}]")
        print(f"[DEBUG] Avg Prob - Mean: {debug_info['avg_prob_mean']:.6f}, "
              f"Std: {debug_info['avg_prob_std']:.6f}")
    else:
        log_probs, sigma_stats = compute_heteroscedastic_log_probs(logits, labels, T=T, return_sigma=True)
    
    # Masking (handled here, not in heteroscedastic function)
    valid_mask = (labels != -100).float()
    
    # Heteroscedastic Loss = -Σ log_prob / num_valid
    loss = -(log_probs * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    
    # Standard Cross-Entropy Loss (for comparison/logging)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss_fct = nn.CrossEntropyLoss(reduction='none')
    ce_token_loss = ce_loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    ce_valid_mask = (shift_labels != -100).float().view(-1)
    ce_loss = (ce_token_loss * ce_valid_mask).sum() / ce_valid_mask.sum().clamp(min=1)
    
    if debug:
        valid_count = valid_mask.sum().int().item()
        log_prob_mean = (log_probs * valid_mask).sum().item() / max(valid_count, 1)
        print(f"[DEBUG] Log Prob - Mean: {log_prob_mean:.4f}, Valid tokens: {valid_count}")
        print(f"[DEBUG] Cross-Entropy Loss: {ce_loss.item():.4f}")
        print(f"[DEBUG] Heteroscedastic Loss: {loss.item():.4f}")
    
    return LossResult(
        total_loss=loss,
        components={
            "cross_entropy": ce_loss.item(),
            "heteroscedastic_ce": loss.item(),
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
        },
        outputs=outputs
    )
