import torch
import torch.nn as nn
import torch.nn.functional as F

from training_logger import LossResult
from heteroscedastic_utils import (
    compute_heteroscedastic_log_probs,
    compute_learned_heteroscedastic_log_probs
)

def get_loss_handler(loss_type):
    """
    Returns a loss computation function based on loss_type.
    The returned function should have signature:
    func(model, inputs, trainer) -> LossResult
    
    Available loss types:
    - "cross_entropy": Standard causal LM loss
    - "gdpo": Group reward-Decoupled Normalization Policy Optimization (3 objectives)
    - "heteroscedastic_gdpo": GDPO with uncertainty reward (4 objectives)
    - "non_learnable_heteroscedastic_uncertainty": σ = logits.std() (not learned)
    - "heteroscedastic_uncertainty": Learned heteroscedastic uncertainty (σ from model)
    
    Note: "gdpo" and "heteroscedastic_gdpo" now use class-based implementation 
    with optional temperature contrastive support. Configure via trainer.gdpo_config:
    - use_temperature_contrastive: bool (default False)
    - low_temperature: float (default 0.3)
    - high_temperature: float (default 1.2)
    """
    if loss_type == "cross_entropy":
        return compute_cross_entropy_loss
    elif loss_type == "gdpo":
        import GDPO  # Lazy import to avoid circular dependency
        return GDPO.compute_gdpo_loss
    elif loss_type == "non_learnable_heteroscedastic_uncertainty":
        return non_learnable_heteroscedastic_uncertainty_loss
    elif loss_type == "heteroscedastic_gdpo":
        import GDPO  # Lazy import to avoid circular dependency
        return GDPO.compute_heteroscedastic_gdpo_loss
    elif loss_type == "heteroscedastic_uncertainty":
        return heteroscedastic_uncertainty_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. "
                        f"Available: cross_entropy, gdpo, heteroscedastic_gdpo, "
                        f"non_learnable_heteroscedastic_uncertainty, heteroscedastic_uncertainty")

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
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return LossResult(
            total_loss=loss,
            components={"cross_entropy": loss.item()},
            outputs=outputs
        )
        
    raise ValueError("Could not compute loss: 'loss' not in outputs and 'labels' missing.")


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
    sequential = getattr(trainer_context, 'heteroscedastic_sequential', False)
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

    # Pre-compute predictions before freeing outputs for predict logging
    # shift_logits is an independent copy (.contiguous()), so argmax is safe before del
    predicted_ids = shift_logits.argmax(dim=-1).detach()  # (batch, seq_len-1)
    pred_response_mask = (shift_labels != -100)            # (batch, seq_len-1)

    # Free original tensors before MC sampling to reclaim VRAM
    # shift_logits/shift_log_var are independent copies (.contiguous()), so this is safe
    # Saves ~10.7 GB (the original logits tensor held by outputs)
    del logits, log_variance, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Monte Carlo log probs with LEARNED σ
    log_probs, sigma_stats = compute_learned_heteroscedastic_log_probs(
        shift_logits, shift_log_var, shift_labels, T=T, sequential=sequential
    )
    
    # Masking
    valid_mask = (shift_labels != -100).float()
    
    # Heteroscedastic Loss = -Σ log_prob / num_valid
    loss = -(log_probs * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    
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
        print(f"[DEBUG] Heteroscedastic Loss: {loss.item():.4f}")
        print(f"[DEBUG] Total Loss: {loss.item():.4f}")
    
    return LossResult(
        total_loss=loss,
        components={
            "heteroscedastic_loss": loss.item(),
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
            "variance_mean": sigma_stats["variance_mean"],
            "log_variance_mean": sigma_stats["log_variance_mean"],
        },
        outputs=None,          # outputs freed before MC sampling for VRAM efficiency
        predicted_ids=predicted_ids,
        response_mask=pred_response_mask,
    )


def non_learnable_heteroscedastic_uncertainty_loss(model, inputs, trainer_context) -> LossResult:
    """
    Non-Learnable Heteroscedastic Uncertainty Loss.
    
    Same formula as heteroscedastic_uncertainty_loss, but σ = logits.std() (not learned).
    Uses Monte Carlo sampling with σ computed from logits standard deviation.
    
    Formula (Kendall & Gal, 2017):
        x̂ = f + σ * ε,  ε ~ N(0, I)
        L = -log( (1/T) * Σ_t softmax(x̂_t)[c] )
    
    Where σ = std(logits) across vocab dimension (detached, not learned).
    
    Returns:
        LossResult with components: {"heteroscedastic_loss", "cross_entropy", "sigma_*"}
    """
    T = getattr(trainer_context, 'heteroscedastic_T', 3)
    sequential = getattr(trainer_context, 'heteroscedastic_sequential', False)
    debug = getattr(trainer_context, 'debug', False)
    
    # Extract labels before forward pass
    labels = inputs.get("labels")
    
    # Forward pass WITHOUT labels to prevent HF from computing internal loss
    forward_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
    outputs = model(**forward_inputs)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    if logits is None or labels is None:
        raise ValueError("Could not compute heteroscedastic loss: logits or labels missing.")
    
    # Shift for next-token prediction (same as heteroscedastic_uncertainty_loss)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Monte Carlo log probs with σ = logits.std() (not learned)
    if debug:
        log_probs, debug_info = compute_heteroscedastic_log_probs(
            shift_logits, shift_labels, T=T, sequential=sequential, debug=True
        )
        sigma_stats = {k: v for k, v in debug_info.items() if k.startswith("sigma_")}
        print(f"[DEBUG] Non-Learnable Heteroscedastic Uncertainty Loss - T={T}, sequential={sequential}")
        print(f"[DEBUG] Sigma - Mean: {debug_info['sigma_mean']:.4f}, "
              f"Std: {debug_info['sigma_std']:.4f}, "
              f"Range: [{debug_info['sigma_min']:.4f}, {debug_info['sigma_max']:.4f}]")
        print(f"[DEBUG] Avg Prob - Mean: {debug_info['avg_prob_mean']:.6f}, "
              f"Std: {debug_info['avg_prob_std']:.6f}")
    else:
        log_probs, sigma_stats = compute_heteroscedastic_log_probs(
            shift_logits, shift_labels, T=T, sequential=sequential, return_sigma=True
        )
    
    # Masking on shifted labels
    valid_mask = (shift_labels != -100).float()
    
    # Heteroscedastic Loss = -Σ log_prob / num_valid
    loss = -(log_probs * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    
    if debug:
        valid_count = valid_mask.sum().int().item()
        log_prob_mean = (log_probs * valid_mask).sum().item() / max(valid_count, 1)
        print(f"[DEBUG] Log Prob Mean: {log_prob_mean:.4f}, Valid tokens: {valid_count}")
        print(f"[DEBUG] Heteroscedastic Loss: {loss.item():.4f}")
        print(f"[DEBUG] Total Loss: {loss.item():.4f}")
    
    return LossResult(
        total_loss=loss,
        components={
            "heteroscedastic_loss": loss.item(),
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
        },
        outputs=outputs
    )
