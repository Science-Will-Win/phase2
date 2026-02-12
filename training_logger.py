"""
Training CSV Logger - Module for logging training results to CSV

LossResult: Loss function return format
TrainingLogger: CSV logging management
CSVLoggingCallback: Integration with Trainer
TokenErrorTracker: Token-level error tracking for validation
"""
import os
import gc
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import Counter
import torch
from transformers import TrainerCallback
from utils.paths import get_result_dir


@dataclass
class LossResult:
    """
    Loss function return format.
    All loss functions should return this format.
    
    Attributes:
        total_loss: Final loss used for backpropagation (torch.Tensor)
        components: Individual loss values {name: value} - used as CSV column names
        outputs: Model outputs (logits, etc.), optional
    """
    total_loss: torch.Tensor
    components: Dict[str, float] = field(default_factory=dict)
    outputs: Any = None
    
    @property
    def component_names(self) -> List[str]:
        """Loss component names to use as CSV column names"""
        return list(self.components.keys())


class TrainingLogger:
    """
    Logs training results to CSV.
    
    Filename format: {model_type}-{freeze_str}-{param_str}-{epochs}ep-{save_info}-{date_str}-{time_str}.csv
    Save location: result/
    """
    
    def __init__(
        self,
        model_type: str,
        freeze_str: str,
        param_str: str,
        epochs: int,
        save_info: str,
        output_dir: str = None
    ):
        """
        Args:
            model_type: Model type (e.g., "ministral_3_3b_instruct")
            freeze_str: Freeze setting (e.g., "full", "layer_10")
            param_str: Parameter count (e.g., "3.2B")
            epochs: Total training epochs
            save_info: Save strategy info (e.g., "epoch", "500step")
            output_dir: CSV save folder (default: from config.yaml)
        """
        # Use configured result directory if not specified
        if output_dir is None:
            output_dir = get_result_dir()
        
        # Generate filename
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        
        filename = f"{model_type}-{freeze_str}-{param_str}-{epochs}ep-{save_info}-{date_str}-{time_str}.csv"
        
        # Create result folder
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, filename)
        
        self.records: List[Dict] = []
        
        print(f"[TrainingLogger] Will save to: {self.output_path}")
    
    @classmethod
    def from_args(cls, args, total_params: int) -> "TrainingLogger":
        """
        Create directly from argparse args.
        
        Args:
            args: Arguments parsed by argparse
            total_params: Total model parameters
            
        Returns:
            TrainingLogger instance
        """
        freeze_str = args.freeze_until_layer if args.freeze_until_layer else "full"
        param_str = f"{total_params/1e9:.1f}B"
        
        if args.save_strategy == "steps":
            save_info = f"{args.save_steps}step"
        else:
            save_info = f"{args.save_strategy}"
        
        return cls(
            model_type=args.model_type,
            freeze_str=freeze_str,
            param_str=param_str,
            epochs=args.epochs,
            save_info=save_info
        )
    
    def log(
        self,
        step: int,
        epoch: float,
        loss_result: Optional[LossResult] = None,
        predict: Optional[str] = None,
        label: Optional[str] = None,
        **extra
    ):
        """
        Log results for one step.
        
        Args:
            step: Global step number
            epoch: Current epoch (float, e.g., 0.5, 1.0)
            loss_result: LossResult object (contains loss components)
            predict: Model prediction text (optional)
            label: Ground truth text (optional)
            **extra: Additional fields to log
        """
        record = {
            "step": step,
            "epoch": epoch,
            "predict": predict,
            "label": label,
        }
        
        # Add loss components (dynamic columns)
        if loss_result is not None:
            total_loss_val = loss_result.total_loss.item() \
                if isinstance(loss_result.total_loss, torch.Tensor) \
                else loss_result.total_loss
            record["total_loss"] = total_loss_val
            record.update(loss_result.components)
        
        record.update(extra)
        self.records.append(record)
    
    def save(self):
        """Save to CSV file (overwrite)"""
        if not self.records:
            print("[TrainingLogger] No records to save.")
            return
        
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"[TrainingLogger] Saved {len(self.records)} records to {self.output_path}")


class CSVLoggingCallback(TrainerCallback):
    """
    Trainer Callback to synchronize CSV saving with model checkpoint.
    
    - on_save: Save CSV when checkpoint is saved
    - on_train_end: Final save when training ends + generate figures
    """
    
    def __init__(self, logger: TrainingLogger):
        """
        Args:
            logger: TrainingLogger instance
        """
        self.logger = logger
    
    def on_save(self, args, state, control, **kwargs):
        """Save CSV when model checkpoint is saved"""
        self.logger.save()
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Final save when training ends + generate figures"""
        self.logger.save()
        
        # Generate visualization figures
        try:
            from visualization import generate_training_figures
            generate_training_figures(self.logger.records, self.logger.output_path)
        except ImportError as e:
            print(f"[CSVLoggingCallback] Warning: Could not import visualization module: {e}")
        except Exception as e:
            print(f"[CSVLoggingCallback] Warning: Figure generation failed: {e}")
        
        # Force garbage collection to free memory after visualization
        gc.collect()
        
        return control


class EvalAccuracyCallback(TrainerCallback):
    """
    Callback to log epoch-level accuracy from TokenErrorTracker.
    
    - on_evaluate: Get accuracy from tracker, log to records, reset epoch counters
    """
    
    def __init__(self, trainer_ref, logger: TrainingLogger):
        """
        Args:
            trainer_ref: Reference to the CustomTrainer (to access token_error_tracker)
            logger: TrainingLogger instance (to log eval_accuracy)
        """
        self.trainer_ref = trainer_ref
        self.logger = logger
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log epoch accuracy, eval_loss, and reset epoch counters."""
        trainer = self.trainer_ref
        epoch = int(state.epoch) if state.epoch else 0
        
        # Capture eval_loss from HuggingFace metrics
        if metrics and 'eval_loss' in metrics:
            self.logger.log(
                step=state.global_step,
                epoch=epoch,
                eval_loss=metrics['eval_loss']
            )
        
        if hasattr(trainer, 'token_error_tracker'):
            accuracy = trainer.token_error_tracker.get_epoch_accuracy()
            
            # Log eval_accuracy to records for visualization
            self.logger.log(
                step=state.global_step,
                epoch=epoch,
                eval_accuracy=accuracy
            )
            
            # Reset epoch counters for next evaluation
            trainer.token_error_tracker.reset_epoch()
            
            print(f"[EvalAccuracyCallback] Epoch {epoch} - Val Loss: {metrics.get('eval_loss', 'N/A'):.4f}, Val Accuracy: {accuracy:.2f}%")
        
        return control


class EpochSummaryCallback(TrainerCallback):
    """
    Callback to log epoch-level training summary.
    Logs average loss and accuracy at the end of each epoch.
    
    This provides aggregate statistics even when detailed logging
    is disabled (via log_every_n_epochs > 1).
    """
    
    def __init__(self, trainer_ref, logger: TrainingLogger):
        """
        Args:
            trainer_ref: Reference to the CustomTrainer (to access epoch accumulators)
            logger: TrainingLogger instance (to log train_loss/train_accuracy)
        """
        self.trainer_ref = trainer_ref
        self.logger = logger
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch average loss and accuracy, then reset accumulators."""
        trainer = self.trainer_ref
        epoch = int(state.epoch) if state.epoch else 0
        
        if trainer.epoch_step_count > 0:
            avg_loss = trainer.epoch_loss_sum / trainer.epoch_step_count
            avg_acc = trainer.epoch_accuracy_sum / trainer.epoch_step_count
            
            self.logger.log(
                step=state.global_step,
                epoch=epoch,
                train_loss=avg_loss,
                train_accuracy=avg_acc
            )
            
            # Print summary for debugging
            print(f"[EpochSummaryCallback] Epoch {epoch} - Avg Train Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc:.2f}%")
        
        # Reset for next epoch
        trainer.epoch_loss_sum = 0.0
        trainer.epoch_accuracy_sum = 0.0
        trainer.epoch_step_count = 0
        
        return control


class TokenErrorTracker:
    """
    Tracks token-level prediction errors during validation.
    Excludes only pad tokens from tracking.
    
    Supports dual tracking:
    - Cumulative: For final error analysis (token_errors.csv)
    - Per-epoch: For visualization (accuracy plots)
    """
    
    def __init__(self, pad_token_id=None):
        """
        Args:
            pad_token_id: Token ID for padding (will be excluded from tracking)
        """
        # Cumulative counters (entire training)
        self.error_counts = Counter()  # (predicted_id, label_id) -> count
        self.total_tokens = 0
        self.correct_tokens = 0
        self.pad_token_id = pad_token_id
        
        # Per-epoch counters (reset each epoch)
        self.epoch_total = 0
        self.epoch_correct = 0
    
    def update(self, predicted_ids, label_ids, mask):
        """
        Update error counts from a batch.
        
        Args:
            predicted_ids: (batch, seq_len) - argmax of logits
            label_ids: (batch, seq_len) - ground truth
            mask: (batch, seq_len) - valid positions (label != -100)
        """
        pred_flat = predicted_ids[mask].cpu().tolist()
        label_flat = label_ids[mask].cpu().tolist()
        
        for pred, label in zip(pred_flat, label_flat):
            # Skip pad tokens only
            if self.pad_token_id is not None and label == self.pad_token_id:
                continue
            
            # Update cumulative counters
            self.total_tokens += 1
            # Update per-epoch counters
            self.epoch_total += 1
            
            if pred == label:
                self.correct_tokens += 1
                self.epoch_correct += 1
            else:
                self.error_counts[(pred, label)] += 1
    
    def get_epoch_accuracy(self) -> float:
        """
        Get accuracy for the current epoch.
        
        Returns:
            Accuracy as percentage (0-100)
        """
        if self.epoch_total == 0:
            return 0.0
        return self.epoch_correct / self.epoch_total * 100
    
    def reset_epoch(self):
        """
        Reset per-epoch counters.
        Called after each evaluation epoch.
        Cumulative counters are preserved for final error analysis.
        """
        self.epoch_total = 0
        self.epoch_correct = 0
    
    def save_csv(self, output_path, tokenizer, top_k=100):
        """
        Save top-k errors to CSV with summary statistics at the end.
        
        Args:
            output_path: Path to save the CSV file
            tokenizer: Tokenizer for decoding token IDs
            top_k: Number of top errors to save (default: 100)
        """
        accuracy = self.correct_tokens / self.total_tokens * 100 if self.total_tokens > 0 else 0
        total_errors = self.total_tokens - self.correct_tokens
        
        # Error records
        records = []
        for (pred_id, label_id), count in self.error_counts.most_common(top_k):
            pred_token = tokenizer.decode([pred_id])
            label_token = tokenizer.decode([label_id])
            error_rate = count / self.total_tokens * 100
            records.append({
                "rank": len(records) + 1,
                "predicted_token": repr(pred_token),
                "label_token": repr(label_token),
                "count": count,
                "error_rate": f"{error_rate:.2f}%"
            })
        
        # Add empty row as separator
        records.append({
            "rank": "",
            "predicted_token": "",
            "label_token": "",
            "count": "",
            "error_rate": ""
        })
        
        # Add summary statistics at the end
        records.append({
            "rank": "SUMMARY",
            "predicted_token": f"total_tokens: {self.total_tokens}",
            "label_token": f"correct: {self.correct_tokens}",
            "count": f"errors: {total_errors}",
            "error_rate": f"accuracy: {accuracy:.2f}%"
        })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[TokenErrorTracker] Saved {len(records)-2} error types to {output_path}")
        print(f"[TokenErrorTracker] Accuracy: {accuracy:.2f}% ({self.correct_tokens}/{self.total_tokens})")
