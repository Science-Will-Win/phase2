"""
Training CSV Logger - Module for logging training results to CSV

LossResult: Loss function return format
TrainingLogger: CSV logging management
CSVLoggingCallback: Integration with Trainer
"""
import os
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
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
    - on_train_end: Final save when training ends
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
        """Final save when training ends"""
        self.logger.save()
        return control
