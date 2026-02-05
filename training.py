import argparse
import os
import sys

# Early GPU Configuration (MUST be before torch import)
from utils.gpu_config import configure_gpu
configure_gpu()

import torch
import gc
from transformers import Trainer, TrainingArguments, default_data_collator
import model as model_module
from tokenizer import TokenizerManager
import loss as loss_module
import dataset as dataset_module
from training_logger import TrainingLogger, CSVLoggingCallback
from utils import get_file_config
from utils.paths import set_local_mode, get_model_dir

def unload_model(obj_list, debug=False):
    """
    Explicitly deletes objects and clears CUDA cache.
    """
    for obj in obj_list:
        if obj is not None:
            # If obj is a Trainer, try to clear its internal references first
            if hasattr(obj, "model"):
                try:
                    obj.model.to("cpu")
                    del obj.model
                except:
                    pass
            if hasattr(obj, "optimizer"):
                del obj.optimizer
            if hasattr(obj, "lr_scheduler"):
                del obj.lr_scheduler
                
            # Move explicit model objects to CPU
            if hasattr(obj, "to"):
                try:
                    obj.to("cpu")
                except:
                    pass
            del obj
    
    # Run GC multiple times to catch cyclic references
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if debug:
        print("[DEBUG] Model and related objects unloaded. Memory cleared.")

def print_memory_stats(step_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class CustomTrainer(Trainer):
    def __init__(self, loss_type="cross_entropy", logger=None, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_handler = loss_module.get_loss_handler(loss_type)
        self.loss_type = loss_type
        self.logger = logger
        self.debug = debug
        if debug:
            print(f"[DEBUG] CustomTrainer initialized with loss_type: {self.loss_type}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Overridden compute_loss to delegate to loss handlers.
        Supports both old and new Transformers signatures.
        Loss handlers return LossResult.
        """
        # Our loss handlers return LossResult
        loss_result = self.loss_handler(model, inputs, self)
        
        # Log to CSV if logger is available
        if self.logger is not None:
            # Decode predict and label texts (response portion only)
            predict_text = None
            label_text = None
            
            # Find response start position from labels (-100 marks prompt)
            response_start = 0
            if "labels" in inputs and inputs["labels"] is not None:
                label_ids = inputs["labels"][0]
                response_mask = label_ids != -100
                if response_mask.any():
                    response_start = response_mask.nonzero()[0].item()
                
                # Decode labels (response portion only, already filtered by -100)
                valid_label_ids = label_ids[response_mask]
                if len(valid_label_ids) > 0:
                    label_text = self.processing_class.decode(
                        valid_label_ids, skip_special_tokens=True
                    )
            
            if loss_result.outputs is not None and hasattr(loss_result.outputs, 'logits'):
                # Get predicted token IDs (argmax of logits)
                predicted_ids = loss_result.outputs.logits.argmax(dim=-1)
                
                # Get actual sequence length from attention_mask (exclude padding)
                seq_len = len(predicted_ids[0])
                if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                    seq_len = inputs["attention_mask"][0].sum().item()
                
                # Decode response portion only (from response_start to seq_len, excluding padding)
                predict_ids = predicted_ids[0][response_start:seq_len]
                if len(predict_ids) > 0:
                    predict_text = self.processing_class.decode(
                        predict_ids, skip_special_tokens=True
                    )
            
            self.logger.log(
                step=self.state.global_step,
                epoch=int(self.state.epoch) + 1 if self.state.epoch is not None else 1,
                loss_result=loss_result,
                predict=predict_text,
                label=label_text
            )
        
        if return_outputs:
            return (loss_result.total_loss, loss_result.outputs)
        return loss_result.total_loss

def set_seed(seed, debug=False):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if debug:
        print(f"[DEBUG] Random seed set to {seed}")

def run_training(args):
    # Set environment based on --local flag (for path management)
    set_local_mode(args.local)
    
    # Set random seed for reproducibility
    import random as py_random
    if args.random_seed < 0:
        seed = py_random.randint(0, 2**32 - 1)
        print(f"Using random seed: {seed}")
    else:
        seed = args.random_seed
    set_seed(seed, debug=args.debug)
    
    # Initial cleanup
    unload_model([])

    # Check and print CUDA info
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is NOT available. Using CPU.")

    model = None
    trainer = None
    tokenizer = None
    ref_model = None

    try:
        # Load Tokenizer using model_type
        tok_manager = TokenizerManager(model_type=args.model_type, tokenizer_base_dir="model")
        tokenizer = tok_manager.load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load chat_template from FileConfig
        file_config = get_file_config(args.model_type)
        if file_config:
            chat_template_path = os.path.join(file_config.BASE_PATH, file_config.CHAT_TEMPLATE)
            if os.path.exists(chat_template_path):
                with open(chat_template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read().strip()
                if args.debug:
                    print(f"[DEBUG] Chat template loaded from {chat_template_path}")
            else:
                print(f"[WARNING] Chat template not found at {chat_template_path}")
        else:
            print(f"[WARNING] No FileConfig found for {args.model_type}")

        # Load Model
        model = model_module.get_model(args)
        print_memory_stats("After Base Model Load")
        
        # Ensure lm_head (or final layer) is trainable if we are freezing
        # Note: embed_tokens is excluded because it's at the input side.
        # With tie_word_embeddings=True, lm_head shares weights with embed_tokens,
        # so gradients will update the shared weight through lm_head's path only.
        if args.freeze_until_layer:
             for name, param in model.named_parameters():
                 if "lm_head" in name or "embed_out" in name:
                     param.requires_grad = True
                     if args.debug:
                         print(f"[DEBUG] Explicitly enabled gradients for: {name}")
        
        # Enable input gradients to ensure the graph is connected
        # Only needed for full training; when freezing layers, this would force
        # activation storage for all frozen layers (wasting memory)
        if not args.freeze_until_layer:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # Load Dataset using custom module
        tokenized_datasets = dataset_module.get_dataset(args, tokenizer)
        
        # Validate dataset
        if tokenized_datasets is None or len(tokenized_datasets) == 0:
            raise ValueError("Dataset is empty or None. Check --data_path argument.")

        # Calculate Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        param_str = f"{total_params/1e9:.1f}B"
        
        # Initialize CSV Logger
        logger = TrainingLogger.from_args(args, total_params)
        
        # Construct Folder Name
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        
        freeze_str = args.freeze_until_layer if args.freeze_until_layer else "full"
        
        if args.save_strategy == "steps":
            save_info = f"{args.save_steps}step"
        else:
            save_info = f"{args.save_strategy}"
            
        folder_name = f"{args.model_type}-{freeze_str}-{param_str}-{args.epochs}ep-{save_info}-{date_str}-{time_str}"
        
        base_output_dir = os.path.join("model", "train")
        final_output_dir = os.path.join(base_output_dir, folder_name)
        
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir, exist_ok=True)
            if args.debug:
                print(f"[DEBUG] Created output directory: {final_output_dir}")

        # Map torch types to TrainingArguments flags
        # This allows for easy extension (e.g. if 'fp8' becomes a flag)
        DTYPE_ARG_MAP = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
        }
        
        dtype_kwargs = {}
        # Use actual parameter dtype instead of model.dtype attribute (more reliable for custom models)
        model_dtype = next(model.parameters()).dtype
        if model_dtype in DTYPE_ARG_MAP:
             arg_name = DTYPE_ARG_MAP[model_dtype]
             # Only enable if hardware supports it (sanity check)
             if arg_name == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
                 print("[WARNING] Model is bfloat16 but hardware doesn't support it. Keeping default.")
             else:
                 dtype_kwargs[arg_name] = True
                 if args.debug:
                     print(f"[DEBUG] Syncing Trainer precision with model dtype ({model_dtype}): Enabling {arg_name}=True")
 
        # Training Arguments
        # We strictly use the arguments provided by the user + dynamic precision flags.
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_dir=os.path.join(final_output_dir, 'logs'),
            remove_unused_columns=False,
            report_to="none",
            **dtype_kwargs
        )

        if args.loss_type == "gdpo" or args.loss_type == "heteroscedastic_gdpo":
             if args.debug:
                 print("[DEBUG] Loading Reference Model for GDPO KL Penalty...")
             # Load ref model (same config/weights as base)
             # We use the same get_model logic but ensure it's frozen
             ref_model = model_module.get_model(args)
             ref_model.eval()
             for param in ref_model.parameters():
                 param.requires_grad = False
             if args.debug:
                 print("[DEBUG] Reference Model Loaded and Frozen.")
             print_memory_stats("After Ref Model Load")
        else:
             ref_model = None
             print_memory_stats("After (Skipped) Ref Model Load")

        trainer = CustomTrainer(
            loss_type=args.loss_type,
            logger=logger,
            debug=args.debug,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=default_data_collator,
            callbacks=[CSVLoggingCallback(logger)],
        )
        trainer.processing_class = tokenizer # Ensure trainer has tokenizer for GDPO
        trainer.ref_model = ref_model # Attach ref_model for GDPO
        
        # Attach GDPO configuration to trainer
        trainer.gdpo_config = {
            "group_size": args.gdpo_group_size,
            "kl_coef": args.gdpo_kl_coef,
            "max_new_tokens": args.gdpo_max_new_tokens,
            "temperature": args.gdpo_temperature,
            # Priority Variation
            "reward_weights": {
                "format": args.gdpo_reward_weight_format,
                "length": args.gdpo_reward_weight_length,
                "accuracy": args.gdpo_reward_weight_accuracy,
                "uncertainty": args.gdpo_reward_weight_uncertainty,
                "temperature": args.gdpo_reward_weight_temperature,
            },
            "use_conditioned_rewards": args.gdpo_use_conditioned_rewards,
            "accuracy_threshold": args.gdpo_accuracy_threshold,
            "target_length": args.gdpo_target_length,
            # Temperature Contrastive
            "use_temperature_contrastive": args.gdpo_use_temperature_contrastive,
            "low_temperature": args.gdpo_low_temperature,
            "high_temperature": args.gdpo_high_temperature,
            # Uncertainty Reward
            "uncertainty_threshold": args.gdpo_uncertainty_threshold,
            "uncertainty_full_sequence": args.gdpo_uncertainty_full_sequence,
            # Tool Correctness
            "tool_correctness_threshold": args.gdpo_tool_correctness_threshold,
            # Heteroscedastic weight (legacy, for heteroscedastic_gdpo before refactor)
            "heteroscedastic_weight": args.heteroscedastic_weight,
        }
        
        # Attach heteroscedastic configuration to trainer
        trainer.heteroscedastic_T = args.heteroscedastic_T

        print("Starting training...")
        print_memory_stats("Before Trainer Loop (Pre-Optimizer Init)")
        trainer.train()
        
        print(f"Saving fine-tuned model to {final_output_dir}")
        
        # Clear GPU memory before saving to avoid file write issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save model with safetensors format
        model.save_pretrained(final_output_dir, safe_serialization=True)
        print("Model saved with safetensors format.")
        
        tokenizer.save_pretrained(final_output_dir)
        print("Tokenizer saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        print("Starting cleanup sequence...")
        sys.stdout.flush()
        unload_model([model, trainer, ref_model], debug=args.debug)

if __name__ == "__main__":
    # Check for detailed help (--arg_name --help)
    from utils.detailed_help import check_detailed_help
    check_detailed_help()
    
    parser = argparse.ArgumentParser(description="Run training for a specific model type")
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--data_path", type=str, default=None)
    # parser.add_argument("--output_dir", type=str, default="fine_tuned_model") # We override this now
    
    # Standard HF Save Strategy Arguments
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"], 
                        help="The checkpoint save strategy to use.")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Save checkpoint every X updates steps.")
    
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps")
    
    # Loss Type Argument
    parser.add_argument("--loss_type", type=str, default="cross_entropy", 
                        help="Type of loss function to use (defined in loss.py)")

    # GDPO Specific Arguments
    parser.add_argument("--gdpo_group_size", type=int, default=4,
                        help="Group size G for GDPO rollout (num_return_sequences)")
    parser.add_argument("--gdpo_kl_coef", type=float, default=0.01,
                        help="KL divergence penalty coefficient (beta) for GDPO")
    parser.add_argument("--gdpo_max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate during GDPO rollout")
    parser.add_argument("--gdpo_temperature", type=float, default=1.0,
                        help="Temperature for generation diversity (higher = more diverse, 0.7-1.5 recommended)")
    
    # GDPO Priority Variation Arguments
    parser.add_argument("--gdpo_reward_weight_format", type=float, default=1.0,
                        help="Weight for format reward")
    parser.add_argument("--gdpo_reward_weight_length", type=float, default=1.0,
                        help="Weight for length reward")
    parser.add_argument("--gdpo_reward_weight_accuracy", type=float, default=1.0,
                        help="Weight for accuracy reward")
    parser.add_argument("--gdpo_use_conditioned_rewards", action="store_true",
                        help="Enable conditioning easier rewards on accuracy")
    parser.add_argument("--gdpo_accuracy_threshold", type=float, default=1.0,
                        help="Accuracy threshold for conditioned rewards (default: 1.0, binary accuracy)")
    parser.add_argument("--gdpo_target_length", type=int, default=1024,
                        help="Target length for length penalty calculation")
    
    # GDPO Temperature Contrastive Arguments (NEW)
    parser.add_argument("--gdpo_use_temperature_contrastive", action="store_true",
                        help="Enable temperature contrastive sampling (low temp=chosen, high temp=rejected)")
    parser.add_argument("--gdpo_low_temperature", type=float, default=0.3,
                        help="Low temperature for chosen samples (default: 0.3)")
    parser.add_argument("--gdpo_high_temperature", type=float, default=1.2,
                        help="High temperature for rejected samples (default: 1.2)")
    parser.add_argument("--gdpo_reward_weight_temperature", type=float, default=1.0,
                        help="Weight for temperature contrastive reward")
    
    # GDPO Uncertainty Reward Arguments (NEW)
    parser.add_argument("--gdpo_uncertainty_threshold", type=float, default=0.6,
                        help="Uncertainty threshold for penalty (default: 0.6)")
    parser.add_argument("--gdpo_reward_weight_uncertainty", type=float, default=1.0,
                        help="Weight for uncertainty reward (heteroscedastic_gdpo)")
    parser.add_argument("--gdpo_uncertainty_full_sequence", action="store_true",
                        help="Measure uncertainty on full sequence instead of reasoning section only (default: reasoning only)")
    
    # GDPO Tool Correctness Arguments
    parser.add_argument("--gdpo_tool_correctness_threshold", type=float, default=1.5,
                        help="Tool correctness threshold for conditioned rewards (default: 1.5, ~75%% match required)")
    
    # Heteroscedastic Loss Arguments
    parser.add_argument("--heteroscedastic_T", type=int, default=3,
                        help="Number of Monte Carlo samples for heteroscedastic loss (default: 3, memory-efficient)")
    parser.add_argument("--heteroscedastic_weight", type=float, default=0.1,
                        help="Weight for heteroscedastic loss term in heteroscedastic_gdpo (default: 0.1)")
    
    # Training Stability
    parser.add_argument("--random_seed", type=int, default=-1,
                        help="Random seed (-1 for random, positive for fixed seed)")
    
    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output for detailed logging")

    # Environment selection
    parser.add_argument("--local", action="store_true",
                        help="Use local paths instead of server paths (default: server). Also sets GPU default to 0.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU IDs to use (e.g., '0' or '6,7'). Default: '6,7' (server) or '0' (local)")

    # Add Model Hyperparameters
    model_module.add_model_args(parser)

    args = parser.parse_args()
    run_training(args)
