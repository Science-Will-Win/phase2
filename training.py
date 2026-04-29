"""
Training script for fine-tuning language models.
Supports various loss types including GDPO and heteroscedastic variants.

Note: Heavy imports (torch, transformers, etc.) are deferred to run_training()
for fast --help response.
"""
import argparse
import os
import sys


def run_training(args):
    """Main training function. Heavy imports happen here."""
    
    # ==========================================================================
    # Heavy Imports (deferred for fast --help)
    # ==========================================================================
    
    # GPU Configuration (MUST be before torch import)
    from utils.gpu_config import configure_gpu
    configure_gpu()
    
    import torch
    import gc
    from transformers import Trainer, TrainingArguments, default_data_collator, EarlyStoppingCallback
    import numpy as np
    import model as model_module
    from tokenizer import TokenizerManager
    import loss as loss_module
    import data_loaders as dataset_module
    from training_logger import TrainingLogger, CSVLoggingCallback, EvalAccuracyCallback, EpochSummaryCallback
    from utils import get_file_config, get_token_config
    from utils.paths import set_local_mode, get_model_dir, get_result_dir
    
    # ==========================================================================
    # Helper Functions (defined here to use imports above)
    # ==========================================================================
    
    def unload_model(obj_list, debug=False):
        """Explicitly deletes objects and clears CUDA cache."""
        for obj in obj_list:
            if obj is not None:
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
                if hasattr(obj, "to"):
                    try: 
                        obj.to("cpu")
                    except:
                        pass
                del obj
        
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if debug:
            print("[DEBUG] Model and related objects unloaded. Memory cleared.")

    def print_memory_stats(step_name):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    def set_seed(seed, debug=False):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if debug:
            print(f"[DEBUG] Random seed set to {seed}")

    # ==========================================================================
    # CustomTrainer Class
    # ==========================================================================
    
    class CustomTrainer(Trainer):
        def __init__(self, loss_type="cross_entropy", logger=None, debug=False, 
                     track_token_errors=False, pad_token_id=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_handler = loss_module.get_loss_handler(loss_type)
            self.loss_type = loss_type
            self.logger = logger
            self.debug = debug
            self.track_token_errors = track_token_errors  # For saving final CSV
            
            # Always enable TokenErrorTracker if we have eval_dataset
            # (for epoch-level accuracy tracking and visualization)
            has_eval = self.eval_dataset is not None and len(self.eval_dataset) > 0
            if has_eval or track_token_errors:
                from training_logger import TokenErrorTracker
                self.token_error_tracker = TokenErrorTracker(pad_token_id=pad_token_id)
                if debug:
                    print("[DEBUG] TokenErrorTracker initialized (eval_dataset detected or track_token_errors=True)")
            
            # Epoch-level accumulator for summary logging
            self.epoch_loss_sum = 0.0
            self.epoch_accuracy_sum = 0.0
            self.epoch_step_count = 0
            self.log_every_n_epochs = 1  # Default, will be set after trainer creation
            
            if debug:
                print(f"[DEBUG] CustomTrainer initialized with loss_type: {self.loss_type}")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            """Overridden compute_loss to delegate to loss handlers."""
            loss_result = self.loss_handler(model, inputs, self)
            
            if self.logger is not None:
                predict_text = None
                label_text = None
                
                response_start = 0
                if "labels" in inputs and inputs["labels"] is not None:
                    label_ids = inputs["labels"][0]
                    response_mask = label_ids != -100
                    if response_mask.any():
                        response_start = response_mask.nonzero()[0].item()
                    
                    valid_label_ids = label_ids[response_mask]
                    if len(valid_label_ids) > 0:
                        label_text = self.processing_class.decode(
                            valid_label_ids, skip_special_tokens=False
                        )
                
                # Calculate train accuracy and predict_text from logits
                train_accuracy = None
                if loss_result.outputs is not None and hasattr(loss_result.outputs, 'logits'):
                    logits = loss_result.outputs.logits
                    labels = inputs.get("labels")

                    if logits is not None and labels is not None:
                        # Shift for next-token prediction (logits[i] predicts labels[i+1])
                        shift_logits = logits[..., :-1, :]
                        shift_labels = labels[..., 1:]
                        preds = shift_logits.argmax(dim=-1)
                        mask = shift_labels != -100
                        correct = ((preds == shift_labels) & mask).sum().item()
                        total = mask.sum().item()
                        train_accuracy = (correct / total * 100) if total > 0 else 0.0

                    # Generate predict_text for logging
                    # logits[i] predicts position i+1, so prediction for token at position
                    # response_start comes from logits[response_start - 1]. Align slicing
                    # with fallback path (pred_start = response_start - 1, range ends at seq_len - 1).
                    predicted_ids = logits.argmax(dim=-1)
                    seq_len = len(predicted_ids[0])
                    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                        seq_len = inputs["attention_mask"][0].sum().item()

                    pred_start = max(response_start - 1, 0)
                    predict_ids = predicted_ids[0][pred_start:seq_len - 1]
                    if len(predict_ids) > 0:
                        predict_text = self.processing_class.decode(
                            predict_ids, skip_special_tokens=False
                        )

                elif loss_result.predicted_ids is not None:
                    # Fallback: use pre-computed prediction info
                    # (heteroscedastic: logits argmax saved before del,
                    #  GDPO sequential: first generated sequence saved before del)
                    preds = loss_result.predicted_ids
                    mask = loss_result.response_mask

                    # accuracy: shape이 일치할 때만 계산
                    # (SFT: preds shape == shift_labels shape,
                    #  GDPO: gen_len ≠ label seq_len → skip)
                    if "labels" in inputs and inputs["labels"] is not None:
                        shift_labels = inputs["labels"][..., 1:]
                        if preds.shape == shift_labels.shape:
                            correct = ((preds == shift_labels) & mask).sum().item()
                            total = mask.sum().item()
                            train_accuracy = (correct / total * 100) if total > 0 else 0.0

                    # predict_text: shape 무관, 항상 추출
                    pred_start = max(response_start - 1, 0)
                    seq_len = len(preds[0])
                    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                        seq_len = min(inputs["attention_mask"][0].sum().item() - 1, seq_len)

                    predict_ids = preds[0][pred_start:seq_len]
                    if len(predict_ids) > 0:
                        predict_text = self.processing_class.decode(
                            predict_ids, skip_special_tokens=False
                        )
                
                current_epoch = int(self.state.epoch) + 1 if self.state.epoch is not None else 1
                
                # Accumulate epoch-level statistics (for summary logging)
                current_loss = loss_result.total_loss.item() if hasattr(loss_result.total_loss, 'item') else loss_result.total_loss
                self.epoch_loss_sum += current_loss
                self.epoch_accuracy_sum += train_accuracy if train_accuracy is not None else 0.0
                self.epoch_step_count += 1
                
                # Detailed logging: only every N epochs
                log_every_n = getattr(self, 'log_every_n_epochs', 1)
                if log_every_n == 1 or current_epoch % log_every_n == 0:
                    self.logger.log(
                        step=self.state.global_step,
                        epoch=current_epoch,
                        loss_result=loss_result,
                        predict=predict_text,
                        label=label_text,
                        accuracy=train_accuracy
                    )
            
            if return_outputs:
                return (loss_result.total_loss, loss_result.outputs)
            return loss_result.total_loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            Custom prediction_step that avoids storing full outputs.
            Uses the same loss_handler as training for consistency.
            
            Key differences from super().prediction_step():
            - Does NOT call super() which internally stores outputs/logits
            - Uses the same loss_handler as training (MC sampling for heteroscedastic)
            - Explicit memory cleanup (no backward() to trigger automatic cleanup)
            """
            model.eval()
            labels = inputs.get("labels")
            
            with torch.no_grad():
                # Use the same loss handler as training (MC sampling for heteroscedastic)
                loss_result = self.loss_handler(model, inputs, self)
                loss = loss_result.total_loss.detach()
                
                # Update token error tracker (from loss_result.outputs)
                if hasattr(self, 'token_error_tracker') and loss_result.outputs is not None:
                    logits = loss_result.outputs.logits
                    if logits is not None and labels is not None:
                        # Shift for next-token prediction (logits[i] predicts labels[i+1])
                        shift_logits = logits[..., :-1, :]
                        shift_labels = labels[..., 1:]
                        
                        predicted_ids = shift_logits.argmax(dim=-1)
                        mask = shift_labels != -100
                        self.token_error_tracker.update(predicted_ids, shift_labels, mask)
            
            # Return only loss (no logits to accumulate)
            return loss, None, labels

    # ==========================================================================
    # Training Logic
    # ==========================================================================
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

        # ── LoRA wrapping (optional, before freeze logic) ──
        # Wraps the base model with PEFT LoRA adapters.
        # log_variance_head is NOT in modules_to_save → stays in base model,
        # not packaged with LoRA adapter. Trained via requires_grad=True below.
        # After training, head weights are saved to {BASE_PATH}/heads/ for reuse.
        if args.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            lora_cfg = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            # autocast_adapter_dtype=False: peft 0.19 references float8_e8m0fnu
            # which doesn't exist in torch 2.6. Model is already bf16, so safe to skip.
            model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)

            # Sanity: ensure log_variance_head params are trainable
            for name, p in model.named_parameters():
                if "log_variance_head" in name:
                    p.requires_grad = True

            if hasattr(model, "print_trainable_parameters"):
                print("[INFO] LoRA wrapping complete. Trainable parameter summary:")
                model.print_trainable_parameters()
            print_memory_stats("After LoRA Wrap")

        # Ensure lm_head (or final layer) is trainable if we are freezing
        # Note: With tie_word_embeddings=True, lm_head does NOT appear in
        # named_parameters() (PyTorch deduplicates by data_ptr). Use
        # --train_embeddings to explicitly unfreeze embed_tokens/lm_head.
        if args.freeze_until_layer:
             for name, param in model.named_parameters():
                 if "lm_head" in name or "embed_out" in name:
                     param.requires_grad = True
                     if args.debug:
                         print(f"[DEBUG] Explicitly enabled gradients for: {name}")

        # --train_embeddings: unfreeze embed_tokens even when layers are frozen.
        # Direct tensor access is required because tie_word_embeddings=True makes
        # lm_head.weight invisible in named_parameters() (same data_ptr as embed_tokens).
        if args.train_embeddings:
            embed_param = None
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_param = model.model.embed_tokens.weight
            elif hasattr(model, 'get_input_embeddings'):
                embed_param = model.get_input_embeddings().weight
            if embed_param is not None:
                embed_param.requires_grad = True
                print(f"[INFO] --train_embeddings: embed_tokens.weight requires_grad=True "
                      f"(shape={list(embed_param.shape)})")

        # Enable input gradients to ensure the graph is connected
        # Needed for full training OR when training embeddings with frozen layers.
        # Skipped under LoRA: get_peft_model() already handles input-require-grad internally,
        # and double-registering breaks the forward hook chain.
        if not args.use_lora and (not args.freeze_until_layer or args.train_embeddings):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # Auto-download data if not found
        if args.data_path:
            from utils.data_download import ensure_data
            from utils.paths import get_data_dir
            args.data_path = ensure_data(args.data_path, data_dir=get_data_dir())

        # Load Dataset using custom module (now returns tuple)
        train_dataset, eval_dataset = dataset_module.get_dataset(args, tokenizer)
        
        # Validate dataset
        if train_dataset is None or len(train_dataset) == 0:
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
        if args.train_embeddings:
            freeze_str += "+emb"
        if args.use_lora:
            # LoRA fully replaces the freeze annotation (base is frozen by LoRA semantics)
            freeze_str = f"lora-r{args.lora_rank}-a{args.lora_alpha}"
        
        if args.save_strategy == "steps":
            save_info = f"{args.save_steps}step"
        else:
            save_info = f"{args.save_strategy}"
            
        folder_name = f"{args.model_type}-{freeze_str}-{param_str}-{args.epochs}ep-{save_info}-{date_str}-{time_str}"
        
        base_output_dir = os.path.join(get_model_dir(), "train")
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
 
        # Evaluation settings (only if val_ratio > 0)
        eval_kwargs = {}
        if args.val_ratio > 0 and eval_dataset is not None:
            eval_kwargs = {
                "eval_strategy": "epoch",  # renamed from evaluation_strategy in newer transformers
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,  # Lower loss is better
            }
            # load_best_model_at_end requires save_strategy to match eval_strategy
            # Only enable when save_strategy is not "no"
            if args.early_stopping_patience > 0 and args.save_strategy != "no":
                eval_kwargs["load_best_model_at_end"] = True
            if args.debug:
                print(f"[DEBUG] Evaluation enabled: strategy=epoch, early_stopping_patience={args.early_stopping_patience}")
                if args.save_strategy == "no" and args.early_stopping_patience > 0:
                    print(f"[DEBUG] Warning: load_best_model_at_end disabled because save_strategy='no'")

        # FSDP configuration (Fully Sharded Data Parallel)
        # Shards model weights, optimizer states, and gradients across GPUs.
        # Requires torchrun launch: CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 training.py --fsdp
        fsdp_kwargs = {}
        if args.fsdp:
            fsdp_kwargs["fsdp"] = "full_shard auto_wrap"
            fsdp_kwargs["fsdp_config"] = {
                "transformer_layer_cls_to_wrap": "Mistral3DecoderLayer",
            }
            if args.debug:
                print(f"[DEBUG] FSDP enabled: full_shard auto_wrap, wrapping Mistral3DecoderLayer")

        # Training Arguments
        # We strictly use the arguments provided by the user + dynamic precision flags.
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_dir=os.path.join(final_output_dir, 'logs'),
            remove_unused_columns=False,
            report_to="none",
            optim=args.optim,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            **eval_kwargs,
            **dtype_kwargs,
            **fsdp_kwargs
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

        # Setup callbacks
        callbacks = [CSVLoggingCallback(logger)]
        if args.val_ratio > 0 and args.early_stopping_patience > 0 and eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0  # Any improvement counts
            ))
            if args.debug:
                print(f"[DEBUG] EarlyStoppingCallback added with patience={args.early_stopping_patience}")

        trainer = CustomTrainer(
            loss_type=args.loss_type,
            logger=logger,
            debug=args.debug,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            callbacks=callbacks,
            track_token_errors=args.track_token_errors,
            pad_token_id=tokenizer.pad_token_id,
        )
        trainer.processing_class = tokenizer # Ensure trainer has tokenizer for GDPO
        trainer.ref_model = ref_model # Attach ref_model for GDPO
        trainer.log_every_n_epochs = args.log_every_n_epochs  # Set epoch logging frequency
        
        # Add EvalAccuracyCallback after trainer is created (needs trainer reference)
        if eval_dataset is not None:
            trainer.add_callback(EvalAccuracyCallback(trainer, logger))
            if args.debug:
                print("[DEBUG] EvalAccuracyCallback added for epoch-level accuracy tracking")
        
        # Add EpochSummaryCallback for epoch-level training summary logging
        epoch_summary_callback = EpochSummaryCallback(trainer, logger)
        trainer.add_callback(epoch_summary_callback)
        if args.debug:
            print(f"[DEBUG] EpochSummaryCallback added (log_every_n_epochs={args.log_every_n_epochs})")
        
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
                "reasoning_quality": args.gdpo_reward_weight_reasoning_quality,
                "temperature": args.gdpo_reward_weight_temperature,
            },
            "enable_reasoning_judge": args.gdpo_enable_reasoning_judge,
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
            # Memory Optimization
            "sequential": args.gdpo_sequential,
            # Heteroscedastic configuration (for MC sampling in uncertainty computation)
            "heteroscedastic_T": args.heteroscedastic_T,
            "heteroscedastic_sequential": args.heteroscedastic_sequential,
            # Heteroscedastic weight (legacy, for heteroscedastic_gdpo before refactor)
            "heteroscedastic_weight": args.heteroscedastic_weight,
            # Token configuration (for [THINK] tokens, etc.)
            "token_config": get_token_config(args.model_type),
        }
        
        # Attach heteroscedastic configuration to trainer
        trainer.heteroscedastic_T = args.heteroscedastic_T
        trainer.heteroscedastic_sequential = args.heteroscedastic_sequential

        print("Starting training...")
        print_memory_stats("Before Trainer Loop (Pre-Optimizer Init)")
        trainer.train()
        
        print(f"Saving fine-tuned model to {final_output_dir}")
        
        # Clear GPU memory before saving to avoid file write issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save model in base model format (compatible with vLLM without conversion)
        from model_saver import save_model_as_base_format, save_head_weights

        base_model_dir = file_config.BASE_PATH if file_config else None

        if args.use_lora:
            # (a) LoRA adapter 분리 저장
            adapter_dir = os.path.join(final_output_dir, "lora_adapter")
            model.save_pretrained(adapter_dir)
            print(f"[INFO] LoRA adapter saved to {adapter_dir}")

            # (b) Trained head weights → base model 디렉토리에 저장
            if base_model_dir:
                save_head_weights(model, base_model_dir)

            # (c) LoRA를 base에 병합 + PeftModel 해제
            # → state_dict 키가 원래 형식으로 복원 (model_saver 호환)
            model = model.merge_and_unload()
            print("[INFO] LoRA merged and unloaded for base format save")

        save_model_as_base_format(
            model=model,
            tokenizer=tokenizer,
            output_dir=final_output_dir,
            base_model_dir=base_model_dir,
            max_shard_bytes=5_000_000_000,
        )
        
        # Save token error tracking results (if enabled)
        if args.track_token_errors and hasattr(trainer, 'token_error_tracker'):
            base_name = os.path.basename(logger.output_path).replace(".csv", "")
            error_csv_path = os.path.join(get_result_dir(), f"token_errors_{base_name}.csv")
            trainer.token_error_tracker.save_csv(error_csv_path, tokenizer)

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
    
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs. Default: 1")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size per device. Default: 2")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length for tokenization. Default: 4096")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer. Default: 2e-5")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data JSON file (required)")
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Dataset module name or path (e.g. 'instruction_dataset', 'agent_dataset'). "
                             "Must match a .py file in data_loaders/ or be a file path. "
                             "The .py extension is optional.")
    
    # Validation & Early Stopping Arguments
    parser.add_argument("--val_ratio", type=float, default=0.3,
                        help="Validation set ratio (0.0-1.0). Default: 0.3")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience. Stop if val_loss doesn't improve for N evals. Default: 5")
    parser.add_argument("--stratify", type=str, default=None,
                        help="Data property name for stratified split (e.g., 'type', 'task'). None=random split")
    parser.add_argument("--track_token_errors", action="store_true",
                        help="Track and save token-level prediction errors (validation only)")
    parser.add_argument("--log_every_n_epochs", type=int, default=1,
                        help="Log detailed training samples every N epochs (default: 1 = every epoch)")
    # parser.add_argument("--output_dir", type=str, default="fine_tuned_model") # We override this now
    
    # Standard HF Save Strategy Arguments
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"], 
                        help="The checkpoint save strategy to use.")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Save checkpoint every X updates steps.")
    
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory (~1/3) at cost of ~30%% slower training.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps")

    # Optimizer & Scheduler Arguments
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer type (adamw_torch, adam_torch, sgd, adafactor, etc.). Default: adamw_torch")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam/AdamW beta1. Default: 0.9")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam/AdamW beta2. Default: 0.999")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for optimizer. Default: 0.0")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps. Overridden by warmup_ratio if > 0. Default: 0")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup ratio (0.0-1.0). Takes precedence over warmup_steps. Default: 0.0")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="LR scheduler: linear, cosine, cosine_with_restarts, polynomial, "
                             "constant, constant_with_warmup. Default: cosine")

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
    
    # GDPO Reasoning Judge Arguments
    parser.add_argument("--gdpo_enable_reasoning_judge", action="store_true",
                        help="Enable LLM-based reasoning quality reward (config in config.yaml)")
    parser.add_argument("--gdpo_reward_weight_reasoning_quality", type=float, default=1.0,
                        help="Weight for reasoning quality reward (default: 1.0)")
    
    # GDPO Memory Optimization
    parser.add_argument("--gdpo_sequential", action="store_true",
                        help="Use sequential processing for lower memory (slower but fits on smaller GPUs)")
    
    # Heteroscedastic Loss Arguments
    parser.add_argument("--heteroscedastic_T", type=int, default=3,
                        help="Number of Monte Carlo samples for heteroscedastic loss (default: 3, memory-efficient)")
    parser.add_argument("--heteroscedastic_weight", type=float, default=0.1,
                        help="Weight for heteroscedastic loss term in heteroscedastic_gdpo (default: 0.1)")
    parser.add_argument("--heteroscedastic_sequential", action="store_true",
                        help="Use sequential MC sampling for lower memory (slower)")

    # Distributed Training
    parser.add_argument("--fsdp", action="store_true",
                        help="Enable FSDP (Fully Sharded Data Parallel) for multi-GPU memory optimization. "
                             "Automatically launches with torchrun if needed.")

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

    # Model Hyperparameters (inlined from model.add_model_args for fast --help)
    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument("--model_type", type=str, default="ministral_3_3b_instruct", 
                             help="Type of model to use. Must match a python filename (e.g. ministral_3_3b_instruct)")
    model_group.add_argument("--model_path", type=str, default=None, 
                             help="Explicit path to load model weights from (overrides default resolved path)")
    model_group.add_argument("--load_until_layer", type=str, default=None, 
                             help="Load weights only up to this layer name (inclusive)")
    model_group.add_argument("--freeze_until_layer", type=str, default=None,
                             help="Freeze gradients up to this layer name (inclusive). "
                                  "Use layer number (e.g., '24') or '-1' to freeze ALL layers "
                                  "(useful for training only new parameters like mHC)")
    model_group.add_argument("--train_embeddings", action="store_true", default=False,
                             help="Train embed_tokens even when layers are frozen. "
                                  "Handles tie_word_embeddings by direct tensor access.")
    model_group.add_argument("--base_model_path", type=str, default=None,
                             help="Path to base model for partial weight loading. "
                                  "Loads matching weights from base model into target model. "
                                  "Non-matching parameters (e.g., mHC layers) keep their initial values.")
    # Model config overrides
    model_group.add_argument("--hidden_size", type=int, default=None)
    model_group.add_argument("--num_hidden_layers", type=int, default=None)
    model_group.add_argument("--num_attention_heads", type=int, default=None)
    model_group.add_argument("--num_key_value_heads", type=int, default=None)
    model_group.add_argument("--intermediate_size", type=int, default=None)
    model_group.add_argument("--vocab_size", type=int, default=None)
    model_group.add_argument("--max_position_embeddings", type=int, default=None)
    model_group.add_argument("--rope_theta", type=float, default=None)
    model_group.add_argument("--quantization", type=str, default="fp8",
                             choices=["fp8", "bf16", "fp16", "fp32"],
                             help="Model weight precision. "
                                  "fp8=FP8 storage→BF16 compute (default), "
                                  "bf16/fp16/fp32=pure precision")

    # ── LoRA Hyperparameters ──
    lora_group = parser.add_argument_group("LoRA Hyperparameters")
    lora_group.add_argument("--use_lora", action="store_true",
                            help="Wrap the base model with LoRA adapters before training. "
                                 "log_variance_head stays fully trainable via modules_to_save.")
    lora_group.add_argument("--lora_rank", type=int, default=16,
                            help="LoRA rank r (default: 16).")
    lora_group.add_argument("--lora_alpha", type=int, default=32,
                            help="LoRA alpha (scaling = alpha/r). Default: 32.")
    lora_group.add_argument("--lora_dropout", type=float, default=0.05,
                            help="LoRA dropout (default: 0.05).")
    lora_group.add_argument("--lora_target_modules", type=str,
                            default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                            help="Comma-separated module name suffixes to inject LoRA into. "
                                 "Default: all attention + MLP projections.")
    lora_group.add_argument("--lora_bias", type=str, default="none",
                            choices=["none", "all", "lora_only"],
                            help="PEFT bias setting (default: 'none').")

    args = parser.parse_args()

    # Auto-relaunch with torchrun when --fsdp is specified but not already under torchrun
    if args.fsdp and "LOCAL_RANK" not in os.environ:
        from utils.gpu_config import configure_gpu
        gpu_ids = configure_gpu()  # sets CUDA_VISIBLE_DEVICES
        num_gpus = len(gpu_ids.split(",")) if gpu_ids else 1
        if num_gpus < 2:
            print(f"[WARNING] --fsdp requires 2+ GPUs. Current: {gpu_ids}. Proceeding single-GPU.")
        else:
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   f"--nproc_per_node={num_gpus}"] + sys.argv
            print(f"[FSDP] Auto-launching with torchrun: {' '.join(cmd)}")
            os.execvp(sys.executable, cmd)
            # execvp replaces process, never reaches here

    run_training(args)
