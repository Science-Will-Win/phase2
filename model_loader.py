
import torch
import os
import json
from safetensors.torch import load_file
from safetensors import safe_open

def _load_params_json(weight_path, params_filename="params.json"):
    params_path = os.path.join(weight_path, params_filename)
    if not os.path.exists(params_path):
        return None
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read params.json ({e})")
        return None


def _is_mistral_params(params):
    if not isinstance(params, dict):
        return False
    return any(
        key in params
        for key in ("tied_embeddings", "llama_4_scaling", "vision_encoder")
    )


def load_model_weights(model, weight_path, load_until=None):
    """
    Loads weights into the model from weight_path.
    Supports:
    - Single .bin or .pt file
    - Single .safetensors file
    - Directory with pytorch_model.bin
    - Directory with model.safetensors
    - Directory with sharded model.safetensors.index.json
    - Partial loading via load_until
    """
    
    if not os.path.exists(weight_path):
        print(f"[WARNING] Weight path {weight_path} does not exist.")
        return model

    print(f"[DEBUG] Loading weights from {weight_path}...")
    state_dict = {}
    sharded_files = []
    files_to_load = []
    selected_source = None
    
    # 1. Determine file structure
    if os.path.isfile(weight_path):
        files_to_load = [weight_path]
        selected_source = os.path.basename(weight_path)
    elif os.path.isdir(weight_path):
        params = _load_params_json(weight_path)
        prefer_consolidated = _is_mistral_params(params)
        # Check for index (sharded)
        index_file = os.path.join(weight_path, "model.safetensors.index.json")
        bin_index_file = os.path.join(weight_path, "pytorch_model.bin.index.json")
        consolidated_index_file = os.path.join(weight_path, "consolidated.safetensors.index.json")
        consolidated_file = os.path.join(weight_path, "consolidated.safetensors")
        
        if os.path.exists(index_file):
            print("[DEBUG] Detected sharded safetensors model.")
            with open(index_file, "r") as f:
                index_data = json.load(f)
            # Get unique filenames from the weight_map
            sharded_files = sorted(list(set(index_data["weight_map"].values())))
            sharded_files = [os.path.join(weight_path, f) for f in sharded_files]
            selected_source = os.path.basename(index_file)
        elif prefer_consolidated and os.path.exists(consolidated_index_file):
            print("[DEBUG] Detected sharded consolidated safetensors model.")
            with open(consolidated_index_file, "r") as f:
                index_data = json.load(f)
            sharded_files = sorted(list(set(index_data["weight_map"].values())))
            sharded_files = [os.path.join(weight_path, f) for f in sharded_files]
            selected_source = os.path.basename(consolidated_index_file)
        elif os.path.exists(bin_index_file):
            print("[DEBUG] Detected sharded pytorch model.")
            with open(bin_index_file, "r") as f:
                index_data = json.load(f)
            sharded_files = sorted(list(set(index_data["weight_map"].values())))
            sharded_files = [os.path.join(weight_path, f) for f in sharded_files]
            selected_source = os.path.basename(bin_index_file)
        else:
            # Check for non-sharded
            safe_file = os.path.join(weight_path, "model.safetensors")
            bin_file = os.path.join(weight_path, "pytorch_model.bin")
            if os.path.exists(safe_file):
                files_to_load = [safe_file]
                selected_source = os.path.basename(safe_file)
            elif prefer_consolidated and os.path.exists(consolidated_file):
                files_to_load = [consolidated_file]
                selected_source = os.path.basename(consolidated_file)
            elif os.path.exists(bin_file):
                files_to_load = [bin_file]
                selected_source = os.path.basename(bin_file)
            else:
                # Check for any .safetensors
                candidates = [f for f in os.listdir(weight_path) if f.endswith(".safetensors")]
                if candidates:
                    # heuristic: load all? or just model? usually split requires index.
                    # If multiple without index, likely parts.
                    files_to_load = [os.path.join(weight_path, f) for f in candidates]
                    selected_source = f"{len(candidates)} safetensors files"
                else:
                    print("[DEBUG] Could not find weights file (bin/safetensors) in directory.")
                    return model
    else:
        return model

    if sharded_files:
        files_to_load = sharded_files
    
    if selected_source:
        print(f"[DEBUG] Selected weight source: {selected_source}")

    # 2. Logic for Partial Loading (Pre-calculation of target keys)
    # We identify which keys we want to load.
    all_params = [n for n, _ in model.named_parameters()]
    target_param_names = set(all_params)
    
    if load_until:
        last_match_idx = -1
        for i, name in enumerate(all_params):
            if load_until in name:
                last_match_idx = i
        
        if last_match_idx != -1:
            print(f"[DEBUG] Partial loading enabled. Stopping at layer match: {load_until}")
            target_param_names = set(all_params[:last_match_idx+1])
        else:
            print(f"[WARNING] Layer '{load_until}' not found. Loading full model.")

    # 3. Iterative Loading (Memory Efficient-ish)
    # Instead of loading one huge state_dict, we load file by file and update model.
    # Note: reset/load into model progressively.
    
    missing_keys = set(target_param_names)
    unexpected_keys = []
    
    # Detect device from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"[DEBUG] Loading weights to {device} in {dtype}...")

    for file_path in files_to_load:
        print(f"[DEBUG] Processing {os.path.basename(file_path)}...")
        try:
            # Load to device directly to save System RAM
            if file_path.endswith(".safetensors"):
                loaded_dict = load_file(file_path, device=str(device))
            else:
                try:
                    loaded_dict = torch.load(file_path, map_location=device, weights_only=True)
                except Exception as e:
                    print(f"[WARNING] weights_only=True failed ({e}), falling back to unsafe load.")
                    loaded_dict = torch.load(file_path, map_location=device)
            
            # Normalize keys: detect model format and remap accordingly
            # Two possible formats:
            # - Custom model: model.embed_tokens.weight
            # - Official transformers: model.language_model.embed_tokens.weight
            
            # Detect which format the model uses
            uses_official_format = any("model.language_model." in p for p in target_param_names)
            
            normalized_dict = {}
            prefix_remapped_count = 0
            vision_remapped_count = 0
            
            for k, v in loaded_dict.items():
                new_key = k
                
                # Remap language_model keys based on model format
                if new_key.startswith("language_model."):
                    if uses_official_format:
                        # Official transformers: language_model.model.X -> model.language_model.X
                        # e.g., language_model.model.embed_tokens -> model.language_model.embed_tokens
                        new_key = "model." + new_key.replace("language_model.model.", "language_model.")
                    else:
                        # Custom model: language_model.model.X -> model.X
                        new_key = new_key[len("language_model."):]
                    prefix_remapped_count += 1
                
                # Remap vision_tower keys to match model architecture
                if "vision_tower" in new_key:
                    original_key = new_key
                    
                    if uses_official_format:
                        # Official: vision_tower.X -> model.vision_tower.X
                        if new_key.startswith("vision_tower."):
                            new_key = "model." + new_key
                    
                    # Common remappings for vision tower structure
                    # vision_tower.patch_conv.weight -> vision_tower.patch_embedding.proj.weight (or patch_conv for official)
                    if not uses_official_format:
                        new_key = new_key.replace("vision_tower.patch_conv.", "vision_tower.patch_embedding.proj.")
                    
                    # vision_tower.transformer.layers.X -> vision_tower.layers.X
                    new_key = new_key.replace(".transformer.layers.", ".layers.")
                    
                    # attention -> self_attn (only for custom model)
                    if not uses_official_format:
                        new_key = new_key.replace(".attention.", ".self_attn.")
                        new_key = new_key.replace(".attention_norm.", ".input_layernorm.")
                        new_key = new_key.replace(".feed_forward.", ".mlp.")
                        new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
                    
                    if new_key != original_key:
                        vision_remapped_count += 1
                
                # Handle multi_modal_projector for official format
                if uses_official_format and new_key.startswith("multi_modal_projector."):
                    new_key = "model." + new_key
                
                normalized_dict[new_key] = v
            
            if prefix_remapped_count > 0:
                format_name = "official transformers" if uses_official_format else "custom model"
                print(f"[DEBUG]   Remapped {prefix_remapped_count} keys for {format_name} format")
            if vision_remapped_count > 0:
                print(f"[DEBUG]   Remapped {vision_remapped_count} vision_tower keys")
            loaded_dict = normalized_dict

            unmapped_language_keys = [
                k for k in loaded_dict.keys()
                if k.startswith("language_model.") or k.startswith("model.language_model.")
            ]
            if unmapped_language_keys:
                print(f"[WARNING] {len(unmapped_language_keys)} language_model keys did not map to model params")
                for key in unmapped_language_keys[:3]:
                    print(f"[DEBUG]     - {key}")
                
            # Filter dict (skip quantization metadata keys defensively)
            skip_suffixes = (".activation_scale", ".weight_scale_inv", ".qscale_weight", ".qscale_act")
            filtered_dict = {
                k: v for k, v in loaded_dict.items()
                if k in target_param_names and not k.endswith(skip_suffixes)
            }
            
            # FP8 Dequantization: Apply weight_scale_inv if present
            # FP8 weights are stored as: fp8_weight = original_weight / scale
            # To restore: original_weight = fp8_weight * scale_inv (where scale_inv = 1/scale... wait, naming is confusing)
            # Actually based on the values, it seems: original_weight = fp8_weight * weight_scale_inv
            fp8_dequant_count = 0
            for k, v in list(filtered_dict.items()):
                scale_key = k.replace(".weight", ".weight_scale_inv")
                if scale_key in loaded_dict:
                    scale_inv = loaded_dict[scale_key]
                    # Dequantize using float32 for stability, then cast to target dtype
                    dequant = v.to(dtype=torch.float32) * scale_inv.to(dtype=torch.float32)
                    filtered_dict[k] = dequant.to(dtype=dtype)
                    fp8_dequant_count += 1
                elif v.dtype != dtype:
                    # Regular dtype casting for non-FP8 weights
                    filtered_dict[k] = v.to(dtype=dtype)
            
            if fp8_dequant_count > 0:
                print(f"[DEBUG]   Dequantized {fp8_dequant_count} FP8 weights")

            # Load into model
            try:
                msg = model.load_state_dict(filtered_dict, strict=False)
                
                # Track what we loaded
                loaded_keys = set(filtered_dict.keys())
                missing_keys -= loaded_keys
                unexpected_keys.extend(msg.unexpected_keys)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"\n[DEBUG][ERROR] Shape mismatch detected while loading {os.path.basename(file_path)}.")
                    print(f"[DEBUG] Details: {e}")
                    return model 
                else:
                    raise e
            
            # Aggressive cleanup
            del loaded_dict
            del filtered_dict
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[DEBUG] Error processing file {file_path}: {e}")

    # Final Verification
    # ...
    
    # Force model to target dtype one last time to be completely sure
    model.to(dtype=dtype)
    
    # Tie weights (e.g., lm_head.weight = embed_tokens.weight) if the model supports it
    # This is critical for models with tie_word_embeddings=True
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
        print("[DEBUG] Tied model weights (lm_head ↔ embed_tokens)")
    
    tie_word_embeddings = False
    if hasattr(model, "config"):
        tie_word_embeddings = bool(
            getattr(model.config, "tie_word_embeddings", False)
            or getattr(model.config, "tied_embeddings", False)
            or getattr(getattr(model.config, "text_config", None), "tie_word_embeddings", False)
        )
    if tie_word_embeddings and missing_keys:
        missing_keys = {k for k in missing_keys if not k.endswith("lm_head.weight")}

    optional_missing = {"vision_tower.norm.weight", "model.vision_tower.norm.weight"}
    if missing_keys:
        missing_keys = {k for k in missing_keys if k not in optional_missing}
    
    if missing_keys:
        print("\n" + "="*30)
        print("[DEBUG][Partial Loading Report]")
        print(f"[DEBUG] Missing parameters: {len(missing_keys)}")
        print("[DEBUG] The following parameters were NOT found in the checkpoint.")
        print("[DEBUG] They retain their initial random weights (e.g., custom layers or matrices you added):")
        
        missing_list = sorted(list(missing_keys))
        for i, key in enumerate(missing_list):
            if i < 10:
                print(f"[DEBUG]   - {key}")
            else:
                print(f"[DEBUG]   ... and {len(missing_list) - 10} more.")
                break
        print("="*30 + "\n")
    
    # Load trained head weights from heads/ subdirectory (if available)
    # Extensible pattern: each .safetensors file in heads/ is auto-loaded,
    # overriding zero-initialized weights (e.g., log_variance_head).
    if os.path.isdir(weight_path):
        heads_dir = os.path.join(weight_path, "heads")
        if os.path.isdir(heads_dir):
            for head_file in sorted(os.listdir(heads_dir)):
                if head_file.endswith(".safetensors"):
                    head_path = os.path.join(heads_dir, head_file)
                    head_name = head_file.replace(".safetensors", "")
                    head_weights = load_file(head_path, device=str(device))
                    head_weights = {k: v.to(dtype=dtype) for k, v in head_weights.items()}
                    msg = model.load_state_dict(head_weights, strict=False)
                    loaded_count = len(head_weights) - len(msg.unexpected_keys)
                    print(f"[DEBUG] Loaded head '{head_name}': {loaded_count} weights from {head_file}")
                    missing_keys -= set(head_weights.keys())

    # Release temporary memory from FP8 dequantization and loading process
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[DEBUG] VRAM after loading: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

    return model


def load_weights_from_base(target_model, base_model_path, device="cuda", dtype=torch.bfloat16):
    """
    Load weights from a base model into target model where parameter names match.
    Non-matching parameters (e.g., new mHC layers) keep their initial random values.
    
    This is useful for transfer learning scenarios where the target model extends
    the base model with additional layers/modules.
    
    Args:
        target_model: The model to load weights into
        base_model_path: Path to base model directory containing weights
        device: Device to load weights to
        dtype: Data type for weights
        
    Returns:
        Tuple of (model, matched_keys) where matched_keys are the parameters that were loaded
    """
    import os
    from safetensors import safe_open
    
    print(f"\n[DEBUG][Partial Weight Loading] Loading from base model: {base_model_path}")
    
    # Find weight file
    weight_file = None
    if os.path.isdir(base_model_path):
        for fname in ["model.safetensors", "pytorch_model.bin"]:
            fpath = os.path.join(base_model_path, fname)
            if os.path.exists(fpath):
                weight_file = fpath
                break
    elif os.path.isfile(base_model_path):
        weight_file = base_model_path
    
    if weight_file is None:
        print(f"[WARNING] No weight file found in {base_model_path}. Skipping partial loading.")
        return target_model, []
    
    print(f"[DEBUG] Loading base weights from: {weight_file}")
    
    # Load base model state dict
    base_state_dict = {}
    if weight_file.endswith(".safetensors"):
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_state_dict[key] = f.get_tensor(key)
    else:
        base_state_dict = torch.load(weight_file, map_location="cpu")
    
    # Get target model state dict
    target_state_dict = target_model.state_dict()
    
    # Match and load weights
    matched_keys = []
    mismatched_shape = []
    
    for key in target_state_dict.keys():
        if key in base_state_dict:
            if target_state_dict[key].shape == base_state_dict[key].shape:
                target_state_dict[key] = base_state_dict[key].to(dtype=dtype)
                matched_keys.append(key)
            else:
                mismatched_shape.append(key)
    
    # Load matched weights
    target_model.load_state_dict(target_state_dict, strict=False)
    
    # Report
    new_params = [k for k in target_state_dict.keys() if k not in matched_keys]
    
    print(f"[DEBUG]   Matched and loaded: {len(matched_keys)} parameters")
    print(f"[DEBUG]   New parameters (not in base): {len(new_params)}")
    if mismatched_shape:
        print(f"[DEBUG]   Shape mismatch (skipped): {len(mismatched_shape)}")
    
    if new_params and len(new_params) <= 20:
        print("[DEBUG]   New parameters:")
        for k in new_params:
            print(f"[DEBUG]     - {k}")
    elif new_params:
        print(f"[DEBUG]   New parameters (showing first 10 of {len(new_params)}):")
        for k in new_params[:10]:
            print(f"[DEBUG]     - {k}")
    
    return target_model, matched_keys


def freeze_model_weights(model, freeze_until=None):
    """
    Freezes model weights up to (and including) the layer matching `freeze_until`.
    If freeze_until is None, does nothing.
    If freeze_until is "-1", freezes ALL parameters.
    If freeze_until matches a layer name, sets requires_grad = False for those parameters.
    """
    if not freeze_until:
        return model
    
    # Special case: -1 means freeze ALL parameters
    if freeze_until == "-1":
        print("[DEBUG] Freezing ALL model weights (freeze_until=-1)")
        frozen_count = 0
        for name, param in model.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"[DEBUG] Frozen {frozen_count} parameters (all layers).")
        return model
        
    print(f"[DEBUG] Freezing model weights up to: {freeze_until}")
    
    all_params = [n for n, _ in model.named_parameters()]
    target_param_names = set()
    
    # 1. Identify cutoff index
    last_match_idx = -1
    for i, name in enumerate(all_params):
        if freeze_until in name:
            last_match_idx = i
            
    if last_match_idx == -1:
        print(f"[WARNING] Layer '{freeze_until}' not found for freezing. No weights frozen.")
        return model
    
    # 2. Define parameters to freeze
    params_to_freeze = set(all_params[:last_match_idx+1])
    
    # 3. Apply freeze
    frozen_count = 0
    for name, param in model.named_parameters():
        if name in params_to_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True # explicit
            
    print(f"[DEBUG] Frozen {frozen_count} parameters (up to layer {freeze_until}).")
    return model
