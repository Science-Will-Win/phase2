
import os
import torch
import torch.nn as nn
import argparse
import importlib
import inspect
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel, PretrainedConfig
# from ministral_3b import MinistralForCausalLM, MinistralConfig (Removed explicit import)
import model_loader
from utils import get_file_config
from utils.model_utils import maybe_wrap_dataparallel, get_model_device, get_unwrapped_model


# Dictionary to store default configurations
# Keys must match the model_type (filename)
MODEL_CONFIGS = {
    "ministral_3_3b_instruct": {
        "hidden_size": 3072,
        "num_hidden_layers": 26,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9216,
        "vocab_size": 131072,
        "max_position_embeddings": 262144,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
    "ministral_3_3b_reasoning": {
        "hidden_size": 3072,
        "num_hidden_layers": 26,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9216,
        "vocab_size": 131072,
        "max_position_embeddings": 262144,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
}

def download_and_setup_model(model_type, dest_dir=None):
    """
    Downloads the model from HF if not present.
    Also sets up tokenizer files in tokenizer/{model_type}.
    """
    file_config = get_file_config(model_type)
    if not file_config or not hasattr(file_config, 'HF_REPO_ID'):
        # No FileConfig or HF_REPO_ID available, skip auto-download
        return
    
    repo_id = file_config.HF_REPO_ID

    if dest_dir is None:
        dest_dir = file_config.BASE_PATH if hasattr(file_config, 'BASE_PATH') else os.path.join("model", model_type)
    
    model_dir = dest_dir
    # print(f"Checking model directory: {model_dir}")
    
    # Check if model dir exists and contains weights
    has_weights = False
    if os.path.exists(model_dir):
        if any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(model_dir)):
            has_weights = True
            
    if not has_weights:
        print(f"[DEBUG] Model weights not found at {model_dir}. Downloading {repo_id}...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
            print("[DEBUG] Download completed.")
        except Exception as e:
            print(f"[DEBUG] Failed to download model: {e}")
            return

    # Tokenizer is in the model directory, so no extra setup needed.
    print(f"[DEBUG] Model and tokenizer ready in {model_dir}")

def add_model_args(parser: argparse.ArgumentParser):
    """
    Adds model-specific arguments to the parser.
    """
    group = parser.add_argument_group("Model Hyperparameters")
    
    group.add_argument("--model_type", type=str, default="ministral_3_3b_instruct", 
                       help="Type of model to use. Must match a python filename (e.g. ministral_3_3b_instruct)")
    # group.add_argument("--model_base_dir", type=str, default="model", 
    #                    help="Base directory for models") # Removed
    group.add_argument("--model_path", type=str, default=None, 
                       help="Explicit path to load model weights from (overrides default resolved path)")
    group.add_argument("--load_until_layer", type=str, default=None, 
                       help="Load weights only up to this layer name (inclusive)")
    group.add_argument("--freeze_until_layer", type=str, default=None, 
                       help="Freeze gradients up to this layer name (inclusive). "
                            "Use layer number (e.g., '24') or '-1' to freeze ALL layers "
                            "(useful for training only new parameters like mHC)")
    group.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base model for partial weight loading. "
                            "Loads matching weights from base model into target model. "
                            "Non-matching parameters (e.g., mHC layers) keep their initial values.")
    
    # Overrides
    group.add_argument("--hidden_size", type=int, default=None)
    group.add_argument("--num_hidden_layers", type=int, default=None)
    group.add_argument("--num_attention_heads", type=int, default=None)
    group.add_argument("--num_key_value_heads", type=int, default=None)
    group.add_argument("--intermediate_size", type=int, default=None)
    group.add_argument("--vocab_size", type=int, default=None)
    group.add_argument("--max_position_embeddings", type=int, default=None)
    group.add_argument("--rope_theta", type=float, default=None)
    
    group.add_argument("--quantization", type=str, default="fp8",
                       choices=["fp8", "bf16", "fp16", "fp32"],
                       help="Model weight precision. "
                            "fp8=FP8 storage→BF16 compute (default), "
                            "bf16/fp16/fp32=pure precision")
    
    return parser

def get_model(args):
    """
    Initializes and returns the model.
    Dynamically imports the module named args.model_type.
    """
    model_type = args.model_type
    
    try:
        model_module = importlib.import_module(f"architectures.{model_type}")
        print(f"[DEBUG] Successfully imported module: {model_type} from {model_module.__file__}")
    except ImportError:
        raise ImportError(f"Could not import module '{model_type}'. Please ensure architectures/{model_type}.py exists.")

    # Find Config and Model classes in the module
    config_class = None
    model_class = None
    
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if issubclass(obj, PretrainedConfig) and obj is not PretrainedConfig:
             # Heuristic: usually named *Config
             if "Config" in name:
                 # Prefer the main config (Mistral3Config or MistralConfig) over sub-configs
                 if "Mistral" in name and "Vision" not in name and "Text" not in name:
                     config_class = obj
                 elif config_class is None:
                     config_class = obj
        if issubclass(obj, PreTrainedModel) and obj is not PreTrainedModel:
             # Heuristic: usually named *ForCausalLM or *Model
             # We prefer ForCausalLM for generation
             if "ForCausalLM" in name:
                 model_class = obj
             elif model_class is None and "Model" in name:
                 model_class = obj
    
    if config_class is None:
        raise ValueError(f"Could not find a subclass of PretrainedConfig in {model_type}.py")
    if model_class is None:
        raise ValueError(f"Could not find a subclass of PreTrainedModel in {model_type}.py")

    print(f"[DEBUG] Found Config Class: {config_class.__name__}")
    print(f"[DEBUG] Found Model Class: {model_class.__name__}")

    # Determine loading path
    if args.model_path:
        load_path = args.model_path
    else:
        # Use FileConfig.BASE_PATH if available
        file_config = get_file_config(model_type)
        if file_config and hasattr(file_config, 'BASE_PATH'):
            load_path = file_config.BASE_PATH
        else:
            load_path = os.path.join("model", model_type)
        # Check and auto-download if a repo mapping exists
        download_and_setup_model(model_type)
        
    # 1. Try to load config from load_path/config.json if it exists.
    config = None
    config_path = os.path.join(load_path, "config.json")
    if os.path.exists(config_path):
        print(f"[DEBUG] Loading configuration from {config_path}...")
        try:
            config = config_class.from_pretrained(load_path)
        except Exception as e:
            print(f"[DEBUG] Error loading config from {config_path}: {e}")
    
    # 2. If no config found in file, use presets or class defaults
    if config is None:
        # Check if we have a hardcoded preset for this model_type
        if model_type in MODEL_CONFIGS:
            print(f"[DEBUG] Using hardcoded presets for {model_type}.")
            config = config_class(**MODEL_CONFIGS[model_type])
        else:
            print(f"[DEBUG] No config.json or presets found for {model_type}. Using architecture defaults from {config_class.__name__}.")
            config = config_class()
    
    # 3. Apply Overrides from CLI
    if args.hidden_size is not None: config.hidden_size = args.hidden_size
    if args.num_hidden_layers is not None: config.num_hidden_layers = args.num_hidden_layers
    if args.num_attention_heads is not None: config.num_attention_heads = args.num_attention_heads
    if args.num_key_value_heads is not None: config.num_key_value_heads = args.num_key_value_heads
    if args.intermediate_size is not None: config.intermediate_size = args.intermediate_size
    if args.vocab_size is not None: config.vocab_size = args.vocab_size
    if args.max_position_embeddings is not None: config.max_position_embeddings = args.max_position_embeddings
    if args.rope_theta is not None: config.rope_theta = args.rope_theta
    
    print(f"[DEBUG] Final Configuration: {config}")

    # Determine quantization mode and target dtype
    quantization = getattr(args, 'quantization', 'fp8')
    
    # Map quantization to target dtype
    dtype_map = {
        "fp8": torch.bfloat16,   # FP8 storage → BF16 compute
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(quantization, torch.bfloat16)
    
    print(f"[DEBUG] Quantization mode: {quantization}, Target dtype: {target_dtype}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model directly on GPU to avoid CPU RAM spike
    print(f"[DEBUG] Initializing {model_type} with dtype: {target_dtype}")
    
    try:
        with torch.device(device):
            model = model_class(config)
    except Exception as e:
        print(f"[WARNING] Failed to init on {device}: {e}. Falling back to CPU.")
        model = model_class(config)

    # Cast to target dtype
    if target_dtype:
        model.to(dtype=target_dtype)
        
    print(f"[DEBUG] Model initialized on {device} in {target_dtype if target_dtype else 'float32'}")
    
    # Load Generation Config if exists
    try:
        from transformers import GenerationConfig
        file_config = get_file_config(model_type)
        gen_config_filename = file_config.GENERATION_CONFIG if file_config else "generation_config.json"
        gen_config_path = os.path.join(load_path, gen_config_filename)
        if os.path.exists(gen_config_path):
            print(f"[DEBUG] Loading generation config from {gen_config_path}...")
            model.generation_config = GenerationConfig.from_pretrained(load_path)
    except Exception as e:
        print(f"[DEBUG] Failed to load generation config: {e}")

    # Check if load_path exists and has weights
    if os.path.exists(load_path):
        is_weight_folder = False
        if os.path.isdir(load_path):
            if any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(load_path)):
                is_weight_folder = True
        elif os.path.isfile(load_path):
            is_weight_folder = True
            
        if is_weight_folder:
            print(f"[DEBUG] Found model weights at {load_path}. Loading...")
            model = model_loader.load_model_weights(model, load_path, load_until=args.load_until_layer)
        else:
            print("\n" + "!"*50)
            print(f"[WARNING] Directory {load_path} exists but NO WEIGHTS found.")
            print("[DEBUG] Using RANDOM INITIALIZATION for the model.")
            print("[DEBUG] This is expected for training from scratch, but NOT for fine-tuning.")
            print("!"*50 + "\n")
    else:
        print("\n" + "!"*50)
        print(f"[WARNING] No weights found at {load_path}.")
        print("[DEBUG] Using RANDOM INITIALIZATION for the model.")
        print("!"*50 + "\n")
    
    # Load weights from base model if specified (for partial loading, e.g., mHC training)
    if args.base_model_path:
        model, matched_keys = model_loader.load_weights_from_base(
            model, 
            args.base_model_path, 
            device=device, 
            dtype=target_dtype
        )
    
    # Freeze layers if requested
    if args.freeze_until_layer:
        model = model_loader.freeze_model_weights(model, freeze_until=args.freeze_until_layer)
    
    # Final Ensure device (just in case)
    model.to(device)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()
    model = get_model(args)
