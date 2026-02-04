"""
Mistral3 Vision-Language Model Implementation
Based on mistralai/Ministral-3-3B-Reasoning-2512 architecture
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers import AutoConfig

# ============================================================================
# KV Cache Configuration (set by inference.py at runtime)
# ============================================================================
KV_CACHE_DTYPE = "bf16"  # Default: bf16 for Reasoning model (BF16 weights), options: bf16, fp16, fp8

def get_kv_cache_dtype():
    """Get torch dtype for KV cache based on global setting."""
    global KV_CACHE_DTYPE
    if KV_CACHE_DTYPE == "fp8":
        try:
            return torch.float8_e4m3fn
        except AttributeError:
            return torch.bfloat16
    elif KV_CACHE_DTYPE == "fp16":
        return torch.float16
    else:  # bf16 or default
        return torch.bfloat16

def convert_to_kv_cache_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to KV cache dtype for storage."""
    target_dtype = get_kv_cache_dtype()
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(target_dtype)

def convert_from_kv_cache_dtype(tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor from KV cache dtype back to original dtype for computation."""
    if tensor.dtype == original_dtype:
        return tensor
    return tensor.to(original_dtype)

# ============================================================================
# Token Configuration for Reasoning
# ============================================================================

class Ministral3TokenConfig:
    """Reasoning token configuration"""
    
    # Required tokens
    THINK_START: str = "[THINK]"
    THINK_END: str = "[/THINK]"
    
    # Optional tokens (None if not used)
    ANSWER_START: str | None = "<SPECIAL_36>"
    ANSWER_END: str | None = "<SPECIAL_37>"


class Ministral3FileConfig:
    """Model-related file paths/roles configuration"""
    
    # Base path
    BASE_PATH: str = "model/ministral_3_3b_reasoning"
    
    # For download
    HF_REPO_ID: str = "mistralai/Ministral-3-3B-Reasoning-2512"
    
    # === Model config ===
    PARAMS: str = "params.json"
    GENERATION_CONFIG: str = "generation_config.json"
    
    # === Tokenizer related ===
    TOKENIZER: str = "tokenizer.json"
    TOKENIZER_CONFIG: str = "tokenizer_config.json"
    SPECIAL_TOKENS_MAP: str = "special_tokens_map.json"
    TEKKEN: str = "tekken.json"
    
    # === Processor ===
    PROCESSOR_CONFIG: str = "processor_config.json"
    
    # === Chat/Prompt ===
    CHAT_TEMPLATE: str = "chat_template.jinja"
    SYSTEM_PROMPT: str = "SYSTEM_PROMPT.txt"


# ============================================================================
# Configuration Classes
# ============================================================================

class PixtralVisionConfig(PretrainedConfig):
    """Configuration for the Pixtral Vision Tower"""
    model_type = "pixtral"
    
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        head_dim=64,
        num_channels=3,
        image_size=1540,
        patch_size=14,
        hidden_act="silu",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta


class Ministral3TextConfig(PretrainedConfig):
    """Configuration for the Text (Language) model"""
    model_type = "ministral3"
    
    def __init__(
        self,
        vocab_size=131072,
        hidden_size=3072,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        sliding_window=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window


class Mistral3Config(PretrainedConfig):
    """Main configuration for the Mistral3 Vision-Language Model"""
    model_type = "mistral3"
    
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if isinstance(text_config, dict):
            self.text_config = Ministral3TextConfig(**text_config)
        elif text_config is None:
             self.text_config = Ministral3TextConfig()
        else:
            self.text_config = text_config
            
        if isinstance(vision_config, dict):
            self.vision_config = PixtralVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = PixtralVisionConfig()
        else:
            self.vision_config = vision_config
        
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size
        self.vision_feature_layer = vision_feature_layer
        
        # Expose commonly used text config attributes at top level for easy access
        if self.text_config is not None:
             for key, value in self.text_config.to_dict().items():
                 if not hasattr(self, key):
                     setattr(self, key, value)
        else:
             # Basic defaults only if absolutely necessary for initialization failure cases
             self.hidden_size = getattr(self, "hidden_size", 3072)
             self.num_hidden_layers = getattr(self, "num_hidden_layers", 26)


# ============================================================================
# RMSNorm
# ============================================================================

class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# Rotary Position Embedding
# ============================================================================
# Use official transformers RoPE implementation which supports YARN and other types

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding as _OfficialRoPE
    HAS_OFFICIAL_ROPE = True
except ImportError:
    HAS_OFFICIAL_ROPE = False

class Mistral3RotaryEmbedding(nn.Module):
    """RoPE with YARN support via transformers official implementation."""
    
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        
        # Convert rope_parameters to rope_scaling format for transformers compatibility
        rope_params = getattr(config, 'rope_parameters', None) or {}
        if isinstance(rope_params, dict):
            self.rope_type = rope_params.get('rope_type', rope_params.get('type', 'default'))
        else:
            self.rope_type = 'default'
        
        if HAS_OFFICIAL_ROPE and self.rope_type != 'default':
            # Use official implementation which supports YARN
            self._use_official = True
            
            # Create a compatible config with rope_scaling
            # transformers expects config.rope_scaling, not rope_parameters
            if not hasattr(config, 'rope_scaling') and isinstance(rope_params, dict):
                config.rope_scaling = rope_params
            
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            
            try:
                self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
                if self.rope_init_fn is None:
                    # Fallback to linear if rope_type not found
                    self.rope_init_fn = ROPE_INIT_FUNCTIONS.get('linear')
                inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
                self.register_buffer("inv_freq", inv_freq, persistent=False)
            except Exception as e:
                print(f"Warning: Official RoPE init failed ({e}), falling back to basic implementation")
                self._use_official = False
                self._init_basic_rope(config, device)
        else:
            # Fallback to basic implementation
            self._use_official = False
            self._init_basic_rope(config, device)
    
    def _init_basic_rope(self, config, device):
        """Basic RoPE without YARN scaling."""
        self.attention_scaling = 1.0
        dim = config.head_dim
        base = getattr(config, 'rope_theta', 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# MLP
# ============================================================================

class Mistral3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Text Attention
# ============================================================================

class Mistral3Attention(nn.Module):
    def __init__(self, config: Ministral3TextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = Mistral3RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Store original dtype for computation
        original_dtype = key_states.dtype
        
        if past_key_value is not None:
            # Handle standard tuple (K, V)
            if isinstance(past_key_value, tuple):
                if past_key_value[0] is not None:
                    # Convert cached K/V from storage dtype back to computation dtype
                    cached_k = convert_from_kv_cache_dtype(past_key_value[0], original_dtype)
                    cached_v = convert_from_kv_cache_dtype(past_key_value[1], original_dtype)
                    key_states = torch.cat([cached_k, key_states], dim=2)
                    value_states = torch.cat([cached_v, value_states], dim=2)
            # Handle Cache object (DynamicCache etc.)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Update cache return value - convert to KV cache dtype for storage
        if use_cache:
            new_past_key_value = (
                convert_to_kv_cache_dtype(key_states),
                convert_to_kv_cache_dtype(value_states)
            )
        else:
            new_past_key_value = None

        # GQA: repeat k/v heads for SDPA compatibility
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        # Use SDPA for memory-efficient attention (O(n) instead of O(n²))
        # Fallback to naive attention if output_attentions is requested
        if output_attentions:
            # Naive attention for debugging (returns attention weights)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # SDPA: Memory-efficient, supports Flash Attention
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None and q_len > 1),
            )
            attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, new_past_key_value


# ============================================================================
# Text Decoder Layer
# ============================================================================

class Mistral3DecoderLayer(nn.Module):
    def __init__(self, config: Ministral3TextConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Mistral3Attention(config, layer_idx)
        self.mlp = Mistral3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Mistral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Mistral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ============================================================================
# Vision Components (Pixtral)
# ============================================================================

class PixtralPatchEmbedding(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        if isinstance(config, dict):
            config = PixtralVisionConfig(**config)
            
        if not hasattr(config, 'num_channels'):
             in_channels = getattr(config, 'num_channels', 3)
        else:
             in_channels = config.num_channels

        self.proj = nn.Conv2d(
            in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        patches = self.proj(pixel_values)  # [B, hidden_size, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        return patches


class PixtralAttention(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use SDPA for memory-efficient attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class PixtralEncoderLayer(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.self_attn = PixtralAttention(config)
        self.mlp = Mistral3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Mistral3RMSNorm(config.hidden_size)
        self.post_attention_layernorm = Mistral3RMSNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class PixtralVisionModel(nn.Module):
    def __init__(self, config: Union[PixtralVisionConfig, dict]):
        super().__init__()
        if isinstance(config, dict):
            config = PixtralVisionConfig(**config)
        self.config = config
        self.patch_embedding = PixtralPatchEmbedding(config)
        self.layers = nn.ModuleList([PixtralEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Mistral3RMSNorm(config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================================
# Multimodal Projector
# ============================================================================

class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ============================================================================
# Text Model (Language Model Backbone)
# ============================================================================

class Mistral3TextModel(PreTrainedModel):
    config_class = Ministral3TextConfig
    
    def __init__(self, config: Union[Ministral3TextConfig, dict]):
        if isinstance(config, dict):
            config = Ministral3TextConfig(**config)
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Mistral3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Mistral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create causal mask only when needed
        # SDPA handles causal masking internally via is_causal=True when attention_mask is None
        # We only need explicit causal_mask when combining with padding mask or for output_attentions
        if attention_mask is not None:
            # Padding mask exists - need to create and combine with causal mask
            if attention_mask.dim() == 2:
                # Expand to (B, 1, 1, L) and convert to correct dtype for SDPA
                expanded_mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
                attention_mask = expanded_mask
            
            # Create causal mask for combination with padding mask
            if seq_length > 1:
                causal_mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask[None, None, :, :]  # (1, 1, Q, K)
                attention_mask = attention_mask + causal_mask
        elif output_attentions:
            # Need explicit causal mask for naive attention fallback
            if seq_length > 1:
                causal_mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                attention_mask = causal_mask[None, None, :, :]  # (1, 1, Q, K)
        # else: attention_mask stays None, SDPA will use is_causal=True


        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For DynamicCache, we pass the object itself
            # For Legacy tuple cache, we pass indexed tuple
            if isinstance(past_key_values, tuple) or past_key_values is None:
                past_key_value = past_key_values[idx] if past_key_values is not None else None
            else:
                past_key_value = past_key_values

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # If we used a Cache object, it was updated in-place or returned updated
                # If we used a tuple, we need to collect it
                if isinstance(past_key_values, tuple) or past_key_values is None:
                    next_decoder_cache += (layer_outputs[-1],)
                else:
                    next_decoder_cache = past_key_values

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if use_cache:
            pass

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ============================================================================
# Main Vision-Language Model
# ============================================================================

class Mistral3ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """
    Mistral3 Vision-Language Model for Conditional Generation.
    Combines Pixtral Vision Tower + Multimodal Projector + Mistral3 Text Model.
    """
    config_class = Mistral3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    
    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        
        # Ensure sub-configs are properly instantiated
        if isinstance(config.vision_config, dict):
            config.vision_config = PixtralVisionConfig(**config.vision_config)
        if isinstance(config.text_config, dict):
            config.text_config = Ministral3TextConfig(**config.text_config)
            
        self.vision_tower = PixtralVisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.model = Mistral3TextModel(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        self.image_token_index = config.image_token_index
        self.spatial_merge_size = config.spatial_merge_size
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        """Tie lm_head weights to embed_tokens weights if configured."""
        if getattr(self.config, 'text_config', None) and getattr(self.config.text_config, 'tie_word_embeddings', True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Merge text embeddings with image features at image token positions.
        """
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        
        # Find image token positions
        image_token_mask = input_ids == self.image_token_index
        
        # If no image tokens, return text embeddings as-is
        if not image_token_mask.any():
            return inputs_embeds
        
        # Replace image tokens with image features
        # This is a simplified version; production code needs proper handling
        new_embeds = inputs_embeds.clone()
        
        for batch_idx in range(batch_size):
            image_positions = torch.where(image_token_mask[batch_idx])[0]
            if len(image_positions) > 0 and image_features is not None:
                # Insert image features at image token positions
                num_image_tokens = min(len(image_positions), image_features.shape[1])
                for i, pos in enumerate(image_positions[:num_image_tokens]):
                    if i < image_features.shape[1]:
                        new_embeds[batch_idx, pos] = image_features[batch_idx, i]
        
        return new_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else True

        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Process images if provided
        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(image_features)
            
            # Merge text and image embeddings
            inputs_embeds = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_features
            )

        # Forward through language model
        outputs = self.model(
            input_ids=None,  # We use inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, pixel_values=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {"input_ids": input_ids}
        
        # Only pass pixel_values on first iteration
        if past_key_values is None:
            model_inputs["pixel_values"] = pixel_values

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })
        return model_inputs


# ============================================================================
# Backwards Compatibility Aliases
# ============================================================================

# Register for AutoConfig
try:
    AutoConfig.register("pixtral", PixtralVisionConfig)
    AutoConfig.register("ministral3", Ministral3TextConfig)
    AutoConfig.register("mistral3", Mistral3Config)
except Exception:
    pass

# Keep old names for compatibility with existing code
MinistralConfig = Mistral3Config
MinistralForCausalLM = Mistral3ForConditionalGeneration
