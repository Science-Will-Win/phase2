"""
Mistral3 Vision-Language Model with mHC (Manifold-Constrained Hyper-Connections)
Based on ministral_3_3b_instruct.py with residual connections replaced by mHC

Reference: arXiv:2512.24880 - mHC: Manifold-Constrained Hyper-Connections
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers import AutoConfig

# Import mHC components (from root directory)
from mHC import mHCConfig, mHCConnection, mHCBlock, expand_to_streams, collapse_from_streams

# ============================================================================
# Import shared components from original model (relative import)
# ============================================================================

from .ministral_3_3b_instruct import (
    Ministral3TokenConfig,
    PixtralVisionConfig,
    Ministral3TextConfig,
    Mistral3RMSNorm,
    Mistral3RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    Mistral3MLP,
    Mistral3Attention,
    # Vision components (unchanged)
    PixtralPatchEmbedding,
    PixtralAttention,
    PixtralEncoderLayer,
    PixtralVisionModel,
    Mistral3MultiModalProjector,
)


# ============================================================================
# File Configuration for mHC Model
# ============================================================================

class Ministral3mHCFileConfig:
    """Model-related file paths/roles configuration (mHC version)"""
    
    # Base path (uses same files as original model)
    BASE_PATH: str = "model/ministral_3_3b_instruct"
    
    # For download
    HF_REPO_ID: str = "mistralai/Ministral-3-3B-Instruct-2512"
    
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
# mHC-specific Configuration
# ============================================================================

class Mistral3Config_mHC(PretrainedConfig):
    """Main configuration for the Mistral3 VLM with mHC support"""
    model_type = "mistral3_mhc"
    
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        # mHC specific parameters
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iterations: int = 20,
        mhc_alpha_init: float = 0.01,
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
        
        # mHC parameters
        self.mhc_n_streams = mhc_n_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_alpha_init = mhc_alpha_init
        
        # Expose commonly used text config attributes at top level
        if self.text_config is not None:
            for key, value in self.text_config.to_dict().items():
                if not hasattr(self, key):
                    setattr(self, key, value)
        else:
            self.hidden_size = getattr(self, "hidden_size", 3072)
            self.num_hidden_layers = getattr(self, "num_hidden_layers", 26)
    
    def get_mhc_config(self) -> mHCConfig:
        """Create mHCConfig from model config."""
        return mHCConfig(
            hidden_size=self.text_config.hidden_size,
            n_streams=self.mhc_n_streams,
            sinkhorn_iterations=self.mhc_sinkhorn_iterations,
            alpha_init=self.mhc_alpha_init,
        )


# ============================================================================
# Text Decoder Layer with mHC
# ============================================================================

class Mistral3DecoderLayer_mHC(nn.Module):
    """
    Mistral3 Decoder Layer with mHC replacing standard residual connections.
    
    Standard residual:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    mHC residual:
        x = H_res @ x + H_post @ Attention(LayerNorm(H_pre @ x))
        x = H_res @ x + H_post @ MLP(LayerNorm(H_pre @ x))
    
    Where x is [B, S, n, C] (n-stream hidden states)
    """
    
    def __init__(self, config: Ministral3TextConfig, layer_idx: int = 0, mhc_config: mHCConfig = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Original components
        self.self_attn = Mistral3Attention(config, layer_idx)
        self.mlp = Mistral3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Mistral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Mistral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # mHC connections (replace residual connections)
        if mhc_config is None:
            mhc_config = mHCConfig(hidden_size=config.hidden_size)
        self.mhc_block = mHCBlock(mhc_config)
        self.n_streams = mhc_config.n_streams

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, n, C] - n-stream format
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """
        Forward pass with mHC connections.
        
        Args:
            hidden_states: [B, S, n, C] - n-stream hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional KV cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple of (hidden_states, attention_weights, present_key_value)
        """
        # ================================================================
        # 1. Self-Attention with mHC
        # ================================================================
        
        # Aggregate n-stream to single input for attention: [B, S, n, C] -> [B, S, C]
        attn_input = self.mhc_block.aggregate_for_attention(hidden_states)
        
        # Apply input layer norm
        attn_input = self.input_layernorm(attn_input)
        
        # Self-attention forward
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # Apply mHC residual connection: [B, S, n, C]
        hidden_states = self.mhc_block.apply_attention_residual(hidden_states, attn_output)
        
        # ================================================================
        # 2. MLP with mHC
        # ================================================================
        
        # Aggregate n-stream to single input for MLP: [B, S, n, C] -> [B, S, C]
        mlp_input = self.mhc_block.aggregate_for_mlp(hidden_states)
        
        # Apply post-attention layer norm
        mlp_input = self.post_attention_layernorm(mlp_input)
        
        # MLP forward
        mlp_output = self.mlp(mlp_input)
        
        # Apply mHC residual connection: [B, S, n, C]
        hidden_states = self.mhc_block.apply_mlp_residual(hidden_states, mlp_output)
        
        # ================================================================
        # Return outputs
        # ================================================================
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ============================================================================
# Text Model with mHC
# ============================================================================

class Mistral3TextModel_mHC(PreTrainedModel):
    """
    Mistral3 Text Model with mHC n-stream residual architecture.
    
    The hidden states flow through the model as [B, S, n, C] tensors,
    where n is the number of parallel streams defined in mHC.
    """
    config_class = Ministral3TextConfig
    
    def __init__(self, config: Union[Ministral3TextConfig, dict], mhc_config: mHCConfig = None):
        if isinstance(config, dict):
            config = Ministral3TextConfig(**config)
        super().__init__(config)
        
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        
        # mHC configuration
        if mhc_config is None:
            mhc_config = mHCConfig(hidden_size=config.hidden_size)
        self.mhc_config = mhc_config
        self.n_streams = mhc_config.n_streams
        
        # Embedding layer (standard, outputs [B, S, C])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Decoder layers with mHC
        self.layers = nn.ModuleList([
            Mistral3DecoderLayer_mHC(config, i, mhc_config) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization (operates on collapsed [B, S, C])
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

        # Get embeddings: [B, S, C]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create causal mask
        if seq_length > 1:
            causal_mask = torch.full(
                (seq_length, seq_length), 
                torch.finfo(inputs_embeds.dtype).min, 
                device=inputs_embeds.device, 
                dtype=inputs_embeds.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :]
        else:
            causal_mask = None
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :]
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
                attention_mask = expanded_mask
            
            if causal_mask is not None:
                attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask

        # ================================================================
        # Expand to n-stream format: [B, S, C] -> [B, S, n, C]
        # ================================================================
        hidden_states = expand_to_streams(inputs_embeds, self.n_streams)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # ================================================================
        # Process through decoder layers
        # ================================================================
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # Collapse for output: [B, S, n, C] -> [B, S, C]
                all_hidden_states += (collapse_from_streams(hidden_states),)

            if isinstance(past_key_values, tuple) or past_key_values is None:
                past_key_value = past_key_values[idx] if past_key_values is not None else None
            else:
                past_key_value = past_key_values

            layer_outputs = decoder_layer(
                hidden_states,  # [B, S, n, C]
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]  # [B, S, n, C]

            if use_cache:
                if isinstance(past_key_values, tuple) or past_key_values is None:
                    next_decoder_cache += (layer_outputs[-1],)
                else:
                    next_decoder_cache = past_key_values

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # ================================================================
        # Collapse from n-stream and apply final norm: [B, S, n, C] -> [B, S, C]
        # ================================================================
        hidden_states = collapse_from_streams(hidden_states)
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
# Main Vision-Language Model with mHC
# ============================================================================

class Mistral3ForConditionalGeneration_mHC(PreTrainedModel, GenerationMixin):
    """
    Mistral3 Vision-Language Model with mHC for Conditional Generation.
    Combines Pixtral Vision Tower + Multimodal Projector + Mistral3 Text Model with mHC.
    """
    config_class = Mistral3Config_mHC
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    
    def __init__(self, config: Mistral3Config_mHC):
        super().__init__(config)
        
        # Ensure sub-configs are properly instantiated
        if isinstance(config.vision_config, dict):
            config.vision_config = PixtralVisionConfig(**config.vision_config)
        if isinstance(config.text_config, dict):
            config.text_config = Ministral3TextConfig(**config.text_config)
        
        # Get mHC config
        mhc_config = config.get_mhc_config()
        
        # Vision components (unchanged from original)
        self.vision_tower = PixtralVisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        
        # Text model with mHC
        self.model = Mistral3TextModel_mHC(config.text_config, mhc_config)
        
        # LM head
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
        """Merge text embeddings with image features at image token positions."""
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        
        image_token_mask = input_ids == self.image_token_index
        
        if not image_token_mask.any():
            return inputs_embeds
        
        new_embeds = inputs_embeds.clone()
        
        for batch_idx in range(batch_size):
            image_positions = torch.where(image_token_mask[batch_idx])[0]
            if len(image_positions) > 0 and image_features is not None:
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
            
            inputs_embeds = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_features
            )

        # Forward through language model with mHC
        outputs = self.model(
            input_ids=None,
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
# Register for AutoConfig
# ============================================================================

try:
    AutoConfig.register("mistral3_mhc", Mistral3Config_mHC)
except Exception:
    pass


# ============================================================================
# Convenience function to convert standard model to mHC model
# ============================================================================

def convert_to_mhc_config(original_config, n_streams: int = 4) -> Mistral3Config_mHC:
    """
    Convert a standard Mistral3Config to mHC version.
    
    Args:
        original_config: Original Mistral3Config
        n_streams: Number of mHC streams (default 4)
        
    Returns:
        Mistral3Config_mHC with same parameters plus mHC settings
    """
    return Mistral3Config_mHC(
        text_config=original_config.text_config,
        vision_config=original_config.vision_config,
        image_token_index=original_config.image_token_index,
        projector_hidden_act=original_config.projector_hidden_act,
        multimodal_projector_bias=original_config.multimodal_projector_bias,
        spatial_merge_size=original_config.spatial_merge_size,
        vision_feature_layer=original_config.vision_feature_layer,
        mhc_n_streams=n_streams,
    )
