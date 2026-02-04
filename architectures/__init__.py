# Architecture definitions package
# Contains model architecture implementations

from .ministral_3_3b_instruct import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Ministral3TokenConfig,
)

# Reasoning model variant
from .ministral_3_3b_reasoning import (
    Mistral3Config as Mistral3Config_Reasoning,
    Mistral3ForConditionalGeneration as Mistral3ForConditionalGeneration_Reasoning,
    Ministral3TokenConfig as Ministral3TokenConfig_Reasoning,
)

# Optional: mHC variant
try:
    from .ministral_3_3b_instruct_mHC import (
        Mistral3Config_mHC,
        Mistral3ForConditionalGeneration_mHC,
    )
except ImportError:
    pass
