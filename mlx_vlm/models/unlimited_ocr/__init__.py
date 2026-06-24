"""baidu/Unlimited-OCR support for MLX-VLM.

Unlimited-OCR (`model_type: unlimited-ocr`, arch `UnlimitedOCRForCausalLM`) is a
DeepSeek-OCR-derived document parser: a DeepSeek-V2 MLA+MoE language model with a
dual DeepEncoder vision stack (SAM ViT-B + CLIP-L) and an MLP projector.

Its full weight set maps onto the existing DeepSeek-OCR (v1) MLX architecture
with no missing or extra parameters (verified via Backend/convert_unlimited_ocr.py
`probe-load --alias-model-type deepseekocr --model-only` against the real
checkpoint). This package therefore reuses the v1 modules unchanged and exists so
that `model_type: unlimited-ocr` resolves to a real architecture; see the
`MODEL_REMAPPING` entry in mlx_vlm/utils.py that maps the hyphenated model_type
to this package.
"""

from ..base import install_auto_processor_patch
from ..deepseekocr import (
    DeepseekOCRProcessor,
    LanguageModel,
    Model,
    VisionModel,
)
from ..deepseekocr.config import (
    MLPConfig,
    ModelConfig,
    ProjectorConfig,
    TextConfig,
    VisionConfig,
)
from .processing import UnlimitedOCRProcessor

# Unlimited-OCR's processor_config.json declares processor_class
# "UnlimitedOCRHFProcessor", which transformers' AutoProcessor cannot resolve.
# Reuse the DeepSeek-OCR processor (matching preprocessing) via the composable
# AutoProcessor patch, keyed on the `unlimited-ocr` model_type. We register the
# UnlimitedOCRProcessor subclass (not the base) because the base patch forces
# trust_remote_code=True, which would pull in the repo's torch-only remote code;
# the subclass forces it back off. The patch chains with the deepseekocr one.
install_auto_processor_patch("unlimited-ocr", UnlimitedOCRProcessor)

__all__ = [
    "DeepseekOCRProcessor",
    "LanguageModel",
    "MLPConfig",
    "Model",
    "ModelConfig",
    "ProjectorConfig",
    "TextConfig",
    "UnlimitedOCRProcessor",
    "VisionConfig",
    "VisionModel",
]
