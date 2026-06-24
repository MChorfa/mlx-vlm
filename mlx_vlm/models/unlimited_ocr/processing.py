"""Processor for baidu/Unlimited-OCR.

Reuses DeepSeek-OCR's processor — the preprocessing matches (image_mean/std 0.5,
patch_size 16, candidate_resolutions [[1024, 1024]], image_token "<image>").

The raw HF repo ships torch-based custom modeling code (modeling_unlimitedocr.py).
transformers' processor loader pulls that remote code in when
``trust_remote_code=True``, which fails in an MLX-only environment (it needs
torch, torchvision, addict). MLX reconstructs the processor purely from
``processor_config.json`` + the tokenizer and never needs that code, so this
subclass forces ``trust_remote_code`` off before delegating.
"""

from ..deepseekocr import DeepseekOCRProcessor


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepSeek-OCR processor that ignores the HF repo's torch remote code."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs["trust_remote_code"] = False
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
