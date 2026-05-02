"""VG-HuBERT Model Components

Training models require fairseq. For inference-only use, import Segmenter directly:
    from vg_hubert import Segmenter
"""

# Training models (optional dependencies)
__all__ = [
    "AudioEncoder",
    "DualEncoder",
    "vit_tiny",
    "vit_small", 
    "vit_base",
    "Margin_InfoNCE_loss",
]
def __getattr__(name):
    """Lazy import training models to avoid requiring fairseq for inference."""
    if name == "AudioEncoder":
        from .audio_encoder import AudioEncoder
        return AudioEncoder
    elif name == "DualEncoder":
        from .dual_encoder import DualEncoder
        return DualEncoder
    elif name == "vit_tiny":
        from .vision_transformer import vit_tiny
        return vit_tiny
    elif name == "vit_small":
        from .vision_transformer import vit_small
        return vit_small
    elif name == "vit_base":
        from .vision_transformer import vit_base
        return vit_base
    elif name == "Margin_InfoNCE_loss":
        from .utils import Margin_InfoNCE_loss
        return Margin_InfoNCE_loss
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
