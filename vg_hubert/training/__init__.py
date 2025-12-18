"""VG-HuBERT Training Components

Requires fairseq and apex for training.
"""

__all__ = ["Trainer", "BertAdam"]

def __getattr__(name):
    """Lazy import training components."""
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    elif name == "BertAdam":
        from .bert_adam import BertAdam
        return BertAdam
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

