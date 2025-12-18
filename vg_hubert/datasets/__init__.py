"""VG-HuBERT Dataset Loaders

Requires PIL (Pillow) for image loading.
"""

__all__ = ["SpokenCOCODataset", "PlacesAudioDataset", "StatefulSampler"]

def __getattr__(name):
    """Lazy import dataset components."""
    if name == "SpokenCOCODataset":
        from .spokencoco_dataset import ImageCaptionDataset as SpokenCOCODataset
        return SpokenCOCODataset
    elif name == "PlacesAudioDataset":
        from .places_dataset import PlacesAudioDataset
        return PlacesAudioDataset
    elif name == "StatefulSampler":
        from .sampler import StatefulSampler
        return StatefulSampler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

