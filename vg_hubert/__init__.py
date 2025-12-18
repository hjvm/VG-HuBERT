"""
VG-HuBERT: Visually Grounded HuBERT for Speech Segmentation

A self-supervised speech model trained with visual grounding for 
syllable and word discovery tasks.

Papers:
- Word Discovery in Visually Grounded, Self-Supervised Speech Models
  Peng & Harwath, Interspeech 2022
- Syllable Segmentation and Cross-Lingual Generalization in a Visually
  Grounded, Self-Supervised Speech Model
  Peng et al., Interspeech 2023

Original Repos:
- https://github.com/jasonppy/word-discovery
- https://github.com/jasonppy/syllable-discovery
"""

from .segmenter import Segmenter

# Training modules available via subpackages
# from vg_hubert.model import DualEncoder, AudioEncoder
# from vg_hubert.training import Trainer
# from vg_hubert.datasets import SpokenCOCODataset, PlacesAudioDataset

__all__ = ["Segmenter"]

__version__ = "1.0.0"
__version__ = "1.0.0"
__author__ = "Puyuan Peng, David Harwath (original); YOUR_NAME (fork maintainer)"

__all__ = [
    "Segmenter",  # Main user-facing class (Sylber-compatible)
]
