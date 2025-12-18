# VG-HuBERT Package Structure

This document describes the organization of the VG-HuBERT package, modeled after the Sylber library structure.

## Directory Layout

```
VG-HuBERT/
├── vg_hubert/                   # Main package
│   ├── __init__.py             # Exports Segmenter for inference
│   ├── segmenter.py            # High-level Segmenter interface
│   ├── mincut.py               # MinCut segmentation algorithm
│   ├── model/                  # Training models (requires fairseq)
│   │   ├── __init__.py         # Lazy imports
│   │   ├── audio_encoder.py   # VG-HuBERT audio encoder
│   │   ├── dual_encoder.py    # Audio + Vision dual encoder
│   │   ├── vision_transformer.py  # ViT models
│   │   ├── utils.py            # Margin InfoNCE loss
│   │   └── vit_utils.py        # ViT utilities
│   ├── training/               # Training components (requires fairseq, apex)
│   │   ├── __init__.py         # Lazy imports
│   │   ├── trainer.py          # Main training loop
│   │   ├── trainer_utils.py   # Training utilities
│   │   ├── bert_adam.py        # BERT-style Adam optimizer
│   │   └── utils.py            # Helper functions
│   ├── datasets/               # Dataset loaders (requires PIL)
│   │   ├── __init__.py         # Lazy imports
│   │   ├── spokencoco_dataset.py  # SpokenCOCO loader
│   │   ├── places_dataset.py   # Places loader
│   │   └── sampler.py          # StatefulSampler
│   └── utils/                  # Shared utilities (placeholder)
├── configs/                    # Training configurations
│   ├── spokencoco.yaml         # SpokenCOCO training config
│   └── places.yaml             # Places training config
├── examples/                   # Usage examples
│   ├── basic_usage.py          # Simple inference example
│   └── batch_processing.py    # Batch processing example
├── tests/                      # Test suite
│   ├── test_consistency.py    # Cross-mode consistency tests
│   ├── test_segmenter.py       # Segmenter interface tests
│   └── test_attention.py       # Attention mechanism tests
├── train.py                    # Training script (YAML-based)
├── publish_to_hub.py           # HuggingFace Hub upload script
├── demo.ipynb                  # Interactive demo notebook
├── setup.py                    # Package installation
├── requirements.txt            # Core dependencies
├── README.md                   # Main documentation
└── LICENSE                     # BSD-3-Clause license

```

## Key Design Decisions

### 1. Lazy Imports for Optional Dependencies

Training components (`model/`, `training/`, `datasets/`) use lazy imports to avoid requiring heavy dependencies (fairseq, apex, PIL) for inference-only use:

```python
# Inference works without training dependencies
from vg_hubert import Segmenter  # ✓ Works immediately

# Training components loaded on-demand
from vg_hubert.model import DualEncoder  # Only imports when accessed
from vg_hubert.training import Trainer   # Only imports when accessed
```

### 2. Sylber-Inspired Structure

Following Sylber's organization:
- All code inside main package directory (`vg_hubert/`)
- Separate subdirectories for model architecture, training, and data loading
- YAML-based configuration system
- Top-level `train.py` script for entry point

### 3. Dual-Purpose Package

**Inference**: Lightweight, easy to use
```python
from vg_hubert import Segmenter
segmenter = Segmenter(model_ckpt="username/vg-hubert", mode="syllable")
results = segmenter.segment_audio("audio.wav")
```

**Training**: Full research capabilities
```bash
python train.py --config configs/spokencoco.yaml --gpus 4
```

## Training from Scratch

Complete training pipeline preserved:

1. **Dataset Preparation**: SpokenCOCO or Places with image-caption pairs
2. **Configuration**: Edit YAML files in `configs/`
3. **Training**: Run `python train.py --config configs/spokencoco.yaml`
4. **Models Used**:
   - Audio: VG-HuBERT (modified HuBERT-base)
   - Vision: DINO ViT (tiny/small/base)
   - Objective: Margin InfoNCE loss for cross-modal alignment

## Distribution

- **Model Weights**: HuggingFace Hub (`username/vg-hubert`)
- **Code**: PyPI package (`pip install vg-hubert`)
- **Source**: GitHub repository

## Dependencies

### Core (Inference)
- torch >= 2.0.0
- transformers >= 4.20.0
- huggingface-hub >= 0.10.0
- numpy, soundfile, scipy

### Training (Optional)
- fairseq >= 0.10.0
- apex (NVIDIA mixed precision)
- Pillow (image loading)
- matplotlib, scikit-learn, seaborn

## Migration from Original

**Old Structure** (root-level directories):
```
models/          → vg_hubert/model/
steps/           → vg_hubert/training/
datasets/        → vg_hubert/datasets/
scripts/*.sh     → [removed, replaced with train.py + YAML]
```

**Import Changes**:
```python
# Old (required models/ at root)
from models import DualEncoder

# New (inside package)
from vg_hubert.model import DualEncoder
```

## License

BSD-3-Clause (Copyright 2022 Puyuan Peng)
