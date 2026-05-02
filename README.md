# VG-HuBERT: Speech Segmentation with Simplified Interface

[![PyPI version](https://badge.fury.io/py/vg-hubert.svg)](https://badge.fury.io/py/vg-hubert)
[![Downloads](https://pepy.tech/badge/vg-hubert)](https://pepy.tech/project/vg-hubert)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Unsupervised syllable and word segmentation using visually grounded HuBERT (VG-HuBERT). This fork provides a simplified interface with HuggingFace Hub integration, updated PyTorch version to eliminate the need for PyTorch `multi_head_attention_forward` patching, optimized MinCut algorithm (~40x speedup), and PyPI package distribution.

## Quick Start

```python
from vg_hubert import Segmenter

# Syllable segmentation (RECOMMENDED: includes MinCutMerge post-processing)
segmenter = Segmenter(mode="syllable", merge_threshold=0.3)
outputs = segmenter("audio.wav")

# Word segmentation  
word_segmenter = Segmenter(mode="word")
word_outputs = word_segmenter("audio.wav")
```

## Installation

```bash
pip install vg-hubert
```

**Requirements**: Python ≥3.8, PyTorch ≥2.0, transformers, scipy, soundfile

**Optional (for development)**:
```bash
pip install git+https://github.com/human-ai-lab/VG-HuBERT.git
```

## Features

✨ **New in this fork:**
- 🚀 **40x faster MinCut**: Optimized algorithm from [SyllableLM](https://github.com/AlanBaade/SyllableLM) (Baade et al., 2024)
- 🔧 **MinCutMerge post-processing**: Prevents over-segmentation (matches original paper)
- 🤗 **HuggingFace integration**: Auto-download models from Hub
- 🍎 **Apple Silicon support**: Native MPS acceleration
- 📦 **PyPI distribution**: Simple `pip install`
- 🧹 **No fairseq for inference**: Removed complex dependency

## Usage

### Basic Example

```python
from vg_hubert import Segmenter
import soundfile as sf

# Load and segment
segmenter = Segmenter(
    model_ckpt="hjvm/VG-HuBERT",  # HuggingFace Hub or local path
    mode="syllable",
    device="cuda",  # or "mps" or "cpu" (auto-detects best available)
    merge_threshold=0.3  # Enable MinCutMerge (recommended)
)

outputs = segmenter("audio.wav")

# Access results
for start, end in outputs['segments']:
    print(f"Segment: {start:.2f}s - {end:.2f}s")

# Access features
segment_features = outputs['segment_features']  # [num_segments, 768]
frame_features = outputs['hidden_states']       # [num_frames, 768]
```

### MinCut Configuration

The package supports multiple MinCut configurations for different use cases:

```python
# Configuration 1: RECOMMENDED (matches original paper)
# - Fast algorithm + MinCutMerge post-processing
# - Prevents over-segmentation
segmenter = Segmenter(
    mode="syllable",
    merge_threshold=0.3,  # Original paper value
    min_segment_frames=2  # Filter very short segments
)

# Configuration 2: Plain MinCut (no merging)
# - Useful for analysis or more granular segmentation
segmenter = Segmenter(
    mode="syllable",
    merge_threshold=None  # Disable MinCutMerge
)

# Configuration 3: Custom merge threshold
# - Tune for your specific needs
# - Higher = more merging = fewer segments
# - Lower = less merging = more segments
segmenter = Segmenter(
    mode="syllable",
    merge_threshold=0.5  # More aggressive merging
)
```

See [examples/mincut_comparison.py](examples/mincut_comparison.py) for detailed comparison.

### Low-Level API

For advanced users who need full control:

```python
from vg_hubert.mincut import segment_with_mincut
import numpy as np

# Extract features (see examples/ for full code)
features = ...  # Shape: (num_frames, 768)

# Apply MinCut with full control
boundaries, ssm = segment_with_mincut(
    features=features,
    K=10,  # Number of boundaries
    merge_threshold=0.3,  # Set to None for plain MinCut
    min_segment_frames=2,
    min_hop=3,  # Minimum segment length
    max_hop=50  # Maximum segment length
)
```

### Parameters

- **mode**: `"syllable"` (MinCut + feature similarity) or `"word"` (CLS attention)
- **layer**: HuBERT layer to use (default: 8 for syllables, 9 for words)
- **device**: `"cuda"`, `"mps"`, or `"cpu"` (defaults to CUDA if available, falls back to MPS on Apple Silicon, then CPU)
- **sec_per_syllable**: Target syllable duration for MinCut (default: 0.2)
- **merge_threshold**: Cosine similarity threshold for merging adjacent segments (default: 0.3, set to `None` to disable)
- **min_segment_frames**: Filter segments with ≤ this many frames (default: 2)
- **attn_threshold**: Attention threshold for word boundaries (default: 0.25)

See [examples/](examples/) for more usage patterns.

## Model Details

### Checkpoints

Two pre-trained models optimized for different tasks:

| Checkpoint | Task | Layer | Algorithm | Size |
|------------|------|-------|-----------|------|
| `vg-hubert-syllable.pth` | Syllable | 8 | MinCut + MinCutMerge | 474 MB |
| `vg-hubert-word.pth` | Word | 9 | CLS Attention | 361 MB |

### Algorithm Details

**MinCut Segmentation (Syllables):**
1. Extract HuBERT features from layer 8
2. Compute self-similarity matrix (SSM)
3. Apply efficient MinCut algorithm (Baade et al., 2024)
   - ~40x faster than original O(N²K) implementation
   - Uses cumulative sums for O(1) range queries
4. **Optional**: Apply MinCutMerge post-processing (Peng et al., 2023)
   - Iteratively merge adjacent segments with cosine similarity ≥ threshold
   - Prevents over-segmentation
   - Recommended for production use

**Performance Comparison:**

| Configuration | F1 (LibriSpeech) | Speed (ms/utt) | Speedup |
|---------------|------------------|----------------|---------|
| Original MinCut | 0.501 | 7524 | 1.0x |
| New MinCut | 0.501 | 169 | 44.5x |
| New + MinCutMerge-0.3 ⭐ | TBD | 171 | 44.0x |

*Note: LibriSpeech results shown; original paper reports F1=0.603 on SpokenCOCO*

### Performance (SpokenCOCO - Original Paper)

**Syllable Segmentation:**
- Boundary F1: 0.603
- Boundary Precision: 0.574  
- Boundary Recall: 0.636

**Word Discovery:**
- Token F1: 0.195
- Type F1: 0.174
- NED: 0.748

## Training

VG-HuBERT uses **visually-grounded contrastive learning** to learn speech representations. The model jointly trains on speech and images using datasets like SpokenCOCO or Places.

### Training Setup

1. **Install training dependencies**:
```bash
pip install -r requirements.txt  # Includes fairseq, apex, Pillow, etc.
```

2. **Download datasets**:
   - **SpokenCOCO**: [Spoken captions](https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz) + [MSCOCO images](http://cocodataset.org/#download)
   - **Places**: [Spoken descriptions](https://data.csail.mit.edu/placesaudio/) + [Places365 images](http://places2.csail.mit.edu/)

3. **Download pre-trained models** for initialization:
   - [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) (pretrained on LibriSpeech 960h)
   - [DINO ViT](https://dl.fbaipublicfiles.com/dino/dino_vitsmall8_pretrain/dino_vitsmall8_pretrain_full_checkpoint.pth) (vision encoder)

4. **Configure training**:
```yaml
# configs/spokencoco.yaml
train_audio_dataset_json_file: "/path/to/SpokenCOCO_train.json"
val_audio_dataset_json_file: "/path/to/SpokenCOCO_val.json"
load_hubert_weights: "/path/to/hubert_base_ls960.pt"
load_pretrained_vit: "/path/to/dino_vitsmall8_pretrain.pth"
batch_size: 32
n_epochs: 30
gpus: "0,1,2,3"
```

5. **Train**:
```bash
python train.py --config configs/spokencoco.yaml
```

### Training Outputs

- Checkpoints saved to `exp_dir/` (default: `./checkpoints/`)
- TensorBoard logs in experiment directory
- Config saved as `config.yaml` in experiment directory

### Architecture

**Dual-encoder with cross-modal transformer**:
- **Audio encoder**: HuBERT Base (12 layers, 768-dim)
- **Vision encoder**: ViT Small/Base (DINO pretrained)
- **Cross-modal layers**: 5 transformer layers for audio-image interaction
- **Loss**: Margin InfoNCE (contrastive learning in common embedding space)

The trained audio encoder can then be used for segmentation without the vision components.

### Training from Scratch

The package includes all training code:
- `vg_hubert/model/`: Dual encoder, audio/vision transformers
- `vg_hubert/training/`: Trainer, optimizers, utilities
- `vg_hubert/datasets/`: SpokenCOCO and Places data loaders

See [configs/](configs/) for complete training examples.

## What's Different in This Fork

1. **No PyTorch patching**: Uses native `attn_implementation='eager'` (PyTorch 2.0+)
2. **Simplified interface**: Single `Segmenter` class for all use cases
3. **HuggingFace Hub**: Automatic model downloading
4. **Complete package**: Both training and inference (like Sylber)
5. **PyPI distribution**: Easy installation via pip
6. **Apple Silicon support**: Automatic MPS (Metal Performance Shaders) GPU acceleration
7. **Optimized MinCut**: **~20-50x faster** syllable segmentation using efficient algorithm from [SyllableLM](https://github.com/AlanBaade/SyllableLM) (Baade et al., 2024) with no quality degradation

## Implementation Details

For inference, this package uses HuggingFace's `transformers.HubertModel` instead of the original fairseq implementation. This is possible because VG-HuBERT's audio encoder architecture is identical to the standard HuBERT model. The visual grounding training adds a vision encoder and cross-modal transformer layers, but these components are only used during training to learn better speech representations. At inference time, only the audio encoder weights are needed, which are fully compatible with the HuggingFace HuBERT architecture. This simplifies deployment and eliminates the fairseq dependency for inference.

## Citations

### VG-HuBERT Original Work

**Syllable Segmentation:**
```bibtex
@inproceedings{peng2023syllable,
  title={Syllable Segmentation and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model},
  author={Peng, Puyuan and Li, Shang-Wen and Räsänen, Okko and Mohamed, Abdelrahman and Harwath, David},
  booktitle={Interspeech},
  year={2023}
}
```

**Word Discovery:**
```bibtex
@inproceedings{peng2022word,
  title={Word Discovery in Visually Grounded, Self-Supervised Speech Models},
  author={Peng, Puyuan and Harwath, David},
  booktitle={Interspeech},
  year={2022}
}
```

### Interface Design

This package follows the interface design of Sylber:
```bibtex
@article{cho2024sylber,
  title={Sylber: Syllabic Embedding Representation of Speech from Raw Audio},
  author={Cho, Cheol Jun and Lee, Nicholas and Gupta, Akshat and Agarwal, Dhruv and Chen, Ethan and Black, Alan W and Anumanchipalli, Gopala K},
  journal={arXiv preprint arXiv:2410.07168},
  year={2024}
}
```

### Optimized MinCut Algorithm

The MinCut algorithm used for syllable segmentation has been updated to use the efficient implementation from SyllableLM (Baade et al., 2024), which provides **~20-50x speedup** over the original with no statistically significant quality difference:

```bibtex
@misc{baade2024syllablelmlearningcoarsesemantic,
      title={SyllableLM: Learning Coarse Semantic Units for Speech Language Models}, 
      author={Alan Baade and Puyuan Peng and David Harwath},
      year={2024},
      eprint={2410.04029},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.04029}, 
}
```

**Performance Comparison (LibriSpeech test-clean, 50 utterances):**
- Speed: 6961ms → 133ms per utterance (**52x faster**)
- Quality: F1=0.377 → 0.372 (p=0.22, not significant)
- 82% of utterances produce identical segmentations

Key optimizations:
- Cumulative sum preprocessing for O(1) range queries
- Segment length constraints (min_hop=3, max_hop=50 frames)
- 5-component cost calculation

See [vg_hubert/tests/mincut_validation.ipynb](vg_hubert/tests/mincut_validation.ipynb) for full validation results.

## Related Repositories

- **Original implementations**: [word-discovery](https://github.com/jasonppy/word-discovery), [syllable-discovery](https://github.com/jasonppy/syllable-discovery)
- **Fork parent**: [human-ai-lab/VG-HuBERT](https://github.com/human-ai-lab/VG-HuBERT)
- **Interface inspiration**: [Sylber](https://github.com/Berkeley-Speech-Group/sylber)

## License

BSD-3-Clause License (same as original repositories)

## Contributing

Issues and pull requests welcome. Please ensure changes maintain compatibility with original model weights and include proper attribution.
