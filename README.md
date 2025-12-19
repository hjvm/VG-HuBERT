# VG-HuBERT: Speech Segmentation with Simplified Interface

Unsupervised syllable and word segmentation using visually grounded HuBERT (VG-HuBERT). This fork provides a simplified interface with HuggingFace Hub integration, updated PyTorch version to eliminate the need for PyTorch `multi_head_attention_forward` patching, and a PyPl package distribution.

## Quick Start

```python
from vg_hubert import Segmenter

# Syllable segmentation
segmenter = Segmenter(mode="syllable")
outputs = segmenter("audio.wav")

# Word segmentation  
word_segmenter = Segmenter(mode="word")
word_outputs = word_segmenter("audio.wav")
```

## Installation

```bash
# From source
pip install git+https://github.com/hjvm/VG-HuBERT.git

# Or PyPI (after publishing)
pip install vg-hubert
```

**Requirements**: Python ≥3.8, PyTorch ≥2.0, transformers, scipy, soundfile

## Usage

### Basic Example

```python
from vg_hubert import Segmenter
import soundfile as sf

# Load and segment
segmenter = Segmenter(
    model_ckpt="YOUR_USERNAME/vg-hubert",  # HuggingFace Hub or local path
    mode="syllable",
    device="cuda"  # or "mps" or "cpu" (auto-detects best available)
)

outputs = segmenter("audio.wav")

# Access results
for start, end in outputs['segments']:
    print(f"Segment: {start:.2f}s - {end:.2f}s")

# Access features
segment_features = outputs['segment_features']  # [num_segments, 768]
frame_features = outputs['hidden_states']       # [num_frames, 768]
```

### Parameters

- **mode**: `"syllable"` (MinCut + feature similarity) or `"word"` (CLS attention)
- **layer**: HuBERT layer to use (default: 8 for syllables, 9 for words)
- **device**: `"cuda"`, `"mps"`, or `"cpu"` (defaults to CUDA if available, falls back to MPS on Apple Silicon, then CPU)
- **sec_per_syllable**: Target syllable duration for MinCut (default: 0.2)
- **merge_threshold**: Similarity threshold for merging segments (default: 0.3)
- **attn_threshold**: Attention threshold for word boundaries (default: 0.25)

See [examples/](examples/) for more usage patterns.

## Model Details

### Checkpoints

Two pre-trained models optimized for different tasks:

| Checkpoint | Task | Layer | Algorithm | Size |
|------------|------|-------|-----------|------|
| `vg-hubert-syllable.pth` | Syllable | 8 | MinCut + Feature SSM | 474 MB |
| `vg-hubert-word.pth` | Word | 9 | CLS Attention | 361 MB |

### Performance (SpokenCOCO)

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

## Related Repositories

- **Original implementations**: [word-discovery](https://github.com/jasonppy/word-discovery), [syllable-discovery](https://github.com/jasonppy/syllable-discovery)
- **Fork parent**: [human-ai-lab/VG-HuBERT](https://github.com/human-ai-lab/VG-HuBERT)
- **Interface inspiration**: [Sylber](https://github.com/Berkeley-Speech-Group/sylber)

## License

BSD-3-Clause License (same as original repositories)

## Contributing

Issues and pull requests welcome. Please ensure changes maintain compatibility with original model weights and include proper attribution.
