#!/usr/bin/env python3
"""
Upload VG-HuBERT model to HuggingFace Hub.

Usage:
    python publish_to_hub.py --repo_id your-username/vg-hubert --model_path ../findsylls/models/vg-hubert_3
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import os


def create_model_card(repo_id: str) -> str:
    """Create a comprehensive model card for VG-HuBERT."""
    
    card = f"""---
language: en
tags:
- audio
- speech
- syllable-segmentation
- word-segmentation
- unsupervised
- self-supervised
- hubert
- visually-grounded
license: bsd-3-clause
datasets:
- spokencoco
- librispeech
library_name: transformers
base_model: facebook/hubert-base-ls960
---

# VG-HuBERT: Visually Grounded HuBERT for Speech Segmentation

VG-HuBERT is a self-supervised speech model for unsupervised syllable and word segmentation. It uses visually grounded learning to discover linguistic units in speech without transcriptions.

**Original Implementation**: This model is based on the work by Peng et al. from the [word-discovery](https://github.com/jasonppy/word-discovery) and [syllable-discovery](https://github.com/jasonppy/syllable-discovery) repositories. This HuggingFace release provides simplified access to the pre-trained models without requiring PyTorch source code modifications.

## Model Description

This model provides two segmentation modes:

- **Syllable Segmentation**: Uses MinCut algorithm on feature self-similarity matrices (best_bundle.pth, layer 8)
- **Word Segmentation**: Uses CLS token attention for boundary detection (snapshot_20.pth, layer 9)

## Quick Start

```python
from vg_hubert import Segmenter

# Syllable segmentation
segmenter = Segmenter(mode="syllable")
outputs = segmenter("audio.wav")
print(f"Found {{len(outputs['segments'])}} syllables")

# Word segmentation
word_segmenter = Segmenter(mode="word")
word_outputs = word_segmenter("audio.wav")
print(f"Found {{len(word_outputs['segments'])}} words")
```

## Installation

```bash
pip install git+https://github.com/yourusername/VG-HuBERT.git
```

Or install dependencies:
```bash
pip install torch transformers soundfile scipy huggingface-hub
```

## Model Details

- **Architecture**: HuBERT base (12 layers, 768 hidden size, 12 attention heads)
- **Training Data**: SpokenCOCO (visually grounded speech)
- **Base Model**: facebook/hubert-base-ls960
- **Parameters**: ~95M

### Checkpoints

- `vg-hubert-syllable.pth`: Optimized for syllable segmentation (layer 8, MinCut algorithm)
- `vg-hubert-word.pth`: Optimized for word segmentation (layer 9, CLS attention)

## Performance

### Syllable Segmentation (SpokenCOCO)
- Boundary F1: 0.603
- Boundary Precision: 0.574
- Boundary Recall: 0.636

### Word Discovery (SpokenCOCO)
- Token F1: 0.195
- Type F1: 0.174
- NED: 0.748

## Usage

### Basic Usage

```python
from vg_hubert import Segmenter
import soundfile as sf

# Load audio
audio, sr = sf.read("speech.wav")

# Syllable segmentation
segmenter = Segmenter(mode="syllable")
outputs = segmenter(wav=audio)

# Access results
for start, end in outputs['segments']:
    print(f"Syllable: {{start:.2f}}s - {{end:.2f}}s")

# Get segment features
features = outputs['segment_features']  # torch.Tensor of shape (num_segments, 768)
```

### Advanced Usage

```python
# Adjust segmentation parameters
segmenter = Segmenter(
    mode="syllable",
    layer=8,                    # HuBERT layer to use
    sec_per_syllable=0.2,      # Target syllable duration
    merge_threshold=0.3,        # Similarity threshold for merging
    device="cuda"               # Use GPU if available
)

# Word segmentation with custom threshold
word_segmenter = Segmenter(
    mode="word",
    attn_threshold=0.25,        # CLS attention threshold
    layer=9
)
```

### Output Format

Both modes return a dictionary with:
- `segments`: List of (start, end) tuples in seconds
- `segment_features`: Tensor of segment-level features (num_segments, 768)
- `hidden_states`: Frame-level hidden states (num_frames, 768)
- `cls_attention`: CLS attention scores (word mode only)

## Model Checkpoints

The model includes two checkpoints optimized for different tasks:

| Checkpoint | Task | Layer | Algorithm |
|------------|------|-------|-----------|
| best_bundle.pth | Syllable | 8 | MinCut + Feature SSM |
| snapshot_20.pth | Word | 9 | CLS Attention |

## Technical Details

### Syllable Segmentation
1. Extract features from HuBERT layer 8
2. Compute feature self-similarity matrix
3. Apply MinCut dynamic programming algorithm
4. Merge similar adjacent segments

### Word Segmentation
1. Extract features from HuBERT layer 9
2. Get CLS token attention weights (per-head)
3. Find peaks in CLS attention
4. Use peaks as word boundaries

### Attention Implementation
- Uses `attn_implementation='eager'` for word mode (need attention weights)
- Uses `attn_implementation='sdpa'` for syllable mode (faster)
- No PyTorch source patching required (PyTorch 2.0+)

## Limitations

- Trained on English speech (SpokenCOCO)
- Performance varies by language and speaking style
- Word segmentation threshold may need tuning per dataset
- Requires 16kHz mono audio

## Citation

If you use this model, please cite the original papers:

### Syllable Segmentation
```bibtex
@inproceedings{{peng2023syllable,
  title={{Syllable Segmentation and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model}},
  author={{Peng, Puyuan and Li, Shang-Wen and Räsänen, Okko and Mohamed, Abdelrahman and Harwath, David}},
  booktitle={{Interspeech}},
  year={{2023}}
}}
```

### Word Discovery
```bibtex
@inproceedings{{peng2022word,
  title={{Word Discovery in Visually Grounded, Self-Supervised Speech Models}},
  author={{Peng, Puyuan and Harwath, David}},
  booktitle={{Interspeech}},
  year={{2022}}
}}
```

## Original Repositories

- Word Discovery: [jasonppy/word-discovery](https://github.com/jasonppy/word-discovery)
- Syllable Discovery: [jasonppy/syllable-discovery](https://github.com/jasonppy/syllable-discovery)
- Fork parent: [human-ai-lab/VG-HuBERT](https://github.com/human-ai-lab/VG-HuBERT)

**All credit for model architecture and training methodology goes to the original authors**: Puyuan Peng and David Harwath.

## License

BSD-3-Clause License - See LICENSE file

## Acknowledgments

- Original VG-HuBERT implementation: [syllable-discovery](https://github.com/jasonppy/syllable-discovery)
- Base model: [HuBERT](https://huggingface.co/facebook/hubert-base-ls960)
- Interface & packaging structure: [Sylber](https://github.com/Berkeley-Speech-Group/sylber) by Cho et al.

### Interface Design Citation

This package follows the simplified interface and distribution approach introduced by Sylber:

```bibtex
@article{{cho2024sylber,
  title={{Sylber: Syllabic Embedding Representation of Speech from Raw Audio}},
  author={{Cho, Cheol Jun and Lee, Nicholas and Gupta, Akshat and Agarwal, Dhruv and Chen, Ethan and Black, Alan W and Anumanchipalli, Gopala K}},
  journal={{arXiv preprint arXiv:2410.07168}},
  year={{2024}}
}}
```

## Model Card Authors

{repo_id.split('/')[0]}

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/VG-HuBERT).
"""
    return card


def upload_model(
    repo_id: str,
    model_path: str,
    token: str = None,
    private: bool = False
):
    """
    Upload VG-HuBERT model to HuggingFace Hub.
    
    Args:
        repo_id: Repository ID (e.g., "username/vg-hubert")
        model_path: Path to local model directory
        token: HuggingFace token (or set HF_TOKEN env var)
        private: Whether to create a private repository
    """
    api = HfApi(token=token)
    model_path = Path(model_path)
    
    # Map old filenames to new descriptive names
    file_mapping = {
        "best_bundle.pth": "vg-hubert-syllable.pth",
        "snapshot_20.pth": "vg-hubert-word.pth",
        "args.pkl": "args.pkl"
    }
    
    # Check required files exist (try both old and new names)
    for old_name, new_name in file_mapping.items():
        fpath = model_path / old_name
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    print(f"Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token
        )
        print(f"✓ Repository created/verified")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # Create and upload model card
    print("\nCreating model card...")
    model_card = create_model_card(repo_id)
    readme_path = "/tmp/vg_hubert_README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    print(f"Uploading README.md...")
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print("✓ Model card uploaded")
    
    # Upload model files with descriptive names
    print("\nUploading model files...")
    for old_name, new_name in file_mapping.items():
        fpath = model_path / old_name
        fsize_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {old_name} as {new_name} ({fsize_mb:.1f} MB)...")
        
        upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=new_name,  # Upload with new descriptive name
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"  ✓ {new_name} uploaded")
    
    # Create .gitattributes for LFS
    print("\nCreating .gitattributes for large files...")
    gitattributes = """*.pth filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
"""
    gitattributes_path = "/tmp/vg_hubert_gitattributes"
    with open(gitattributes_path, "w") as f:
        f.write(gitattributes)
    
    upload_file(
        path_or_fileobj=gitattributes_path,
        path_in_repo=".gitattributes",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print("✓ .gitattributes uploaded")
    
    print(f"\n✅ Model successfully uploaded to: https://huggingface.co/{repo_id}")
    print(f"\nUsers can now use:")
    print(f'  segmenter = Segmenter(model_ckpt="{repo_id}", mode="syllable")')


def main():
    parser = argparse.ArgumentParser(
        description="Upload VG-HuBERT model to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'username/vg-hubert')"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../findsylls/models/vg-hubert_3",
        help="Path to local model directory"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: HuggingFace token required. Set HF_TOKEN env var or use --token")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return 1
    
    try:
        upload_model(
            repo_id=args.repo_id,
            model_path=args.model_path,
            token=token,
            private=args.private
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
