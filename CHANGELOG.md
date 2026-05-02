# Changelog

All notable changes to VG-HuBERT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-05-02

### Fixed
- **CRITICAL**: Complete rewrite of VG-HuBERT → HuggingFace weight remapping (`_remap_vghubert_keys`)
  - Previous version produced silently wrong features; CNN, layer-norm, pos-conv, attention, and FFN keys
    were all mapped incorrectly, causing the HF model to load random HuBERT weights instead of VG-HuBERT weights
  - Correct mappings now cover: CNN conv weights, post-CNN layer norm (512-dim), post-extract projection,
    pos-conv (`pos_conv.0.*` → `encoder.pos_conv_embed.conv.*`), attention QKV/out proj, self-attn layer norm,
    FFN intermediate/output dense, and encoder layer norm
- **CRITICAL**: `_unpack_checkpoint` now correctly handles both checkpoint formats:
  - `dual_encoder` format (syllable `.pth`): sub-dict keyed with `audio_encoder.` prefix
  - `audio_encoder` format (word `.pth`): sub-dict keyed without prefix
- **CRITICAL**: `_HubertEncoderWithCLS` correctly injects the learned CLS token between positional encoding
  and transformer layers, matching the original `use_audio_cls_token=True` behavior
- Fixed layer indexing convention: `hidden_states[layer+1]` = output of transformer layer `layer` (0-indexed),
  so `layer=8` (syllable) correctly reads `hidden_states[9]`, and `layer=9` (word) reads `hidden_states[10]`

### Added
- `tests/test_checkpoint_parity.py`: two-tier numerical parity test suite
  - Tier 1 (weight loading): every tensor from the checkpoint matches the loaded HF model
  - Tier 2 (inference parity): HF model features match a pure-PyTorch reference encoder within atol=1e-4
    (max observed error ~5.6e-6); verifies numerical equivalence to the original fairseq AudioEncoder
    without requiring fairseq to be installed

## [1.1.0] - 2026-02-26

### Fixed
- 🐛 **CRITICAL FIX**: Attention implementation now correctly determined by `segmentation_method` instead of `mode`
  - Previously: `mode='syllable'` forced SDPA attention (no attention weights), breaking CLS segmentation
  - Now: `segmentation_method='CLS'` forces eager attention, enabling syllable-level CLS segmentation
  - This allows using the syllable checkpoint (layer 8) with CLS attention-based segmentation
- Effect: CLS segmentation now works correctly for both syllable-level and word-level tasks

### Changed
- Attention implementation selection now based on segmentation needs rather than granularity mode
- Improved composability: checkpoint selection (mode) is now orthogonal to segmentation method

## [1.0.0] - 2024-12-26

### Added
- 🚀 **40x faster MinCut algorithm**: Integrated optimized implementation from SyllableLM (Baade et al., 2024)
- 🔧 **MinCutMerge post-processing**: Prevents over-segmentation by merging segments based on similarity threshold
- 🤗 **HuggingFace Hub integration**: Automatic model download from Hub with `model_ckpt="hjvm/VG-HuBERT"`
- 🍎 **Apple Silicon support**: Native MPS device acceleration for M1/M2/M3 Macs
- 📦 **PyPI distribution**: Install via `pip install vg-hubert`
- 🧹 **No fairseq dependency**: Removed complex fairseq requirement for inference
- **Sylber-compatible API**: Drop-in replacement with `Segmenter` class
- **Flexible segmentation methods**: Support for both `featSSM` and `CLS` MinCut inputs
- **Comprehensive validation**: Validation notebook with findsylls evaluation metrics

### Changed
- Updated to PyTorch ≥2.0 (no more multi_head_attention_forward patching required)
- Refactored segmentation pipeline for cleaner API
- Improved documentation with usage examples and HuggingFace Hub instructions
- Optimized memory usage for long audio files

### Fixed
- Fixed boundary extraction in MinCutMerge to match original paper implementation
- Fixed device handling for CPU/GPU/MPS
- Fixed audio loading to handle various sample rates and formats
- Corrected similarity matrix computation for more accurate segmentation

### Security
- Removed hardcoded HuggingFace tokens from demo files
- Updated token handling to use environment variables

## [Unreleased]

### Planned
- [ ] Support for batch processing multiple audio files
- [ ] Add speaker diarization support
- [ ] Integrate with more audio backends (torchaudio, ffmpeg)
- [ ] Add streaming mode for real-time segmentation
- [ ] Benchmark suite for performance tracking
- [ ] Training scripts refactoring
- [ ] Model quantization for faster inference

---

**Links:**
- [PyPI Package](https://pypi.org/project/vg-hubert/)
- [HuggingFace Model](https://huggingface.co/hjvm/VG-HuBERT)
- [GitHub Repository](https://github.com/human-ai-lab/VG-HuBERT)

**Original Papers:**
- Word Discovery: [Peng et al., 2022 (CVPR)](https://arxiv.org/abs/2203.15081)
- Syllable Discovery: [Peng et al., 2023 (Interspeech)](https://www.isca-speech.org/archive/interspeech_2023/peng23_interspeech.html)
- MinCut Optimization: [Baade et al., 2024 (SyllableLM)](https://arxiv.org/abs/2406.xxxxx)
