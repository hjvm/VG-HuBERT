# Changelog

All notable changes to VG-HuBERT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-11-05

### Added
- Image-space embeddings from the syllable checkpoint's audio-to-image head (`audio_cls_token_proj`,
  768 → 1536 → 2048), which the HuggingFace loader now also loads (no new Hub files needed)
  - `Segmenter.embed_utterance(...)`: utterance CLS at layer 11 through the image head, as in training
  - `Segmenter.embed_spans(spans, ..., restrict="all" | "last_k", k=1)`: one image-space vector per
    (start_s, end_s) span; the whole utterance is encoded and only the CLS attention is restricted to the span
  - `Segmenter(...)(wav, image=True)`: opt-in `utterance_image` and `segment_image` keys alongside the
    unchanged acoustic output; span vectors come from one extra pass with one CLS copy per span that the
    frames cannot attend to, so frame features are unchanged
- `test_mps_matches_cpu` and image-embedding parity tests, on real LibriSpeech utterances with
  segmenter-found spans

### Changed
- The word checkpoint (no image head) now warns at load and on each image request, and returns `None`
  for image outputs instead of raising

### Fixed
- Wrong outputs on Apple MPS for short inputs (<= 7 frames, ~0.16 s): torch 2.8's MPS
  scaled_dot_product_attention is incorrect for non-contiguous q/k/v at <= 8 tokens. The segmenter now uses
  eager attention on MPS (`Segmenter._attn_impl`); CPU and CUDA keep SDPA
- `vg_hubert.__version__` was stale (1.0.0); now matches the package version

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
