"""
test_checkpoint_parity.py
=========================
Proves that the HuggingFace reimplementation of VG-HuBERT (vg_hubert/segmenter.py)
produces numerically identical features to what the original fairseq-based
AudioEncoder would produce, WITHOUT needing fairseq installed.

Two test tiers:
  1. Weight loading parity  – every tensor from the checkpoint appears with
     exact same values after _build_hf_state_dict + load_state_dict.
     The ONLY missing key should be ``masked_spec_embed``.
  2. Feature inference parity – a self-contained _ReferenceAudioEncoder
     loads the checkpoint directly and computes features using only standard
     PyTorch; its output must match the HF model output within atol=1e-4.

Skip conditions:
  - Checkpoints not found (env var VGHUBERT_CHECKPOINT_DIR or default sibling
    path ../../findsylls/models/vg-hubert_3/ relative to this file).
  - Test audio not found (../../findsylls/test_samples/SP20_117.wav).
"""

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Checkpoint / audio discovery
# ---------------------------------------------------------------------------

SYLLABLE_CHECKPOINT_NAME = "vg-hubert-syllable.pth"
WORD_CHECKPOINT_NAME = "vg-hubert-word.pth"

_THIS_DIR = Path(__file__).parent


def _find_checkpoint_dir() -> Optional[Path]:
    """Return Path to checkpoint directory, or None if not found."""
    # 1) Explicit env var
    env_dir = os.environ.get("VGHUBERT_CHECKPOINT_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    # 2) Default sibling path relative to this test file.
    # The test file lives at VG-HuBERT/vg_hubert/tests/; the findsylls repo is a
    # sibling of VG-HuBERT, so we need three levels up then into findsylls/.
    default = (_THIS_DIR / "../../../findsylls/models/vg-hubert_3/").resolve()
    if default.is_dir():
        return default

    return None


def _find_test_audio() -> Optional[Path]:
    candidate = (_THIS_DIR / "../../../findsylls/test_samples/SP20_117.wav").resolve()
    return candidate if candidate.is_file() else None


_CKPT_DIR = _find_checkpoint_dir()
_TEST_AUDIO = _find_test_audio()

_SKIP_CKPT = pytest.mark.skipif(
    _CKPT_DIR is None
    or not (_CKPT_DIR / SYLLABLE_CHECKPOINT_NAME).exists()
    or not (_CKPT_DIR / WORD_CHECKPOINT_NAME).exists(),
    reason=(
        f"VG-HuBERT checkpoints not found. Set VGHUBERT_CHECKPOINT_DIR or "
        f"place checkpoints at <repo>/findsylls/models/vg-hubert_3/"
    ),
)

_SKIP_AUDIO = pytest.mark.skipif(
    _TEST_AUDIO is None,
    reason="Test audio SP20_117.wav not found at expected path",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def syllable_ckpt_path():
    if _CKPT_DIR is None:
        pytest.skip("Checkpoint directory not found")
    p = _CKPT_DIR / SYLLABLE_CHECKPOINT_NAME
    if not p.exists():
        pytest.skip(f"Syllable checkpoint not found: {p}")
    return p


@pytest.fixture(scope="module")
def word_ckpt_path():
    if _CKPT_DIR is None:
        pytest.skip("Checkpoint directory not found")
    p = _CKPT_DIR / WORD_CHECKPOINT_NAME
    if not p.exists():
        pytest.skip(f"Word checkpoint not found: {p}")
    return p


@pytest.fixture(scope="module")
def test_audio():
    """Return a float32 mono audio tensor [T] at 16 kHz."""
    if _TEST_AUDIO is None:
        pytest.skip("Test audio SP20_117.wav not found")
    try:
        import torchaudio
        audio, sr = torchaudio.load(str(_TEST_AUDIO))
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(_TEST_AUDIO), dtype="float32")
        audio = torch.from_numpy(data)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

    if sr != 16000:
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        except Exception:
            import librosa
            audio_np = audio.numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            audio = torch.from_numpy(audio_np).unsqueeze(0)

    # Collapse to mono
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(0)
    elif audio.ndim > 1:
        audio = audio.squeeze(0)

    return audio.float()  # [T]


# ---------------------------------------------------------------------------
# _ReferenceAudioEncoder
# ---------------------------------------------------------------------------


class _ReferenceAudioEncoder:
    """
    Self-contained reimplementation of the original fairseq AudioEncoder,
    operating only with torch.nn.functional (no nn.Module subclassing).

    This mirrors the forward pass of the original model for
    args.extractor_mode="default" and args.feature_grad_mult=0.0:

      CNN → layer_norm → post_extract_proj → pos_conv → encoder_layer_norm
          → CLS injection → N transformer layers (post-norm)

    Only stores raw tensors extracted from the checkpoint; never calls
    nn.Module.forward() so the computation path is unambiguous.
    """

    # CNN configs: (out_channels, kernel, stride)
    CNN_CONFIGS = [
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ]

    def __init__(self, raw: dict, cls_token: torch.Tensor, n_layers: int = 12):
        """
        Parameters
        ----------
        raw : dict
            Flat state dict with *no* "audio_encoder." prefix.
        cls_token : Tensor, shape [1, 1, 768]
        n_layers : int
            Number of transformer layers stored in the checkpoint (default 12).
        """
        self.raw = {k: v.float() for k, v in raw.items()}
        self.cls_token = cls_token.float()
        self.n_layers = n_layers
        self.n_heads = 12
        self.head_dim = 64
        self.embed_dim = 768

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cnn_features(self, source: torch.Tensor) -> torch.Tensor:
        """source: [B, T_audio] or [B, 1, T_audio] → [B, T_feat, 512]"""
        r = self.raw
        x = source.float()
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        for i, (_, k, s) in enumerate(self.CNN_CONFIGS):
            w = r[f"feature_extractor.conv_layers.{i}.0.weight"]
            x = F.conv1d(x, w, stride=s)
            if i == 0:
                # Fp32GroupNorm: 512 groups, 512 channels (group size = 1)
                gn_w = r["feature_extractor.conv_layers.0.2.weight"]
                gn_b = r["feature_extractor.conv_layers.0.2.bias"]
                x = F.group_norm(x, 512, gn_w, gn_b)
            x = F.gelu(x)

        return x.transpose(1, 2)  # [B, T_feat, 512]

    def _pos_conv(self, features: torch.Tensor) -> torch.Tensor:
        """Apply weight-normed positional conv and add to features."""
        r = self.raw
        T_feat = features.shape[1]

        # Reconstruct effective weight from weight_g and weight_v
        wg = r["pos_conv.0.weight_g"]  # [1, 1, 128]
        wv = r["pos_conv.0.weight_v"]  # [768, 48, 128]
        pos_bias = r["pos_conv.0.bias"]  # [768]
        norm_wv = torch.norm(wv, p=2, dim=(0, 1), keepdim=True)
        eff_w = wg * wv / norm_wv  # [768, 48, 128]

        x_t = features.transpose(1, 2)  # [B, 768, T_feat]
        # kernel=128, padding=64 → output length = T_feat + 128 - 1; trim to T_feat
        pos_out = F.conv1d(x_t, eff_w, bias=pos_bias, padding=64, groups=16)
        pos_out = pos_out[:, :, :T_feat]
        pos_out = F.gelu(pos_out).transpose(1, 2)  # [B, T_feat, 768]
        return features + pos_out

    def _transformer_layer(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """One post-norm transformer layer (no dropout in eval mode)."""
        r = self.raw
        B, T, D = x.shape
        h, d = self.n_heads, self.head_dim

        # --- Self-attention ---
        q = (x @ r[f"encoder.layers.{i}.self_attn.q_proj.weight"].T
             + r[f"encoder.layers.{i}.self_attn.q_proj.bias"])
        k = (x @ r[f"encoder.layers.{i}.self_attn.k_proj.weight"].T
             + r[f"encoder.layers.{i}.self_attn.k_proj.bias"])
        v = (x @ r[f"encoder.layers.{i}.self_attn.v_proj.weight"].T
             + r[f"encoder.layers.{i}.self_attn.v_proj.bias"])

        # Reshape to multi-head
        q = q.view(B, T, h, d).transpose(1, 2)  # [B, h, T, d]
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        # Scaled dot-product attention (manual, no fused kernel ambiguity)
        scale = d ** -0.5
        attn_w = (q * scale) @ k.transpose(-2, -1)       # [B, h, T, T]
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_out = (attn_w @ v).transpose(1, 2).reshape(B, T, D)  # [B, T, D]

        # Output projection
        attn_out = (attn_out @ r[f"encoder.layers.{i}.self_attn.out_proj.weight"].T
                    + r[f"encoder.layers.{i}.self_attn.out_proj.bias"])

        # Post-norm (residual + layer_norm)
        x = F.layer_norm(
            x + attn_out,
            (D,),
            r[f"encoder.layers.{i}.self_attn_layer_norm.weight"],
            r[f"encoder.layers.{i}.self_attn_layer_norm.bias"],
        )

        # --- FFN ---
        ffn = F.gelu(x @ r[f"encoder.layers.{i}.fc1.weight"].T
                     + r[f"encoder.layers.{i}.fc1.bias"])
        ffn = ffn @ r[f"encoder.layers.{i}.fc2.weight"].T + r[f"encoder.layers.{i}.fc2.bias"]

        x = F.layer_norm(
            x + ffn,
            (D,),
            r[f"encoder.layers.{i}.final_layer_norm.weight"],
            r[f"encoder.layers.{i}.final_layer_norm.bias"],
        )
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, audio: torch.Tensor, tgt_layer: int) -> np.ndarray:
        """
        Parameters
        ----------
        audio : Tensor, shape [T] or [1, T]  (float32, 16 kHz)
        tgt_layer : int  (0-indexed transformer layer, e.g. 8 for syllable)

        Returns
        -------
        np.ndarray, shape [T_feat, 768]  (CLS token stripped)
        """
        r = self.raw
        with torch.no_grad():
            # Prepare source
            src = audio.float()
            if src.ndim == 1:
                src = src.unsqueeze(0)  # [1, T]

            # 1. CNN
            features = self._cnn_features(src)  # [1, T_feat, 512]

            # 2. Fp32LayerNorm on 512-dim CNN output
            features = F.layer_norm(
                features,
                (512,),
                r["layer_norm.weight"],
                r["layer_norm.bias"],
            )

            # 3. Post-extract projection 512 → 768
            features = (features @ r["post_extract_proj.weight"].T
                        + r["post_extract_proj.bias"])  # [1, T_feat, 768]

            # 4. Positional conv
            features = self._pos_conv(features)  # [1, T_feat, 768]

            # 5. Encoder layer norm (applied BEFORE transformer since layer_norm_first=False)
            features = F.layer_norm(
                features,
                (self.embed_dim,),
                r["encoder.layer_norm.weight"],
                r["encoder.layer_norm.bias"],
            )

            # 6. CLS token injection
            B = features.shape[0]
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            x = torch.cat([cls, features], dim=1)  # [1, T_feat+1, 768]

            # 7. Transformer layers up to and including tgt_layer
            for i in range(tgt_layer + 1):
                x = self._transformer_layer(x, i)

        # Strip CLS (position 0), return audio frames only
        return x[0, 1:].numpy()  # [T_feat, 768]


# ---------------------------------------------------------------------------
# Helper: build HF model from checkpoint
# ---------------------------------------------------------------------------


def _build_hf_model(ckpt_path: Path):
    """
    Load a VG-HuBERT .pth checkpoint into a HuggingFace HubertModel.

    Returns a dict with:
      - model          : fully initialised model (with CLS wrapper if applicable)
      - pre_wrap_state : model.state_dict() BEFORE _HubertEncoderWithCLS wrapping
      - cls_token      : Tensor or None
      - state_dict     : raw checkpoint sub-dict (possibly with audio_encoder. prefix)
      - cls_key        : string key name for CLS token within state_dict
      - hf_state       : remapped HuggingFace state dict
      - missing        : list of missing keys from load_state_dict
      - unexpected     : list of unexpected keys from load_state_dict
    """
    from transformers import HubertModel

    from vg_hubert.segmenter import (
        _HubertEncoderWithCLS,
        _build_hf_state_dict,
        _unpack_checkpoint,
    )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict, cls_key = _unpack_checkpoint(checkpoint)
    cls_token = state_dict.get(cls_key, None)

    hf_state = _build_hf_state_dict(state_dict, cls_key)

    model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960",
        attn_implementation="sdpa",
    )
    missing, unexpected = model.load_state_dict(hf_state, strict=False)

    # Capture state BEFORE wrapping; the CLS wrapper renames keys (encoder.base.*)
    pre_wrap_state = {k: v.clone() for k, v in model.state_dict().items()}

    if cls_token is not None:
        model.encoder = _HubertEncoderWithCLS(model.encoder, cls_token)

    model.eval()
    return {
        "model": model,
        "pre_wrap_state": pre_wrap_state,
        "cls_token": cls_token,
        "state_dict": state_dict,
        "cls_key": cls_key,
        "hf_state": hf_state,
        "missing": missing,
        "unexpected": unexpected,
    }


def _extract_raw(state_dict: dict, cls_key: str) -> dict:
    """Strip 'audio_encoder.' prefix if present, returning flat raw dict."""
    has_prefix = cls_key == "audio_encoder.cls_token"
    raw = {}
    for k, v in state_dict.items():
        if has_prefix:
            if k.startswith("audio_encoder."):
                raw[k[len("audio_encoder."):]] = v
        else:
            raw[k] = v
    return raw


# ---------------------------------------------------------------------------
# Tier 1: Weight loading parity
# ---------------------------------------------------------------------------


def _check_weight_parity(hf_state: dict, pre_wrap_state: dict, label: str) -> None:
    """
    Assert every key in hf_state appears with identical values in pre_wrap_state.

    pos_conv weight_g / weight_v are stored as parametrizations in the HF model
    state_dict, so we remap those keys before comparing.
    """
    def _resolve(k: str) -> str:
        if k == "encoder.pos_conv_embed.conv.weight_g":
            return "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        if k == "encoder.pos_conv_embed.conv.weight_v":
            return "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        return k

    mismatches = []
    for k, v in hf_state.items():
        mk = _resolve(k)
        if mk not in pre_wrap_state:
            mismatches.append(f"key not in model_state: {k} -> {mk}")
            continue
        if not torch.allclose(pre_wrap_state[mk].float(), v.float()):
            max_diff = (pre_wrap_state[mk].float() - v.float()).abs().max().item()
            mismatches.append(f"{k}: max_diff={max_diff:.2e}")

    assert not mismatches, (
        f"{label} weight loading mismatches:\n" + "\n".join(mismatches)
    )


@_SKIP_CKPT
def test_weight_loading_syllable_checkpoint(syllable_ckpt_path):
    """Every remapped weight in the syllable checkpoint matches the HF model."""
    result = _build_hf_model(syllable_ckpt_path)
    _check_weight_parity(
        result["hf_state"], result["pre_wrap_state"], "Syllable checkpoint"
    )


@_SKIP_CKPT
def test_weight_loading_word_checkpoint(word_ckpt_path):
    """Every remapped weight in the word checkpoint matches the HF model."""
    result = _build_hf_model(word_ckpt_path)
    _check_weight_parity(
        result["hf_state"], result["pre_wrap_state"], "Word checkpoint"
    )


@_SKIP_CKPT
def test_only_masked_spec_embed_is_missing(syllable_ckpt_path):
    """load_state_dict should report only masked_spec_embed as missing."""
    result = _build_hf_model(syllable_ckpt_path)
    missing = result["missing"]
    unexpected = result["unexpected"]
    assert missing == ["masked_spec_embed"], (
        f"Expected only ['masked_spec_embed'] to be missing, got: {missing}"
    )
    assert unexpected == [], (
        f"Expected no unexpected keys, got: {unexpected}"
    )


@_SKIP_CKPT
def test_cls_token_loaded(syllable_ckpt_path):
    """Segmenter.has_cls_token is True and CLS token matches checkpoint."""
    from vg_hubert.segmenter import (
        _HubertEncoderWithCLS,
        _unpack_checkpoint,
    )

    checkpoint = torch.load(str(syllable_ckpt_path), map_location="cpu")
    state_dict, cls_key = _unpack_checkpoint(checkpoint)
    expected_cls = state_dict[cls_key]

    result = _build_hf_model(syllable_ckpt_path)
    cls_token = result["cls_token"]

    assert cls_token is not None, "cls_token should not be None for syllable checkpoint"
    assert torch.allclose(cls_token.float(), expected_cls.float()), (
        "CLS token in model does not match checkpoint"
    )


# ---------------------------------------------------------------------------
# Tier 2: Feature inference parity
# ---------------------------------------------------------------------------


@_SKIP_CKPT
@_SKIP_AUDIO
def test_feature_parity_syllable_layer8(syllable_ckpt_path, test_audio):
    """HF model features match _ReferenceAudioEncoder at layer 8 (syllable mode)."""
    tgt_layer = 8

    # --- HF model features ---
    result = _build_hf_model(syllable_ckpt_path)
    model = result["model"]
    cls_token = result["cls_token"]
    state_dict = result["state_dict"]
    cls_key = result["cls_key"]
    audio_input = test_audio.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        out = model(
            input_values=audio_input,
            output_hidden_states=True,
            return_dict=True,
        )
    # hidden_states[tgt_layer+1] = output of transformer layer tgt_layer (0-indexed)
    hf_features = out.hidden_states[tgt_layer + 1][0, 1:].float().numpy()

    # --- Reference encoder features ---
    raw = _extract_raw(state_dict, cls_key)
    ref_enc = _ReferenceAudioEncoder(raw, cls_token)
    ref_features = ref_enc.extract(test_audio, tgt_layer=tgt_layer)

    assert ref_features.shape == hf_features.shape, (
        f"Shape mismatch: ref={ref_features.shape} hf={hf_features.shape}"
    )
    np.testing.assert_allclose(
        ref_features,
        hf_features,
        atol=1e-4,
        rtol=0,
        err_msg=(
            f"Feature parity failed for syllable checkpoint at layer {tgt_layer}. "
            f"Max abs diff: {np.abs(ref_features - hf_features).max():.2e}"
        ),
    )


@_SKIP_CKPT
@_SKIP_AUDIO
def test_feature_parity_word_layer9(word_ckpt_path, test_audio):
    """HF model features match _ReferenceAudioEncoder at layer 9 (word mode)."""
    tgt_layer = 9

    # --- HF model features ---
    result = _build_hf_model(word_ckpt_path)
    model = result["model"]
    cls_token = result["cls_token"]
    state_dict = result["state_dict"]
    cls_key = result["cls_key"]
    audio_input = test_audio.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        out = model(
            input_values=audio_input,
            output_hidden_states=True,
            return_dict=True,
        )
    hf_features = out.hidden_states[tgt_layer + 1][0, 1:].float().numpy()

    # --- Reference encoder features ---
    raw = _extract_raw(state_dict, cls_key)
    ref_enc = _ReferenceAudioEncoder(raw, cls_token)
    ref_features = ref_enc.extract(test_audio, tgt_layer=tgt_layer)

    assert ref_features.shape == hf_features.shape, (
        f"Shape mismatch: ref={ref_features.shape} hf={hf_features.shape}"
    )
    np.testing.assert_allclose(
        ref_features,
        hf_features,
        atol=1e-4,
        rtol=0,
        err_msg=(
            f"Feature parity failed for word checkpoint at layer {tgt_layer}. "
            f"Max abs diff: {np.abs(ref_features - hf_features).max():.2e}"
        ),
    )
