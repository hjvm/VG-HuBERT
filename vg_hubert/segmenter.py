"""
VG-HuBERT Segmenter - Simplified Interface

This module provides a Sylber-like interface for VG-HuBERT:
    from vg_hubert import Segmenter
    segmenter = Segmenter()  # That's it!

Interface design inspired by Sylber (Cho et al., 2024):
    https://github.com/Berkeley-Speech-Group/sylber
    
Handles all complexity internally:
- Model download from HuggingFace Hub
- Both word and syllable segmentation
- MinCut algorithm integration
- Native PyTorch attention (no patching required)
"""

import re
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from typing import Dict, Union, Optional, List, Tuple
from pathlib import Path
import logging
import os
import pickle

logger = logging.getLogger(__name__)


def _remap_vghubert_keys(key: str) -> Optional[str]:
    """
    Map a VG-HuBERT audio_encoder state-dict key to the equivalent HuggingFace HuBERT key.

    Returns None for keys that have no HuBERT equivalent (cls_token, project_q, etc.).
    """
    SKIP_PREFIXES = ('cls_token', 'mask_emb', 'project_q', 'project_k', 'final_proj',
                     'masked_spec_embed')
    if any(key == p or key.startswith(p + '.') for p in SKIP_PREFIXES):
        return None

    new_key = key

    # Positional convolution: pos_conv.0.X → encoder.pos_conv_embed.conv.X
    new_key = re.sub(r'^pos_conv\.0\.', 'encoder.pos_conv_embed.conv.', new_key)
    # Feature projection: post_extract_proj.X → feature_projection.projection.X
    new_key = re.sub(r'^post_extract_proj\.', 'feature_projection.projection.', new_key)
    # Post-CNN layer norm (512-dim, fairseq): layer_norm.X → feature_projection.layer_norm.X
    new_key = re.sub(r'^layer_norm\.', 'feature_projection.layer_norm.', new_key)
    # Feature extractor CNN: conv_layers.N.0.weight → conv_layers.N.conv.weight (fairseq style)
    new_key = re.sub(
        r'^(feature_extractor\.conv_layers\.\d+)\.0\.weight$',
        r'\1.conv.weight',
        new_key,
    )
    # First conv layer GroupNorm / LayerNorm: conv_layers.0.2.X → conv_layers.0.layer_norm.X
    new_key = re.sub(
        r'^(feature_extractor\.conv_layers\.0)\.2\.(weight|bias)$',
        r'\1.layer_norm.\2',
        new_key,
    )
    # Transformer attention: .self_attn. → .attention.
    new_key = new_key.replace('.self_attn.', '.attention.')
    # Attention layer norm: .self_attn_layer_norm. → .layer_norm.
    new_key = new_key.replace('.self_attn_layer_norm.', '.layer_norm.')
    # FFN: .fc1. → .feed_forward.intermediate_dense.  ;  .fc2. → .feed_forward.output_dense.
    new_key = new_key.replace('.fc1.', '.feed_forward.intermediate_dense.')
    new_key = new_key.replace('.fc2.', '.feed_forward.output_dense.')
    return new_key


def _unpack_checkpoint(checkpoint: dict):
    """
    Normalize VG-HuBERT checkpoint to a flat state-dict and the cls_token key name.

    Two formats exist in the wild:
    - dual_encoder format (syllable .pth): top key 'dual_encoder', values keyed
      as 'audio_encoder.cls_token', 'audio_encoder.feature_extractor.*', etc.
    - audio_encoder format (word .pth): top key 'audio_encoder', values keyed
      as 'cls_token', 'feature_extractor.*', etc. (no prefix).

    Returns:
        (state_dict, cls_token_key) where state_dict is the unpacked sub-dict
        and cls_token_key is the key name for the CLS token within state_dict.
    """
    if "dual_encoder" in checkpoint:
        # Keys have 'audio_encoder.' prefix inside this sub-dict
        return checkpoint["dual_encoder"], "audio_encoder.cls_token"
    elif "audio_encoder" in checkpoint and isinstance(checkpoint["audio_encoder"], dict):
        # Keys have no prefix inside this sub-dict
        return checkpoint["audio_encoder"], "cls_token"
    else:
        # Flat checkpoint — assume keys have no 'audio_encoder.' prefix
        return checkpoint, "cls_token"


def _build_hf_state_dict(state_dict: dict, cls_key: str) -> dict:
    """
    Build HuggingFace HuBERT state_dict from a VG-HuBERT audio_encoder sub-dict.

    Handles both prefixed ('audio_encoder.X') and unprefixed ('X') key formats.
    """
    has_prefix = cls_key == "audio_encoder.cls_token"
    prefix = "audio_encoder." if has_prefix else ""

    hf_state = {}
    for key, value in state_dict.items():
        if has_prefix:
            if not key.startswith("audio_encoder."):
                continue
            stripped = key[len("audio_encoder."):]
        else:
            stripped = key

        new_key = _remap_vghubert_keys(stripped)
        if new_key is not None:
            hf_state[new_key] = value

    return hf_state


class _HubertEncoderWithCLS(nn.Module):
    """
    Wraps HubertEncoder to inject a learned CLS token between positional encoding
    and the transformer layers, restoring original VG-HuBERT use_audio_cls_token=True behavior.

    The CLS token is prepended AFTER positional encoding (no positional encoding for CLS),
    matching the original AudioEncoder implementation.
    """

    def __init__(self, base_encoder: nn.Module, cls_token: torch.Tensor):
        super().__init__()
        self.base = base_encoder
        # Store as non-trainable parameter so it moves with .to(device) calls.
        self.cls_token = nn.Parameter(cls_token.detach().clone(), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        from transformers.modeling_outputs import BaseModelOutput

        # Zero out padded positions before positional encoding (audio frames only).
        if attention_mask is not None:
            expand_attn = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attn] = 0

        # Apply positional encoding to AUDIO FRAMES (CLS token gets no pos encoding).
        position_embeddings = self.base.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.base.layer_norm(hidden_states)
        hidden_states = self.base.dropout(hidden_states)

        # Prepend CLS token.
        batch_size = hidden_states.size(0)
        cls = self.cls_token.expand(batch_size, 1, hidden_states.size(-1))
        hidden_states = torch.cat([cls, hidden_states], dim=1)

        # Extend padding mask for CLS position (CLS is always unmasked).
        if attention_mask is not None:
            cls_col = torch.ones(
                batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([cls_col, attention_mask], dim=1)

        # Build full additive attention mask (handles T+1 CLS-augmented sequence).
        attention_mask_full = self.base._update_full_mask(attention_mask, hidden_states)

        # Run transformer layers.
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.base.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask_full,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Segmenter:
    """
    VG-HuBERT Segmenter with Sylber-like interface.
    
    Usage (as simple as Sylber):
        >>> from vg_hubert import Segmenter
        >>> segmenter = Segmenter()  # Downloads model automatically
        >>> outputs = segmenter("audio.wav")
        >>> # outputs contains 'segments', 'features', 'hidden_states'
    
    Args:
        model_ckpt: Model checkpoint. Options:
                   - "vg-hubert" (default): Downloads from HuggingFace Hub
                   - Local path to model directory
        mode: Segmentation mode:
             - "syllable": MinCut-based syllable segmentation (default)
             - "word": Attention-based word segmentation
        layer: Which HuBERT layer to use (default: 8 for syllables, 9 for words)
        device: 'cuda', 'mps', or 'cpu' (default: 'cuda', auto-falls back to mps/cpu)
        sec_per_syllable: Target syllable duration for MinCut (default: 0.2)
        merge_threshold: Similarity threshold for merging segments (default: 0.3)
                        Set to None to disable MinCutMerge post-processing
        min_segment_frames: Filter segments with ≤ this many frames (default: 2)
        attn_threshold: Attention threshold for word segmentation (default: 0.7)
    
    Examples:
        >>> # Syllable segmentation (like Sylber)
        >>> segmenter = Segmenter()
        >>> outputs = segmenter("audio.wav")
        >>> for start, end in outputs['segments']:
        ...     print(f"Syllable: {start:.2f}s - {end:.2f}s")
        
        >>> # Word segmentation
        >>> segmenter = Segmenter(mode="word")
        >>> outputs = segmenter("audio.wav")
        >>> for start, end in outputs['segments']:
        ...     print(f"Word: {start:.2f}s - {end:.2f}s")
    """
    
    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        mode: str = "syllable",
        layer: Optional[int] = None,
        device: str = "cuda",
        sec_per_syllable: Optional[float] = None,
        merge_threshold: Optional[float] = None,
        min_segment_frames: int = 2,
        attn_threshold: float = 0.90,
        checkpoint_file: Optional[str] = None,  # Override which .pth file to use
        segmentation_method: Optional[str] = None,  # "featSSM" or "CLS" (None = auto based on mode)
        use_optimized_mincut: bool = True,  # Use optimized MinCut (SyllableLM) vs original
        **kwargs
    ):
        """
        Initialize VG-HuBERT Segmenter.

        Layer convention: layer numbers follow the fairseq / paper convention (0-indexed
        transformer layer). Internally, HuggingFace hidden_states[layer+1] is used so
        that hidden_states[layer+1] == output of transformer layer `layer`.

        Following syllable-discovery / word-discovery repo recommendations:
        - Syllable mode: best_bundle.pth, layer 8 (fairseq tgt_layer=8), sec_per_syllable=0.2
        - Word mode: snapshot_20.pth, layer 9 (fairseq tgt_layer=9), CLS attention
        """
        self.mode = mode
        
        # Set defaults based on mode (from syllable-discovery repo)
        if mode == "syllable":
            self.layer = layer if layer is not None else 8
            self.sec_per_syllable = sec_per_syllable if sec_per_syllable is not None else 0.2
            self.merge_threshold = merge_threshold if merge_threshold is not None else 0.3
            self.checkpoint_file = checkpoint_file if checkpoint_file is not None else "vg-hubert-syllable.pth"
            # Default: featSSM (MinCut) for syllables
            self.segmentation_method = segmentation_method if segmentation_method is not None else "featSSM"
        elif mode == "word":
            self.layer = layer if layer is not None else 9
            self.sec_per_syllable = sec_per_syllable if sec_per_syllable is not None else 0.2
            self.merge_threshold = merge_threshold if merge_threshold is not None else 0.3
            self.checkpoint_file = checkpoint_file if checkpoint_file is not None else "vg-hubert-word.pth"
            # Default: CLS attention for words
            self.segmentation_method = segmentation_method if segmentation_method is not None else "CLS"
        else:
            raise ValueError(f"mode must be 'syllable' or 'word', got {mode}")
        
        if self.segmentation_method not in ["featSSM", "CLS"]:
            raise ValueError(f"segmentation_method must be 'featSSM' or 'CLS', got {self.segmentation_method}")
        
        self.min_segment_frames = min_segment_frames
        self.attn_threshold = attn_threshold
        self.use_optimized_mincut = use_optimized_mincut
        
        # Device setup: try CUDA -> MPS -> CPU
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("CUDA not available, using MPS (Apple Silicon GPU)")
                self.device = "mps"
            else:
                logger.warning("CUDA not available, using CPU")
                self.device = "cpu"
        else:
            self.device = device
        
        # Auto-detect model path if not provided
        if model_ckpt is None:
            model_ckpt = "hjvm/VG-HuBERT" # Will try to download from Hub
        
        # Load model
        self._load_model(model_ckpt)
    
    def _load_model(self, model_ckpt: str):
        """Load VG-HuBERT model."""
        from transformers import HubertModel
        
        # Check if it's a HuggingFace model or local path
        if os.path.isdir(model_ckpt):
            model_path = Path(model_ckpt)
            logger.info(f"Loading model from local path: {model_path}")
            
            # Load from legacy format - use appropriate checkpoint for mode
            checkpoint_path = model_path / self.checkpoint_file
            if not checkpoint_path.exists():
                # Fallback to old names for backward compatibility
                old_name = "best_bundle.pth" if self.mode == "syllable" else "snapshot_20.pth"
                logger.warning(f"{self.checkpoint_file} not found, trying legacy name {old_name}")
                checkpoint_path = model_path / old_name
                
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            logger.info(f"Loading checkpoint: {checkpoint_path.name}")
            
            args_path = model_path / "args.pkl"
            if not args_path.exists():
                raise FileNotFoundError(f"Model args not found: {args_path}")
            
            # Load args
            with open(args_path, "rb") as f:
                args = pickle.load(f)
            
            # Initialize model using transformers HuBERT (avoids fairseq dependency)
            # Use eager attention for CLS segmentation (needs attention weights)
            # SDPA is faster but can't output attention weights
            attn_impl = 'eager' if self.segmentation_method == 'CLS' else 'sdpa'
            self.model = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                attn_implementation=attn_impl
            )
            logger.info(f"Using {attn_impl} attention implementation for {self.segmentation_method} segmentation")
            
            # Load VG-HuBERT weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict, cls_key = _unpack_checkpoint(checkpoint)
            cls_token = state_dict.get(cls_key, None)

            # Map all audio_encoder keys to HuggingFace HuBERT equivalents
            audio_encoder_state = _build_hf_state_dict(state_dict, cls_key)

            missing_keys, unexpected_keys = self.model.load_state_dict(audio_encoder_state, strict=False)
            logger.info(
                f"Loaded {len(audio_encoder_state)} VG-HuBERT weights from {checkpoint_path.name} "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
            )

            # Inject CLS token into the encoder when the checkpoint has one.
            self.has_cls_token = cls_token is not None
            if cls_token is not None:
                self.model.encoder = _HubertEncoderWithCLS(
                    self.model.encoder, cls_token.to(self.device)
                )
                logger.info(f"Injected CLS token into HuBERT encoder (shape: {cls_token.shape})")
        
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Downloading VG-HuBERT from HuggingFace Hub: {model_ckpt}")
            
            # Download checkpoint and args
            try:
                # Use appropriate checkpoint for mode
                checkpoint_path = hf_hub_download(
                    repo_id=model_ckpt,
                    filename=self.checkpoint_file
                )
                args_path = hf_hub_download(
                    repo_id=model_ckpt,
                    filename="args.pkl"
                )
                
                logger.info(f"Downloaded {self.checkpoint_file} and args.pkl")
                
                # Load model
                with open(args_path, "rb") as f:
                    args = pickle.load(f)
                
                attn_impl = 'eager' if self.segmentation_method == 'CLS' else 'sdpa'
                self.model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960",
                    attn_implementation=attn_impl
                )
                logger.info(f"Using {attn_impl} attention for {self.segmentation_method} segmentation")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict, cls_key = _unpack_checkpoint(checkpoint)
                cls_token = state_dict.get(cls_key, None)

                audio_encoder_state = _build_hf_state_dict(state_dict, cls_key)
                missing_keys, unexpected_keys = self.model.load_state_dict(audio_encoder_state, strict=False)
                logger.info(
                    f"Loaded {len(audio_encoder_state)} VG-HuBERT weights from Hub "
                    f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
                )

                # Inject CLS token into encoder
                self.has_cls_token = cls_token is not None
                if cls_token is not None:
                    self.model.encoder = _HubertEncoderWithCLS(
                        self.model.encoder, cls_token.to(self.device)
                    )
                    logger.info(f"Injected CLS token (shape: {cls_token.shape})")
                
            except Exception as e:
                logger.warning(
                    f"Could not download VG-HuBERT from Hub: {e}\n"
                    f"Falling back to base HuBERT model (facebook/hubert-base-ls960).\n"
                    f"For full VG-HuBERT functionality, download the model from:\n"
                    f"https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar"
                )
                # Use base HuBERT as fallback for demonstration
                attn_impl = 'eager' if self.segmentation_method == 'CLS' else 'sdpa'
                self.model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960",
                    attn_implementation=attn_impl
                )
                self.has_cls_token = False
                logger.info(f"Using base HuBERT model ({attn_impl} attention) as fallback")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def __call__(
        self,
        wav_file: Optional[str] = None,
        wav: Optional[np.ndarray] = None,
        in_second: bool = True
    ) -> Dict:
        """
        Segment audio (Sylber-compatible interface).
        
        Args:
            wav_file: Path to audio file
            wav: Audio waveform array (16kHz, mono)
            in_second: If True, return times in seconds; if False, in frames
        
        Returns:
            Dictionary containing:
            - 'segments': List of (start, end) tuples (syllables or words)
            - 'segment_features': Segment-level features
            - 'hidden_states': Frame-level hidden states
            - 'cls_attention': CLS attention scores (word mode only)
        """
        # Load audio
        if wav_file is not None:
            audio, sr = sf.read(wav_file, dtype='float32')
        elif wav is not None:
            audio = wav
            sr = 16000
        else:
            raise ValueError("Must provide either wav_file or wav")
        
        # Ensure 16kHz mono
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        audio_len_sec = len(audio) / sr
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(
                input_values=audio_tensor,
                output_hidden_states=True,
                return_dict=True
            )

            # hidden_states[layer+1] = output of transformer layer `layer` (0-indexed).
            # This matches fairseq tgt_layer=layer convention used in word/syllable-discovery.
            features = outputs.hidden_states[self.layer + 1][0].cpu().float().numpy()

            # Strip CLS token (position 0) — only present when checkpoint had use_audio_cls_token=True.
            if self.has_cls_token:
                features = features[1:]
        
        # Calculate seconds per frame
        spf = audio_len_sec / features.shape[0]
        
        # Segment based on segmentation_method (not mode)
        if self.segmentation_method == "featSSM":
            result = self._segment_featssm(features, spf, audio_len_sec, in_second)
        elif self.segmentation_method == "CLS":
            result = self._segment_cls(audio_tensor, features, spf, audio_len_sec, in_second)
        else:
            raise ValueError(f"Unknown segmentation_method: {self.segmentation_method}")
        
        # Add hidden states
        result['hidden_states'] = features
        
        return result
    
    def _segment_featssm(
        self,
        features: np.ndarray,
        spf: float,
        audio_len_sec: float,
        in_second: bool
    ) -> Dict:
        """Segment using featSSM (MinCut) algorithm with optional MinCutMerge post-processing."""
        # Estimate number of segments
        num_segments = max(1, int(np.ceil(audio_len_sec / self.sec_per_syllable)))
        K = num_segments + 1  # Number of boundaries
        
        # Apply MinCut with optional merging
        try:
            from .mincut import segment_with_mincut
            seg_boundary_frames, ssm = segment_with_mincut(
                features=features,
                K=K,
                merge_threshold=self.merge_threshold,  # None = no merging
                min_segment_frames=self.min_segment_frames,
                use_optimized=self.use_optimized_mincut  # Choose algorithm version
            )
        except Exception as e:
            # Fallback: uniform segmentation
            logger.warning(f"MinCut failed ({e}), using uniform segmentation")
            seg_boundary_frames = np.linspace(0, features.shape[0], K).astype(int)
        
        # Create segments
        seg_pairs = [[l, r] for l, r in zip(seg_boundary_frames[:-1], seg_boundary_frames[1:])]
        
        # Convert to time
        if in_second:
            segments = [[l * spf, r * spf] for l, r in seg_pairs]
        else:
            segments = seg_pairs
        
        # Extract segment features
        segment_features = torch.stack([
            torch.from_numpy(features[l:r].mean(0)) for l, r in seg_pairs
        ])
        
        return {
            'segments': segments,
            'segment_features': segment_features
        }
    
    def _segment_cls(
        self,
        audio_tensor: torch.Tensor,
        features: np.ndarray,
        spf: float,
        audio_len_sec: float,
        in_second: bool
    ) -> Dict:
        """
        Segment using CLS token attention — canonical word-discovery algorithm.

        Implements the reference algorithm from save_seg_feats.py (Peng & Harwath 2022):
        1. Extract CLS row: attention[:, 0, 1:] → [n_heads, T]
        2. Per-head quantile threshold over T values
        3. Union across heads: frame important if ANY head marks it
        4. Group contiguous important frames; filter single-frame segments
        """
        from itertools import groupby
        from operator import itemgetter

        T = features.shape[0]

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_values=audio_tensor,
                    output_attentions=True,
                    return_dict=True,
                )

            # attentions[layer] = attention from transformer layer `layer` (0-indexed).
            # Shape after CLS injection: [n_heads, T+1, T+1].
            attn = outputs.attentions[self.layer][0]  # [n_heads, T+1, T+1]

            # CLS row, audio-frame columns only: [n_heads, T]
            cls_attn = attn[:, 0, 1:]
            if cls_attn.shape[1] > T:
                cls_attn = cls_attn[:, :T]

            # Per-head quantile thresholding (canonical from save_seg_feats.py)
            threshold_value = torch.quantile(
                cls_attn, self.attn_threshold, dim=-1, keepdim=True
            )  # [n_heads, 1]
            important = (cls_attn >= threshold_value).float().sum(0) > 0  # [T]
            important_idx = torch.where(important)[0].cpu().numpy()

            if len(important_idx) == 0:
                seg_pairs = [[0, T]]
            else:
                boundaries_all = []
                boundaries_ex1 = []
                for _, g in groupby(
                    enumerate(important_idx.tolist()), lambda ix: ix[0] - ix[1]
                ):
                    seg = list(map(itemgetter(1), g))
                    t_s, t_e = int(seg[0]), min(int(seg[-1]) + 1, T)
                    if len(seg) > 1:
                        boundaries_ex1.append([t_s, t_e])
                    boundaries_all.append([t_s, t_e])
                seg_pairs = boundaries_ex1 if boundaries_ex1 else boundaries_all

            if in_second:
                segments = [[t_s * spf, t_e * spf] for t_s, t_e in seg_pairs]
            else:
                segments = seg_pairs

            segment_features = torch.stack([
                torch.from_numpy(features[t_s:t_e].mean(0)) for t_s, t_e in seg_pairs
            ])

            cls_attn_sum = cls_attn.sum(0).cpu().numpy()  # [T] — for diagnostics

            return {
                'segments': segments,
                'segment_features': segment_features,
                'cls_attention': cls_attn_sum,
            }

        except Exception as e:
            logger.warning(f"CLS attention segmentation failed ({e}), falling back to featSSM")
            return self._segment_featssm(features, spf, audio_len_sec, in_second)
        
    def save_pretrained(self, save_directory: str):
        """Save model (for compatibility)."""
        raise NotImplementedError("Use the full VGHubertModel for saving to Hub")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load model from HuggingFace Hub (for compatibility)."""
        return cls(model_ckpt=model_name, **kwargs)
