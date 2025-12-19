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

import torch
import numpy as np
import soundfile as sf
from typing import Dict, Union, Optional, List, Tuple
from pathlib import Path
import logging
import os
import pickle

logger = logging.getLogger(__name__)


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
        attn_threshold: float = 0.25,
        checkpoint_file: Optional[str] = None,  # Override which .pth file to use
        **kwargs
    ):
        """
        Initialize VG-HuBERT Segmenter.
        
        Following syllable-discovery repo recommendations:
        - Syllable mode: best_bundle.pth, layer 8, sec_per_syllable=0.2, merge_threshold=0.3
        - Word mode: snapshot_20.pth, layer 9, attention-based
        """
        self.mode = mode
        
        # Set defaults based on mode (from syllable-discovery repo)
        if mode == "syllable":
            self.layer = layer if layer is not None else 8
            self.sec_per_syllable = sec_per_syllable if sec_per_syllable is not None else 0.2
            self.merge_threshold = merge_threshold if merge_threshold is not None else 0.3
            self.checkpoint_file = checkpoint_file if checkpoint_file is not None else "vg-hubert-syllable.pth"
        elif mode == "word":
            self.layer = layer if layer is not None else 9
            self.sec_per_syllable = sec_per_syllable if sec_per_syllable is not None else 0.2
            self.merge_threshold = merge_threshold if merge_threshold is not None else 0.3
            self.checkpoint_file = checkpoint_file if checkpoint_file is not None else "vg-hubert-word.pth"
        else:
            raise ValueError(f"mode must be 'syllable' or 'word', got {mode}")
        
        self.attn_threshold = attn_threshold
        
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
            potential_paths = [
                "../findsylls/models/vg-hubert_3",
                "models/vg-hubert_3",
                "../models/vg-hubert_3"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    model_ckpt = path
                    logger.info(f"Auto-detected model at: {model_ckpt}")
                    break
            
            if model_ckpt is None:
                model_ckpt = "vg-hubert"  # Will try to download from Hub
        
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
            # Use eager attention only for word mode (needs attention weights)
            # SDPA is faster but can't output attention weights
            attn_impl = 'eager' if self.mode == 'word' else 'sdpa'
            self.model = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                attn_implementation=attn_impl
            )
            logger.info(f"Using {attn_impl} attention implementation for {self.mode} mode")
            
            # Load VG-HuBERT weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "dual_encoder" in checkpoint:
                state_dict = checkpoint["dual_encoder"]
            else:
                state_dict = checkpoint
            
            # Simple weight loading (transformers HuBERT is compatible enough)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded VG-HuBERT weights from {checkpoint_path.name}")
        
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
                
                attn_impl = 'eager' if self.mode == 'word' else 'sdpa'
                logger.info(f"Using {attn_impl} attention implementation for {self.mode} mode")
                self.model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960",
                    attn_implementation=attn_impl
                )
                logger.info(f"Using {attn_impl} attention for {self.mode} mode")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "dual_encoder" in checkpoint:
                    state_dict = checkpoint["dual_encoder"]
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Model loaded from HuggingFace Hub")
                
            except Exception as e:
                logger.warning(
                    f"Could not download VG-HuBERT from Hub: {e}\n"
                    f"Falling back to base HuBERT model (facebook/hubert-base-ls960).\n"
                    f"For full VG-HuBERT functionality, download the model from:\n"
                    f"https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar"
                )
                # Use base HuBERT as fallback for demonstration
                attn_impl = 'eager' if self.mode == 'word' else 'sdpa'
                self.model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960",
                    attn_implementation=attn_impl
                )
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
            
            # Get features from specified layer
            features = outputs.hidden_states[self.layer][0].cpu().float().numpy()
            
            # Remove CLS token if present
            if features.shape[0] > len(audio) / 320:  # HuBERT downsamples by 320
                features = features[1:]
        
        # Calculate seconds per frame
        spf = audio_len_sec / features.shape[0]
        
        # Segment based on mode
        if self.mode == "syllable":
            result = self._segment_syllables(features, spf, audio_len_sec, in_second)
        elif self.mode == "word":
            result = self._segment_words(audio_tensor, features, spf, audio_len_sec, in_second)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Add hidden states
        result['hidden_states'] = features
        
        return result
    
    def _segment_syllables(
        self,
        features: np.ndarray,
        spf: float,
        audio_len_sec: float,
        in_second: bool
    ) -> Dict:
        """Segment using MinCut algorithm (syllables)."""
        # Estimate number of syllables
        num_syllables = max(1, int(np.ceil(audio_len_sec / self.sec_per_syllable)))
        
        # Compute self-similarity matrix
        ssm = features @ features.T
        ssm = ssm - np.min(ssm) + 1e-7  # Make non-negative
        
        # Apply MinCut
        try:
            from .mincut import min_cut
            seg_boundary_frames = min_cut(ssm, num_syllables + 1)
        except Exception as e:
            # Fallback: uniform segmentation
            logger.warning(f"MinCut failed ({e}), using uniform segmentation")
            seg_boundary_frames = np.linspace(0, features.shape[0], num_syllables + 1).astype(int)
        
        # Create segments
        seg_pairs = [[l, r] for l, r in zip(seg_boundary_frames[:-1], seg_boundary_frames[1:])]
        seg_pairs = [item for item in seg_pairs if item[1] - item[0] > 2]
        
        # Merge similar segments
        if self.merge_threshold is not None and len(seg_pairs) >= 3:
            seg_pairs = self._merge_similar_segments(features, seg_pairs)
        
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
    
    def _merge_similar_segments(
        self,
        features: np.ndarray,
        seg_pairs: List[List[int]]
    ) -> List[List[int]]:
        """Merge adjacent segments with high similarity."""
        # Compute mean features for each segment
        all_feat = [features[round(l):round(r)].mean(0) for l, r in seg_pairs]
        
        # Compute cosine similarities
        all_sim = [
            np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            for f1, f2 in zip(all_feat[:-1], all_feat[1:])
        ]
        
        # Iteratively merge most similar pairs
        while len(seg_pairs) >= 3:
            max_sim_idx = np.argmax(all_sim)
            
            if all_sim[max_sim_idx] < self.merge_threshold:
                break
            
            # Merge segments
            l_merge = seg_pairs[max_sim_idx]
            r_merge = seg_pairs[max_sim_idx + 1]
            
            seg_pairs = [
                pair for i, pair in enumerate(seg_pairs)
                if i != max_sim_idx and i != max_sim_idx + 1
            ]
            seg_pairs.insert(max_sim_idx, [l_merge[0], r_merge[1]])
            
            # Recompute features and similarities
            all_feat = [features[round(l):round(r)].mean(0) for l, r in seg_pairs]
            all_sim = [
                np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
                for f1, f2 in zip(all_feat[:-1], all_feat[1:])
            ]
        
        return seg_pairs
    
    def _segment_words(
        self,
        audio_tensor: torch.Tensor,
        features: np.ndarray,
        spf: float,
        audio_len_sec: float,
        in_second: bool
    ) -> Dict:
        """
        Segment using CLS token attention weights (words).
        
        Following VG-HuBERT word-discovery paper: use attention from CLS token
        to identify word boundaries. Peaks in CLS attention indicate word onsets.
        """
        try:
            # Extract attention weights from the target layer
            with torch.no_grad():
                outputs = self.model(
                    input_values=audio_tensor,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Get attention from target layer
                # Shape: (batch_size, num_heads, seq_len, seq_len)
                attn = outputs.attentions[self.layer]
                
                # Extract CLS token attention (first token attends to all positions)
                # Shape: (batch_size, num_heads, seq_len)
                cls_attn = attn[0, :, 0, :]
                
                # Remove CLS position itself
                if cls_attn.shape[1] > features.shape[0]:
                    cls_attn = cls_attn[:, 1:]
                
                # Average over heads or use max/specific heads
                # Paper uses different strategies - here we use max across heads
                cls_attn_score = cls_attn.max(dim=0)[0].cpu().numpy()
                
                # Ensure alignment with features
                if len(cls_attn_score) > features.shape[0]:
                    cls_attn_score = cls_attn_score[:features.shape[0]]
            
            # Find peaks in CLS attention (word onsets)
            from scipy.signal import find_peaks
            
            # Normalize attention scores
            cls_attn_score = (cls_attn_score - cls_attn_score.min()) / (cls_attn_score.max() - cls_attn_score.min() + 1e-8)
            
            # Find peaks above threshold
            peaks, _ = find_peaks(cls_attn_score, height=self.attn_threshold, distance=int(0.1 / spf))
            
            # Create segments between peaks
            if len(peaks) == 0:
                # No peaks found, return whole utterance
                seg_pairs = [[0, features.shape[0]]]
            else:
                # Add boundaries at start and end
                boundaries = [0] + peaks.tolist() + [features.shape[0]]
                seg_pairs = [[l, r] for l, r in zip(boundaries[:-1], boundaries[1:])]
                seg_pairs = [item for item in seg_pairs if item[1] - item[0] > 2]
            
            # Convert to time
            if in_second:
                segments = [[l * spf, r * spf] for l, r in seg_pairs]
            else:
                segments = seg_pairs
            
            # Extract segment features
            segment_features = torch.stack([
                torch.from_numpy(features[l:r].mean(0)) for l, r in seg_pairs
            ])
            
            logger.info(f"CLS attention-based word segmentation: {len(segments)} words")
            
            return {
                'segments': segments,
                'segment_features': segment_features,
                'cls_attention': cls_attn_score  # Include attention scores for analysis
            }
            
        except Exception as e:
            logger.warning(f"CLS attention word segmentation failed ({e}), falling back to syllable mode")
            return self._segment_syllables(features, spf, audio_len_sec, in_second)
        
    def save_pretrained(self, save_directory: str):
        """Save model (for compatibility)."""
        raise NotImplementedError("Use the full VGHubertModel for saving to Hub")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load model from HuggingFace Hub (for compatibility)."""
        return cls(model_ckpt=model_name, **kwargs)
