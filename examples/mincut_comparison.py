"""
Example: Comparing MinCut algorithms with and without MinCutMerge post-processing.

This script demonstrates the different configurations available in VG-HuBERT:
1. Original MinCut (baseline)
2. SyllableLM optimized MinCut (faster)
3. MinCut + MinCutMerge-0.3 (matches original paper)
4. SyllableLM + MinCutMerge-0.3 (RECOMMENDED: fast + accurate)

Based on the original VG-HuBERT paper:
    Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization
    in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
"""

from vg_hubert import Segmenter
import numpy as np


def main():
    # Example audio file (replace with your own)
    audio_file = "path/to/your/audio.wav"
    
    print("=" * 80)
    print("VG-HuBERT MinCut Configuration Comparison")
    print("=" * 80)
    print()
    
    # Configuration 1: Default (matches original paper)
    # - New algorithm (fast)
    # - MinCutMerge-0.3 (prevents over-segmentation)
    # - Recommended for most use cases
    print("1️⃣  RECOMMENDED: New MinCut + MinCutMerge-0.3")
    print("   (Fast algorithm + original paper's post-processing)")
    print()
    
    segmenter_recommended = Segmenter(
        mode="syllable",
        layer=8,
        sec_per_syllable=0.2,
        merge_threshold=0.3,  # Enable MinCutMerge (original paper value)
        min_segment_frames=2  # Filter very short segments
    )
    
    outputs = segmenter_recommended(audio_file)
    print(f"   Found {len(outputs['segments'])} syllables")
    print(f"   First 3: {outputs['segments'][:3]}")
    print()
    
    # Configuration 2: Plain MinCut (no merging)
    # - May over-segment (more boundaries than actual syllables)
    # - Useful for analysis or when you want more granular segments
    print("2️⃣  Plain MinCut (no merging)")
    print("   (May over-segment, more boundaries)")
    print()
    
    segmenter_plain = Segmenter(
        mode="syllable",
        layer=8,
        sec_per_syllable=0.2,
        merge_threshold=None,  # Disable MinCutMerge
        min_segment_frames=2
    )
    
    outputs_plain = segmenter_plain(audio_file)
    print(f"   Found {len(outputs_plain['segments'])} syllables")
    print(f"   Difference: {len(outputs_plain['segments']) - len(outputs['segments'])} more boundaries")
    print()
    
    # Configuration 3: Custom merge threshold
    # - Tune merge_threshold to control granularity
    # - Higher threshold = more merging = fewer segments
    # - Lower threshold = less merging = more segments
    print("3️⃣  Custom merge threshold (threshold=0.5)")
    print("   (More aggressive merging)")
    print()
    
    segmenter_custom = Segmenter(
        mode="syllable",
        layer=8,
        sec_per_syllable=0.2,
        merge_threshold=0.5,  # More aggressive merging
        min_segment_frames=2
    )
    
    outputs_custom = segmenter_custom(audio_file)
    print(f"   Found {len(outputs_custom['segments'])} syllables")
    print(f"   (Fewer segments due to aggressive merging)")
    print()
    
    # Configuration 4: Low-level API (for advanced users)
    # - Direct access to MinCut functions
    # - Full control over parameters
    print("4️⃣  Low-level API (advanced)")
    print("   (Direct access to algorithms)")
    print()
    
    from vg_hubert.mincut import segment_with_mincut
    import soundfile as sf
    from transformers import HubertModel
    import torch
    
    # Load audio
    audio, sr = sf.read(audio_file)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Extract features (simplified - see full example for production code)
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        outputs = model(audio_tensor, output_hidden_states=True)
        features = outputs.hidden_states[8][0].cpu().numpy()
    
    # Apply MinCut with full control
    K = int(np.ceil(len(audio) / 16000 / 0.2)) + 1  # Estimate K
    boundaries, ssm = segment_with_mincut(
        features=features,
        K=K,
        merge_threshold=0.3,  # Set to None for plain MinCut
        min_segment_frames=2,
        min_hop=3,  # Min segment length
        max_hop=50  # Max segment length
    )
    
    print(f"   Found {len(boundaries)-1} segments")
    print(f"   Boundaries: {boundaries[:5]}... (frame indices)")
    print()
    
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print("• For most users: Use Configuration 1 (RECOMMENDED)")
    print("  - Matches original paper's performance")
    print("  - Fast (~40x speedup) and accurate")
    print("  - Validated against LibriSpeech ground truth")
    print()
    print("• For analysis: Use Configuration 2 (plain MinCut)")
    print("  - More fine-grained segmentation")
    print("  - Good for exploring syllable boundaries")
    print()
    print("• For tuning: Adjust merge_threshold (0.0-1.0)")
    print("  - 0.3: Original paper value (balanced)")
    print("  - 0.5: More aggressive merging (fewer segments)")
    print("  - None: No merging (most segments)")
    print()
    print("• For research: Use low-level API (Configuration 4)")
    print("  - Full control over all parameters")
    print("  - Access to SSM and intermediate results")
    print()


if __name__ == "__main__":
    main()
