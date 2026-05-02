#!/usr/bin/env python3
"""
Quick demo: Load VG-HuBERT from HuggingFace Hub and segment audio.

This demonstrates the complete workflow:
1. Load model from Hub (automatic download)
2. Segment audio file
3. Display results
"""

from vg_hubert import Segmenter
import os

# Optional: Set token for private repo access
# os.environ['HF_TOKEN'] = 'your_token_here'

print("=" * 60)
print("VG-HuBERT HuggingFace Hub Demo")
print("=" * 60)

# Create segmenter (downloads from Hub automatically)
print("\n1. Loading syllable segmenter from HuggingFace Hub...")
segmenter = Segmenter(
    model_ckpt="hjvm/VG-HuBERT",
    mode="syllable",
    merge_threshold=0.3,  # Enable MinCutMerge
    device="cpu"
)
print("   ✓ Loaded successfully!")

# Test with sample audio
test_audio = "../test_samples/narr_mono.wav"
if os.path.exists(test_audio):
    print(f"\n2. Segmenting test audio: {test_audio}")
    outputs = segmenter(test_audio)
    
    print(f"\n3. Results:")
    print(f"   - Found {len(outputs['segments'])} syllables")
    print(f"   - Segment features shape: {outputs['segment_features'].shape}")
    print(f"   - Frame features shape: {outputs['hidden_states'].shape}")
    
    print(f"\n4. First 5 syllables:")
    for i, (start, end) in enumerate(outputs['segments'][:5]):
        duration = end - start
        print(f"   Syllable {i+1}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
else:
    print(f"\n⚠️  Test audio not found: {test_audio}")
    print("   Segmenter loaded successfully, but no audio to test with.")

print("\n" + "=" * 60)
print("✅ Demo complete!")
print("=" * 60)
print("\nYou can now use VG-HuBERT from anywhere:")
print("  from vg_hubert import Segmenter")
print('  seg = Segmenter(model_ckpt="hjvm/VG-HuBERT")')
print('  outputs = seg("your_audio.wav")')
