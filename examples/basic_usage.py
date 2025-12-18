"""
Basic usage example for VG-HuBERT syllable and word segmentation.

This example demonstrates the simplest way to use VG-HuBERT.
"""

from vg_hubert import Segmenter
import soundfile as sf


def main():
    # Example audio file path
    audio_path = "audio.wav"  # Replace with your audio file
    
    print("=" * 60)
    print("VG-HuBERT Basic Usage Example")
    print("=" * 60)
    
    # ========================================
    # Syllable Segmentation
    # ========================================
    print("\n1. Syllable Segmentation")
    print("-" * 40)
    
    # Load syllable segmenter
    print("Loading syllable segmenter...")
    syl_segmenter = Segmenter(
        model_ckpt="YOUR_USERNAME/vg-hubert",  # Replace after uploading to Hub
        mode="syllable",
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # Segment audio
    print(f"Segmenting {audio_path}...")
    syl_outputs = syl_segmenter(audio_path)
    
    # Print results
    segments = syl_outputs['segments']
    print(f"\nFound {len(segments)} syllables:")
    for i, (start, end) in enumerate(segments[:5], 1):  # Show first 5
        duration = end - start
        print(f"  Syllable {i}: {start:.3f}s - {end:.3f}s ({duration:.3f}s)")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more")
    
    # Statistics
    durations = [end - start for start, end in segments]
    print(f"\nStatistics:")
    print(f"  Total syllables: {len(segments)}")
    print(f"  Avg duration: {sum(durations)/len(durations):.3f}s")
    print(f"  Min/Max: {min(durations):.3f}s / {max(durations):.3f}s")
    
    # ========================================
    # Word Segmentation
    # ========================================
    print("\n2. Word Segmentation")
    print("-" * 40)
    
    # Load word segmenter
    print("Loading word segmenter...")
    word_segmenter = Segmenter(
        model_ckpt="YOUR_USERNAME/vg-hubert",  # Same model, different mode
        mode="word",
        device="cpu"
    )
    
    # Segment audio
    print(f"Segmenting {audio_path}...")
    word_outputs = word_segmenter(audio_path)
    
    # Print results
    word_segments = word_outputs['segments']
    print(f"\nFound {len(word_segments)} words:")
    for i, (start, end) in enumerate(word_segments, 1):
        duration = end - start
        print(f"  Word {i}: {start:.3f}s - {end:.3f}s ({duration:.3f}s)")
    
    # ========================================
    # Access Features
    # ========================================
    print("\n3. Accessing Features")
    print("-" * 40)
    
    print(f"Segment features shape: {syl_outputs['segment_features'].shape}")
    print(f"  -> {syl_outputs['segment_features'].shape[0]} segments")
    print(f"  -> {syl_outputs['segment_features'].shape[1]} dimensions")
    
    print(f"\nFrame-level features: {syl_outputs['hidden_states'].shape}")
    print(f"  -> {syl_outputs['hidden_states'].shape[0]} frames")
    
    # ========================================
    # Using with numpy arrays
    # ========================================
    print("\n4. Using with Audio Arrays")
    print("-" * 40)
    
    wav, sr = sf.read(audio_path)
    print(f"Loaded audio: {len(wav)} samples @ {sr} Hz")
    
    outputs = syl_segmenter(wav=wav)
    print(f"Segmented into {len(outputs['segments'])} syllables")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
