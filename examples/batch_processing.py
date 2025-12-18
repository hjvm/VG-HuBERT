"""
Batch processing example for VG-HuBERT.

Process multiple audio files and save results to JSON.
"""

from vg_hubert import Segmenter
from pathlib import Path
import json
import argparse


def process_directory(
    input_dir: str,
    output_dir: str,
    model_ckpt: str = "YOUR_USERNAME/vg-hubert",
    mode: str = "syllable",
    device: str = "cpu"
):
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save results
        model_ckpt: Model checkpoint (Hub ID or local path)
        mode: 'syllable' or 'word'
        device: 'cpu' or 'cuda'
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load segmenter once
    print(f"Loading VG-HuBERT ({mode} mode)...")
    segmenter = Segmenter(
        model_ckpt=model_ckpt,
        mode=mode,
        device=device
    )
    
    # Find all audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f'**/*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    results = {}
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing {audio_file.name}...")
        
        try:
            # Segment audio
            outputs = segmenter(str(audio_file))
            
            # Save results
            file_id = audio_file.stem
            results[file_id] = {
                'file': str(audio_file),
                'num_segments': len(outputs['segments']),
                'segments': [
                    {'start': float(start), 'end': float(end)}
                    for start, end in outputs['segments']
                ]
            }
            
            # Save individual file results
            output_file = output_path / f"{file_id}_{mode}.json"
            with open(output_file, 'w') as f:
                json.dump(results[file_id], f, indent=2)
                
        except Exception as e:
            print(f"  Error processing {audio_file.name}: {e}")
            results[file_id] = {'error': str(e)}
    
    # Save combined results
    combined_file = output_path / f"all_results_{mode}.json"
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_path}")
    print(f"Combined results: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process audio files with VG-HuBERT"
    )
    parser.add_argument(
        'input_dir',
        help="Directory containing audio files"
    )
    parser.add_argument(
        'output_dir',
        help="Directory to save results"
    )
    parser.add_argument(
        '--model',
        default="YOUR_USERNAME/vg-hubert",
        help="Model checkpoint (default: YOUR_USERNAME/vg-hubert)"
    )
    parser.add_argument(
        '--mode',
        choices=['syllable', 'word'],
        default='syllable',
        help="Segmentation mode (default: syllable)"
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help="Device to use (default: cpu)"
    )
    
    args = parser.parse_args()
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_ckpt=args.model,
        mode=args.mode,
        device=args.device
    )


if __name__ == "__main__":
    main()
