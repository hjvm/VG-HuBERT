"""
VG-HuBERT Training Script

Train VG-HuBERT models with visual grounding on SpokenCOCO or Places datasets.
Follows the training procedure from Peng et al. (2022, 2023).

Usage:
    python train.py --config configs/spokencoco.yaml
    python train.py --config configs/places.yaml --gpus 0,1,2,3

Based on the original training code with simplified interface.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import torch
from logging import getLogger, basicConfig, INFO

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from vg_hubert.training import Trainer

basicConfig(level=INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train VG-HuBERT with visual grounding")
    
    # Config file
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    
    # Override options
    parser.add_argument("--gpus", type=str, default=None,
                       help="GPU IDs (comma-separated, overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--exp-dir", type=str, default=None,
                       help="Experiment directory (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    return parser.parse_args()


def override_config(config, args):
    """Override config with command-line arguments."""
    if args.gpus is not None:
        config['gpus'] = args.gpus
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.exp_dir is not None:
        config['exp_dir'] = args.exp_dir
    if args.resume is not None:
        config['resume_checkpoint'] = args.resume
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override with command-line args
    config = override_config(config, args)
    
    # Convert config dict to argparse.Namespace for compatibility with Trainer
    config_namespace = argparse.Namespace(**config)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("VG-HuBERT Training Configuration")
    logger.info("=" * 80)
    for key, value in sorted(config.items()):
        logger.info(f"  {key:30s}: {value}")
    logger.info("=" * 80)
    
    # Create experiment directory
    os.makedirs(config['exp_dir'], exist_ok=True)
    
    # Save config to experiment directory
    config_save_path = os.path.join(config['exp_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to: {config_save_path}")
    
    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(config_namespace)
    
    # Train
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {config['exp_dir']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
