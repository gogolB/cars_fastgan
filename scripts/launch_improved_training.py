#!/usr/bin/env python3
"""
Fixed training launcher using proper Hydra configuration paths
Compatible with the updated codebase and configuration structure
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import sys


def launch_improved_training(data_path: str, experiment_name: str = None):
    """Launch training with improved settings using correct config paths"""
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cars_fastgan_improved_{timestamp}"
    
    # Absolute path for data
    data_path = Path(data_path).resolve()
    
    # Build command with proper Hydra overrides
    cmd = [
        "python", "main.py",
        f"experiment_name={experiment_name}",
        f"data_path={data_path}",
        
        # Data configuration overrides
        "data.batch_size=16",
        "data.num_workers=0",  # Mac compatibility
        
        # Model configuration overrides
        "model.generator.ngf=64",
        "model.generator.n_layers=4",
        "model.discriminator.ndf=64",
        "model.discriminator.n_layers=4",
        "model.discriminator.num_scales=3",
        
        # Loss configuration
        "model.loss.feature_matching_weight=20.0",
        
        # Optimizer configuration
        "model.optimizer.generator.lr=0.0001",
        "model.optimizer.discriminator.lr=0.0004",
        
        # Training configuration
        "model.training.use_gradient_penalty=true",
        "model.training.gradient_penalty_weight=10.0",
        "model.training.use_ema=true",
        "model.training.ema_decay=0.999",
        
        # Main config overrides (training parameters)
        "max_epochs=1500",
        "log_images_every_n_epochs=10",
        
        # GAN training settings
        "gan_training.warmup_epochs=20",
        "gan_training.discriminator_warmup=10",
    ]
    
    print("🚀 Launching CARS-FASTGAN Training (Fixed v2)")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Data path: {data_path}")
    print("\nKey improvements:")
    print("  - Model: Standard (64 filters, 4 layers)")
    print("  - Feature matching weight: 20.0 (vs 10.0)")
    print("  - Learning rates: G=0.0001, D=0.0004")
    print("  - Gradient penalty: Enabled")
    print("  - EMA: Enabled")
    print("  - Epochs: 1500")
    print("\nCommand:")
    print(" ".join(cmd[:3]))
    for c in cmd[3:]:
        print(f"    {c} \\")
    print("\n" + "=" * 60)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
        sys.exit(0)


def launch_large_model_training(data_path: str, experiment_name: str = None):
    """Launch training with large model configuration"""
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cars_fastgan_large_{timestamp}"
    
    # Absolute path for data
    data_path = Path(data_path).resolve()
    
    cmd = [
        "python", "main.py",
        f"experiment_name={experiment_name}",
        f"data_path={data_path}",
        
        # Data configuration
        "data.batch_size=8",  # Smaller batch for large model
        "data.num_workers=0",
        
        # Large model configuration
        "model.generator.ngf=128",
        "model.generator.n_layers=5",
        "model.discriminator.ndf=128",
        "model.discriminator.n_layers=4",
        "model.discriminator.num_scales=3",
        
        # Loss configuration
        "model.loss.feature_matching_weight=20.0",
        
        # Optimizer configuration
        "model.optimizer.generator.lr=0.0001",
        "model.optimizer.discriminator.lr=0.0004",
        
        # Training configuration
        "model.training.use_gradient_penalty=true",
        "model.training.gradient_penalty_weight=10.0",
        "model.training.use_ema=true",
        "model.training.ema_decay=0.999",
        
        # Training parameters
        "max_epochs=2000",
        "log_images_every_n_epochs=10",
        
        # GAN training settings
        "gan_training.warmup_epochs=30",
        "gan_training.discriminator_warmup=15",
        
        # For GPU/RunPod - uncomment these lines
        # "accelerator=gpu",
        # "devices=1",
        # "precision=16-mixed",
        # "data.num_workers=4",
    ]
    
    print("🚀 Launching CARS-FASTGAN Training (Large Model)")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Data path: {data_path}")
    print("\nLarge model configuration:")
    print("  - Model: Large (128 filters, 5 layers)")
    print("  - Batch size: 8 (memory consideration)")
    print("  - Epochs: 2000")
    print("\n" + "=" * 60)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch improved CARS-FASTGAN training')
    parser.add_argument('--data_path', type=str, default='data/processed_enhanced',
                       help='Path to enhanced data directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--large_model', action='store_true',
                       help='Use large model configuration')
    
    args = parser.parse_args()
    
    # Verify data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("\nMake sure you've run the data preparation scripts:")
        print("1. python scripts/prepare_cars_data_properly.py")
        print("2. python scripts/enhance_cars_contrast.py")
        sys.exit(1)
    
    if args.large_model:
        launch_large_model_training(str(data_path), args.experiment_name)
    else:
        launch_improved_training(str(data_path), args.experiment_name)