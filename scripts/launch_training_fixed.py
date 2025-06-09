#!/usr/bin/env python3
"""
Fixed training launcher using proper Hydra configuration paths
Based on reviewing the actual codebase structure
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
    
    # Based on your main.py and config structure:
    # - max_epochs is under "training" in configs/training/default.yaml
    # - log_images_every_n_epochs is under "training" in configs/training/default.yaml
    # - Model parameters are under "model" in configs/model/fastgan.yaml
    # - Data parameters are under "data" in configs/data/cars_dataset.yaml
    
    cmd = [
        "python", "main.py",
        f"experiment_name={experiment_name}",
        f"data_path={data_path}",
        
        # Data overrides (from configs/data/cars_dataset.yaml)
        "data.batch_size=16",
        "data.num_workers=0",  # Mac compatibility
        
        # Model overrides (from configs/model/fastgan.yaml)
        "model.generator.ngf=64",  # Upgrade from micro to standard
        "model.generator.n_layers=4",
        "model.discriminator.ndf=64",
        "model.discriminator.n_layers=4",
        "model.discriminator.num_scales=3",
        
        # Loss overrides (from configs/model/fastgan.yaml)
        "model.loss.feature_matching_weight=20.0",
        
        # Optimizer overrides (from configs/model/fastgan.yaml)
        "model.optimizer.generator.lr=0.0001",
        "model.optimizer.discriminator.lr=0.0004",
        
        # Training overrides (from configs/training/default.yaml)
        "training.max_epochs=1500",
        "training.log_images_every_n_epochs=10",
        
        # GAN training settings (from configs/training/default.yaml)
        "training.gan_training.warmup_epochs=20",
        "training.gan_training.discriminator_warmup=10",
        
        # Model training settings (from configs/model/fastgan.yaml)
        "model.training.use_gradient_penalty=true",
        "model.training.gradient_penalty_weight=10.0",
        "model.training.use_ema=true",
        "model.training.ema_decay=0.999",
    ]
    
    print("🚀 Launching CARS-FASTGAN Training (Fixed)")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Data path: {data_path}")
    print("\nKey improvements:")
    print("  - Model: Standard (64 filters, 4 layers) vs Micro (32 filters, 3 layers)")
    print("  - Feature matching weight: 20.0 (vs 10.0)")
    print("  - Learning rates: G=0.0001, D=0.0004")
    print("  - Gradient penalty: Enabled")
    print("  - EMA: Enabled")
    print("  - Epochs: 1500")
    print("\nCommand:")
    for i, c in enumerate(cmd):
        if i < 3:
            print(f"  {c}")
        else:
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
        # Print helpful debug command
        print("\nTo debug configuration issues, run:")
        print("HYDRA_FULL_ERROR=1 python main.py --help")
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
    
    args = parser.parse_args()
    
    # Verify data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("\nMake sure you've run:")
        print("1. python scripts/prepare_cars_data_properly.py")
        print("2. python scripts/enhance_cars_contrast.py")
        sys.exit(1)
    
    launch_improved_training(str(data_path), args.experiment_name)