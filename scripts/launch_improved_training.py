#!/usr/bin/env python3
"""
Launch improved CARS-FASTGAN training with fixes for all identified issues
"""

import os
import sys
from pathlib import Path
import subprocess
import json
import yaml
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class ImprovedTrainingLauncher:
    """Launch training with improved configuration based on diagnostic findings"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🚀 CARS-FASTGAN Improved Training Launcher")
        print("=" * 60)
        
    def create_improved_model_config(self):
        """Create improved model configuration addressing structural complexity issues"""
        config = {
            'model_name': 'fastgan_improved',
            
            # Upgrade to standard model for better structural learning
            'generator': {
                'latent_dim': 256,
                'ngf': 64,  # Increased from 32 (micro) to 64 (standard)
                'n_layers': 4,  # Increased from 3 to 4
                'image_size': 512,
                'channels': 1,
                'use_skip_connections': True,
                'skip_connection_scale': 0.1,
                'norm_type': 'batch',
                'activation': 'relu',
                'output_activation': 'tanh',
                'init_type': 'normal',
                'init_gain': 0.02,
                
                # NEW: Add self-attention for better spatial relationships
                'use_self_attention': True,
                'attention_layers': [2, 3],  # Add attention at layers 2 and 3
            },
            
            'discriminator': {
                'ndf': 64,  # Increased from 32 to 64
                'n_layers': 4,  # Increased from 3 to 4 for larger receptive field
                'image_size': 512,
                'channels': 1,
                'use_spectral_norm': True,
                'norm_type': 'batch',
                'activation': 'leaky_relu',
                'leaky_relu_slope': 0.2,
                'use_self_attention': True,
                'attention_layers': [2],
                'use_patch_gan': False,
                'init_type': 'normal',
                'init_gain': 0.02,
                
                # Multi-scale discrimination
                'use_multiscale': True,
                'num_scales': 3,  # Increased from 2 to 3 for better detail capture
            },
            
            'loss': {
                'gan_loss': 'hinge',
                'adversarial_weight': 1.0,
                'feature_matching_weight': 20.0,  # Increased from 10.0
                'use_feature_matching': True,
                'feature_layers': [1, 2, 3, 4],  # Use more layers
                
                # NEW: Add perceptual loss for structural preservation
                'use_perceptual_loss': True,
                'perceptual_weight': 5.0,
                'perceptual_layers': ['conv_3_3', 'conv_4_3'],
            },
            
            'optimizer': {
                'generator': {
                    'type': 'adam',
                    'lr': 0.0001,  # Reduced from 0.0002 for stability
                    'betas': [0.0, 0.999],
                    'weight_decay': 0.0,
                },
                'discriminator': {
                    'type': 'adam',
                    'lr': 0.0004,  # 4x generator LR for better balance
                    'betas': [0.0, 0.999],
                    'weight_decay': 0.0,
                }
            },
            
            'scheduler': {
                'use_scheduler': True,
                'type': 'linear',
                'start_epoch': 500,
                'end_epoch': 1000,
            },
            
            'training': {
                'n_critic': 1,
                'use_gradient_penalty': True,  # Enable for stability
                'gradient_penalty_weight': 10.0,
                'progressive_growing': False,  # Could enable later
                'use_ema': True,  # Enable EMA for stability
                'ema_decay': 0.999,
            }
        }
        
        # Save config
        config_path = Path("configs/model/fastgan_improved.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✅ Created improved model config: {config_path}")
        return config_path
    
    def create_training_config(self):
        """Create optimized training configuration"""
        config = {
            'max_epochs': 1500,  # Increased for better convergence
            'min_epochs': 500,
            'early_stopping_patience': 300,
            
            'val_check_interval': 1.0,
            'check_val_every_n_epoch': 10,
            'save_last': True,
            'save_top_k': 5,  # Keep more checkpoints
            
            'log_every_n_steps': 10,
            'log_images_every_n_epochs': 10,  # More frequent image logging
            'num_sample_images': 16,
            
            # Hardware optimization
            'accelerator': 'auto',
            'devices': 1,
            'precision': '32-true',
            'enable_progress_bar': True,
            
            'gradient_clip_val': 1.0,
            'gradient_clip_algorithm': 'norm',
            'accumulate_grad_batches': 1,
            
            'enable_checkpointing': True,
            'enable_model_summary': True,
            'detect_anomaly': False,
            
            'gan_training': {
                'warmup_epochs': 20,  # Increased warmup
                'warmup_lr_factor': 0.1,
                'discriminator_warmup': 10,  # Train D first
                'generator_skip_threshold': 0.8,
                'loss_smoothing_window': 100,
                'loss_balance_threshold': 0.1,
                'fixed_noise_size': 64,
                'adaptive_lr': True,  # Enable adaptive learning
                'lr_adaptation_window': 50,
                'lr_adaptation_factor': 0.5,
            },
            
            'callbacks': {
                'model_checkpoint': {
                    'monitor': 'val/fid_score',
                    'mode': 'min',
                    'save_top_k': 5,
                    'filename': 'fastgan-improved-{epoch:02d}-{val_fid_score:.2f}',
                    'auto_insert_metric_name': False,
                },
                'early_stopping': {
                    'monitor': 'val/fid_score',
                    'mode': 'min',
                    'patience': 300,
                    'min_delta': 0.01,
                },
                'lr_monitor': {
                    'logging_interval': 'epoch',
                    'log_momentum': False,
                },
                'image_generation': {
                    'enabled': True,
                    'log_interval': 10,
                    'num_samples': 16,
                },
            }
        }
        
        config_path = Path("configs/training/improved.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✅ Created improved training config: {config_path}")
        return config_path
    
    def create_launch_script(self, fixed_data_path: str, use_wandb: bool = False):
        """Create the launch command with all improvements"""
        
        experiment_name = f"cars_fastgan_improved_{self.timestamp}"
        
        # Base command
        cmd = [
            "python", "main.py",
            f"experiment_name={experiment_name}",
            f"data_path={fixed_data_path}",
            
            # Model improvements
            "model=fastgan_improved",  # Use our improved config
            "model.generator.ngf=64",
            "model.generator.n_layers=4",
            "model.discriminator.ndf=64", 
            "model.discriminator.n_layers=4",
            "model.discriminator.num_scales=3",
            
            # Loss improvements
            "model.loss.feature_matching_weight=20.0",
            
            # Training improvements
            "training=improved",  # Use our improved training config
            "training.max_epochs=1500",
            "data.batch_size=16",  # Optimal for standard model on M2
            
            # Optimization improvements
            "model.optimizer.generator.lr=0.0001",
            "model.optimizer.discriminator.lr=0.0004",
            "model.training.use_gradient_penalty=true",
            "model.training.use_ema=true",
            
            # Logging
            "training.log_images_every_n_epochs=10",
            
            # Evaluation
            "evaluation.metrics.fid.enabled=true",
            "evaluation.metrics.fid.num_samples=500",
        ]
        
        if use_wandb:
            cmd.extend([
                "use_wandb=true",
                "wandb.project=cars-fastgan-improved",
                f"wandb.tags=[improved,{self.timestamp}]"
            ])
            
        return cmd
    
    def create_runpod_script(self, fixed_data_path: str):
        """Create a script for RunPod deployment with larger model"""
        script_content = f'''#!/bin/bash
# RunPod training script for CARS-FASTGAN with large model

# Setup environment
cd /workspace
git clone https://github.com/YOUR_USERNAME/cars_fastgan.git
cd cars_fastgan

# Install dependencies
pip install -r requirements.txt
pip install wandb

# Copy data (adjust path as needed)
# You'll need to upload your fixed data to RunPod first
cp -r /workspace/data_fixed/* data/

# Login to wandb (optional)
# wandb login YOUR_API_KEY

# Launch training with large model
python main.py \\
    experiment_name=cars_fastgan_large_{self.timestamp} \\
    data_path=data/processed_fixed \\
    model.generator.ngf=128 \\
    model.generator.n_layers=5 \\
    model.discriminator.ndf=128 \\
    model.discriminator.n_layers=4 \\
    model.discriminator.num_scales=3 \\
    model.loss.feature_matching_weight=20.0 \\
    model.optimizer.generator.lr=0.0001 \\
    model.optimizer.discriminator.lr=0.0004 \\
    model.training.use_gradient_penalty=true \\
    model.training.use_ema=true \\
    training.max_epochs=2000 \\
    data.batch_size=32 \\
    accelerator=gpu \\
    devices=1 \\
    precision=16-mixed \\
    use_wandb=true \\
    wandb.project=cars-fastgan-large
'''
        
        script_path = Path("scripts/runpod_training.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        print(f"✅ Created RunPod script: {script_path}")
        return script_path
    
    def estimate_training_time(self, model_size: str = "standard", epochs: int = 1500):
        """Estimate training time"""
        # Based on your micro model: 1000 epochs with 15 iterations/epoch
        if model_size == "micro":
            time_per_epoch = 1.5  # minutes (rough estimate)
        elif model_size == "standard":
            time_per_epoch = 3.0  # minutes
        else:  # large
            time_per_epoch = 6.0  # minutes
            
        total_hours = (epochs * time_per_epoch) / 60
        
        if total_hours < 24:
            return f"~{total_hours:.1f} hours"
        else:
            return f"~{total_hours/24:.1f} days"
    
    def run(self, fixed_data_path: str, launch_local: bool = True, use_wandb: bool = False):
        """Run the improved training setup"""
        
        # Create improved configs
        model_config = self.create_improved_model_config()
        training_config = self.create_training_config()
        
        # Create launch command
        cmd = self.create_launch_script(fixed_data_path, use_wandb)
        
        # Create RunPod script
        runpod_script = self.create_runpod_script(fixed_data_path)
        
        # Estimate time
        time_estimate = self.estimate_training_time("standard", 1500)
        
        print("\n" + "=" * 60)
        print("📋 Improved Training Configuration Summary:")
        print(f"  - Model: Standard (64 filters, 4 layers)")
        print(f"  - Batch size: 16")
        print(f"  - Epochs: 1500")
        print(f"  - Learning rate: G=0.0001, D=0.0004")
        print(f"  - Feature matching weight: 20.0")
        print(f"  - Gradient penalty: Enabled")
        print(f"  - EMA: Enabled")
        print(f"  - Estimated time: {time_estimate}")
        
        print(f"\n🖥️  Local Training Command:")
        print("  " + " ".join(cmd[:5]) + " \\")
        for c in cmd[5:]:
            print(f"    {c} \\")
        print()
        
        print(f"\n☁️  RunPod Script: {runpod_script}")
        print("  For RunPod: Upload fixed data first, then run the script")
        
        if launch_local:
            print(f"\n❓ Start local training on Mac? (y/n): ", end="")
            response = input().strip().lower()
            
            if response in ['y', 'yes']:
                os.chdir(self.project_root)
                print(f"\n🚀 Starting improved training...")
                print("=" * 60)
                
                try:
                    subprocess.run(cmd, check=True)
                    print("\n✅ Training completed successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ Training failed: {e}")
                except KeyboardInterrupt:
                    print("\n⏸️  Training interrupted")
        else:
            print("\n✅ Configuration ready for manual launch")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch improved CARS-FASTGAN training')
    parser.add_argument('--fixed_data_path', type=str, default='data/processed_fixed',
                       help='Path to fixed data directory')
    parser.add_argument('--no_launch', action='store_true',
                       help='Create configs but do not launch training')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    launcher = ImprovedTrainingLauncher()
    launcher.run(
        fixed_data_path=args.fixed_data_path,
        launch_local=not args.no_launch,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main()