"""
CARS-FASTGAN Training Launcher
Intelligent training script that uses optimization results and prepared data
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class TrainingLauncher:
    """Intelligent training launcher for CARS-FASTGAN"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.optimization_results = None
        self.data_config = None
        
        print("🚀 CARS-FASTGAN Training Launcher")
        print("=" * 50)
    
    def load_optimization_results(self) -> Optional[Dict]:
        """Load M2 optimization results if available"""
        opt_file = self.project_root / "scripts/optimization_results/recommendations.json"
        
        if opt_file.exists():
            with open(opt_file, 'r') as f:
                self.optimization_results = json.load(f)
            print(f"✅ Loaded optimization results from {opt_file}")
            return self.optimization_results
        else:
            print("⚠️  No optimization results found. Using default settings.")
            print("   Run 'python scripts/optimize_for_m2.py' for optimal settings.")
            return None
    
    def detect_prepared_data(self, data_path: Optional[str] = None) -> Optional[Path]:
        """Detect prepared CARS data"""
        search_paths = []
        
        if data_path:
            search_paths.append(Path(data_path))
        
        # Common data locations
        search_paths.extend([
            self.project_root / "data/processed",
            self.project_root / "data/cars_processed",
            Path("data/processed"),
            Path("data/cars_processed")
        ])
        
        for path in search_paths:
            if path.exists():
                # Check if it looks like prepared data
                required_dirs = ['train', 'val', 'test']
                if all((path / dir_name).exists() for dir_name in required_dirs):
                    config_file = None
                    
                    # Look for config file
                    for config_name in ['cars_dataset_prepared.yaml', 'cars_dataset.yaml']:
                        config_path = path / config_name
                        if config_path.exists():
                            config_file = config_path
                            break
                    
                    print(f"✅ Found prepared data at: {path}")
                    if config_file:
                        print(f"✅ Found config file: {config_file}")
                        self.data_config = config_file
                    else:
                        print("⚠️  No config file found, will use defaults")
                    
                    return path
        
        print("❌ No prepared data found!")
        print("   Run 'python scripts/prepare_cars_data.py /path/to/your/images' first")
        return None
    
    def create_experiment_config(
        self, 
        data_path: Path,
        experiment_name: Optional[str] = None,
        model_config: str = "auto",
        batch_size: Optional[int] = None,
        max_epochs: int = 1000,
        use_wandb: bool = False
    ) -> Dict[str, Any]:
        """Create experiment configuration"""
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"cars_fastgan_{timestamp}"
        
        print(f"📝 Creating experiment: {experiment_name}")
        
        config = {
            'experiment_name': experiment_name,
            'data_path': str(data_path.resolve()),
            'max_epochs': max_epochs,
            'use_wandb': use_wandb
        }
        
        # Apply optimization results if available
        if self.optimization_results:
            print("🎯 Applying optimization results...")
            
            # Model configuration
            if model_config == "auto" and 'optimal_model' in self.optimization_results:
                optimal_model = self.optimization_results['optimal_model']
                config['model'] = {
                    'generator': {
                        'ngf': optimal_model['ngf'],
                        'n_layers': optimal_model['g_layers']
                    },
                    'discriminator': {
                        'ndf': optimal_model['ndf'],
                        'n_layers': optimal_model['d_layers']
                    }
                }
                print(f"   - Model: {optimal_model['config_name']}")
                print(f"   - Generator: {optimal_model['ngf']} filters, {optimal_model['g_layers']} layers")
                print(f"   - Discriminator: {optimal_model['ndf']} filters, {optimal_model['d_layers']} layers")
            
            # Batch size
            if batch_size is None:
                model_name = self.optimization_results.get('optimal_model', {}).get('config_name', 'Standard')
                batch_key = f"{model_name}_optimal_batch"
                
                if batch_key in self.optimization_results:
                    optimal_batch = self.optimization_results[batch_key]['batch_size']
                    config['data'] = {'batch_size': optimal_batch}
                    print(f"   - Batch size: {optimal_batch}")
                else:
                    print("   - Using default batch size: 8")
            else:
                config['data'] = {'batch_size': batch_size}
                print(f"   - Batch size: {batch_size}")
        else:
            # Default configuration for small datasets
            if batch_size is None:
                batch_size = 8
            
            config['data'] = {'batch_size': batch_size}
            
            if model_config == "auto":
                # Choose model based on data size (estimate)
                train_dir = data_path / "train"
                if train_dir.exists():
                    train_images = len(list(train_dir.glob("*")))
                    
                    if train_images < 100:
                        model_config = "micro"
                    elif train_images < 300:
                        model_config = "standard"
                    else:
                        model_config = "standard"  # Conservative choice
                
                print(f"   - Auto-selected model: {model_config}")
            
            # Apply model configuration
            if model_config != "auto":
                model_configs = {
                    "micro": {"ngf": 32, "ndf": 32, "g_layers": 3, "d_layers": 2},
                    "standard": {"ngf": 64, "ndf": 64, "g_layers": 4, "d_layers": 3},
                    "large": {"ngf": 128, "ndf": 128, "g_layers": 5, "d_layers": 4}
                }
                
                if model_config in model_configs:
                    mc = model_configs[model_config]
                    config['model'] = {
                        'generator': {'ngf': mc['ngf'], 'n_layers': mc['g_layers']},
                        'discriminator': {'ndf': mc['ndf'], 'n_layers': mc['d_layers']}
                    }
        
        return config
    
    def estimate_training_time(self, config: Dict[str, Any], data_path: Path) -> str:
        """Estimate training time based on configuration"""
        
        # Count training images
        train_dir = data_path / "train"
        train_images = len(list(train_dir.glob("*"))) if train_dir.exists() else 400
        
        batch_size = config.get('data', {}).get('batch_size', 8)
        max_epochs = config.get('max_epochs', 1000)
        
        # Rough estimates based on M2 Mac performance
        # These are very rough estimates - actual time depends on many factors
        batches_per_epoch = max(1, train_images // batch_size)
        
        # Estimate seconds per batch (varies by model size)
        model_ngf = config.get('model', {}).get('generator', {}).get('ngf', 64)
        if model_ngf <= 32:
            seconds_per_batch = 0.5  # Micro model
        elif model_ngf <= 64:
            seconds_per_batch = 1.0  # Standard model
        else:
            seconds_per_batch = 2.0  # Large model
        
        total_seconds = batches_per_epoch * max_epochs * seconds_per_batch
        
        hours = total_seconds / 3600
        
        if hours < 1:
            return f"~{int(total_seconds/60)} minutes"
        elif hours < 24:
            return f"~{hours:.1f} hours"
        else:
            return f"~{hours/24:.1f} days"
    
    def create_training_command(self, config: Dict[str, Any]) -> List[str]:
        """Create the training command"""
        
        cmd = ["python", "main.py"]
        
        # Add configuration overrides
        for key, value in config.items():
            if key == 'model' and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            cmd.append(f"model.{sub_key}.{sub_sub_key}={sub_sub_value}")
                    else:
                        cmd.append(f"model.{sub_key}={sub_value}")
            elif key == 'data' and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    cmd.append(f"data.{sub_key}={sub_value}")
            elif key in ['experiment_name', 'data_path', 'max_epochs', 'use_wandb']:
                cmd.append(f"{key}={value}")
        
        return cmd
    
    def launch_training(
        self,
        data_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_config: str = "auto",
        batch_size: Optional[int] = None,
        max_epochs: int = 1000,
        use_wandb: bool = False,
        dry_run: bool = False
    ) -> bool:
        """Launch training with optimal settings"""
        
        # Load optimization results
        self.load_optimization_results()
        
        # Detect prepared data
        data_path_obj = self.detect_prepared_data(data_path)
        if data_path_obj is None:
            return False
        
        # Create experiment configuration
        config = self.create_experiment_config(
            data_path_obj, experiment_name, model_config, batch_size, max_epochs, use_wandb
        )
        
        # Estimate training time
        time_estimate = self.estimate_training_time(config, data_path_obj)
        print(f"⏱️  Estimated training time: {time_estimate}")
        
        # Create training command
        cmd = self.create_training_command(config)
        
        print(f"\n📋 Training Configuration:")
        print(f"   - Data path: {config['data_path']}")
        print(f"   - Experiment: {config['experiment_name']}")
        print(f"   - Max epochs: {config['max_epochs']}")
        print(f"   - Weights & Biases: {'Enabled' if config['use_wandb'] else 'Disabled'}")
        
        print(f"\n🖥️  Training Command:")
        print(f"   {' '.join(cmd)}")
        
        if dry_run:
            print(f"\n🔍 Dry run mode - not starting training")
            return True
        
        # Confirm before starting
        print(f"\n❓ Start training? (y/n): ", end="")
        response = input().strip().lower()
        
        if response not in ['y', 'yes']:
            print("❌ Training cancelled")
            return False
        
        # Change to project directory
        os.chdir(self.project_root)
        
        print(f"\n🚀 Starting training...")
        print("=" * 50)
        
        try:
            # Launch training
            result = subprocess.run(cmd, check=True)
            
            print("=" * 50)
            print("🎉 Training completed successfully!")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print("=" * 50)
            print(f"❌ Training failed with exit code {e.returncode}")
            return False
        
        except KeyboardInterrupt:
            print("\n" + "=" * 50)
            print("⏸️  Training interrupted by user")
            return False
    
    def show_optimization_summary(self):
        """Show optimization results summary"""
        if self.optimization_results is None:
            print("No optimization results available")
            return
        
        print("\n🎯 M2 Mac Optimization Summary:")
        print("-" * 40)
        
        if 'optimal_model' in self.optimization_results:
            model = self.optimization_results['optimal_model']
            print(f"📐 Recommended Model: {model['config_name']}")
            print(f"   - Parameters: {model['total_params']:,}")
            print(f"   - Memory usage: {model['memory_usage']:.2f} GB")
            print(f"   - Performance: {model['samples_per_second']:.1f} samples/s")
        
        print(f"\n📦 Recommended Batch Sizes:")
        for config_name in ['Micro', 'Standard', 'Large']:
            batch_key = f'{config_name}_optimal_batch'
            if batch_key in self.optimization_results:
                batch = self.optimization_results[batch_key]
                print(f"   - {config_name}: {batch['batch_size']} ")
                print(f"     ({batch['memory_usage']:.2f} GB, {batch['samples_per_second']:.1f} samples/s)")


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Launch CARS-FASTGAN training with optimal settings')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to prepared data (auto-detected if not specified)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--model_config', type=str, default='auto',
                       choices=['auto', 'micro', 'standard', 'large'],
                       help='Model configuration')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (uses optimization results if not specified)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum training epochs')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without starting training')
    parser.add_argument('--show_optimization', action='store_true',
                       help='Show optimization results summary')
    
    args = parser.parse_args()
    
    launcher = TrainingLauncher()
    
    if args.show_optimization:
        launcher.load_optimization_results()
        launcher.show_optimization_summary()
        return
    
    success = launcher.launch_training(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        model_config=args.model_config,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        use_wandb=args.use_wandb,
        dry_run=args.dry_run
    )
    
    if success:
        print("\n✨ Training launched successfully!")
        print("Monitor progress with:")
        print("  - TensorBoard: tensorboard --logdir experiments/logs")
        if args.use_wandb:
            print("  - Weights & Biases: Check your W&B dashboard")
    else:
        print("\n❌ Failed to launch training")


if __name__ == "__main__":
    main()