#!/usr/bin/env python3
"""
CARS Training Manager - Fixed version with proper improved preset handling

This script replaces:
- launch_training.py
- launch_improved_training.py
- launch_training_fixed.py
- optimize_for_m2.py (optimization features)

Usage:
    # Launch training with improved preset
    python cars_training_manager_fixed.py --preset improved --data_path data/processed
    
    # Launch with specific configuration
    python cars_training_manager_fixed.py --data_path data/processed --model_size standard --batch_size 16
    
    # Run hardware optimization only
    python cars_training_manager_fixed.py --optimize_only --device mps
    
    # Launch multiple experiments
    python cars_training_manager_fixed.py --data_path data/processed --experiments baseline improved large
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HardwareOptimizer:
    """Hardware optimization for training configuration"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._detect_device(device)
        self.system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.optimization_results = {}
        self.device_info = self._get_device_info()
        
        print(f"🖥️  Hardware Optimizer")
        print(f"   Device: {self.device}")
        print(f"   Device Info: {self.device_info['description']}")
        if self.device_info['vram_gb'] > 0:
            print(f"   VRAM: {self.device_info['vram_gb']:.1f} GB")
        print(f"   System RAM: {self.system_memory:.1f} GB")
        print("=" * 60)
    
    def _detect_device(self, device: str) -> torch.device:
        """Detect and validate device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        elif device in ['cuda', 'gpu']:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                return torch.device('cpu')
        elif device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("⚠️  MPS not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            'device': str(self.device),
            'vendor': 'Unknown',
            'description': 'Unknown device',
            'vram_gb': 0
        }
        
        if self.device.type == 'cuda':
            info['vendor'] = 'NVIDIA'
            info['description'] = torch.cuda.get_device_name(0)
            info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device.type == 'mps':
            info['vendor'] = 'Apple'
            info['description'] = 'Apple Metal Performance Shaders'
            # MPS doesn't provide direct VRAM query
            info['vram_gb'] = 8.0 if 'Pro' in str(self.system_memory) else 16.0  # Estimate
        else:
            info['vendor'] = 'CPU'
            info['description'] = 'CPU-based computation'
            info['vram_gb'] = 0
        
        return info
    
    def benchmark_configurations(self) -> Dict[str, Any]:
        """Benchmark different model configurations"""
        print("\n🔬 Benchmarking configurations...")
        
        from src.models.fastgan import FastGANGenerator, FastGANDiscriminator
        
        configs = [
            {'name': 'Micro', 'ngf': 32, 'ndf': 32, 'n_layers': 3},
            {'name': 'Small', 'ngf': 48, 'ndf': 48, 'n_layers': 3},
            {'name': 'Standard', 'ngf': 64, 'ndf': 64, 'n_layers': 4},
            {'name': 'Large', 'ngf': 96, 'ndf': 96, 'n_layers': 4},
        ]
        
        batch_sizes = [1, 4, 8, 12, 16, 20, 24, 32]
        results = []
        
        for config in configs:
            print(f"\n📊 Testing {config['name']} configuration...")
            config_results = []
            
            for batch_size in batch_sizes:
                try:
                    # Test model
                    result = self._benchmark_single_config(
                        config['ngf'], config['ndf'], config['n_layers'], 
                        batch_size, image_size=512
                    )
                    
                    if result['success']:
                        config_results.append(result)
                        print(f"   ✅ Batch size {batch_size}: {result['samples_per_second']:.1f} samples/sec")
                    else:
                        print(f"   ❌ Batch size {batch_size}: Failed")
                        break
                        
                except Exception as e:
                    print(f"   ❌ Batch size {batch_size}: {str(e)}")
                    break
            
            results.append({
                'config_name': config['name'],
                'ngf': config['ngf'],
                'ndf': config['ndf'],
                'n_layers': config['n_layers'],
                'image_size': 512,
                'batch_results': config_results
            })
        
        self.optimization_results = {
            'device_info': self.device_info,
            'system_memory_gb': self.system_memory,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        return self._analyze_results()
    
    def _benchmark_single_config(self, ngf: int, ndf: int, n_layers: int, 
                                batch_size: int, image_size: int) -> Dict[str, Any]:
        """Benchmark a single configuration"""
        import torch.nn as nn
        
        try:
            # Create models
            generator = FastGANGenerator(
                latent_dim=256,
                ngf=ngf,
                n_layers=n_layers,
                image_size=image_size,
                channels=1
            ).to(self.device)
            
            discriminator = FastGANDiscriminator(
                ndf=ndf,
                n_layers=n_layers,
                image_size=image_size,
                channels=1
            ).to(self.device)
            
            # Test data
            noise = torch.randn(batch_size, 256).to(self.device)
            real_images = torch.randn(batch_size, 1, image_size, image_size).to(self.device)
            
            # Warmup
            for _ in range(3):
                _ = generator(noise)
                _ = discriminator(real_images)
            
            # Measure memory before
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline_memory = torch.cuda.memory_allocated() / (1024**3)
            else:
                baseline_memory = 0
            
            # Time forward pass
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    fake_images = generator(noise)
                    _ = discriminator(fake_images)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            forward_time = (time.time() - start_time) / 10
            
            # Time backward pass
            fake_images = generator(noise)
            d_fake = discriminator(fake_images)
            loss = d_fake.mean()
            
            start_time = time.time()
            loss.backward()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            backward_time = time.time() - start_time
            
            # Memory usage
            if self.device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                current_memory = torch.cuda.memory_allocated() / (1024**3)
            else:
                peak_memory = current_memory = 0
            
            # Cleanup
            del generator, discriminator, noise, real_images, fake_images
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'batch_size': batch_size,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_time': forward_time + backward_time,
                'baseline_memory': baseline_memory,
                'model_memory': current_memory,
                'peak_memory': peak_memory,
                'memory_increase': peak_memory - baseline_memory,
                'samples_per_second': batch_size / (forward_time + backward_time),
                'success': True
            }
            
        except Exception as e:
            return {
                'batch_size': batch_size,
                'error': str(e),
                'success': False
            }
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations"""
        recommendations = {}
        
        for result in self.optimization_results['results']:
            config_name = result['config_name']
            batch_results = result['batch_results']
            
            if not batch_results:
                continue
            
            # Find optimal batch size (best samples/sec)
            optimal_batch = max(batch_results, key=lambda x: x['samples_per_second'])
            
            # Find maximum stable batch size
            max_batch = batch_results[-1]['batch_size']
            
            recommendations[config_name] = {
                'optimal_batch_size': optimal_batch['batch_size'],
                'max_batch_size': max_batch,
                'optimal_samples_per_second': optimal_batch['samples_per_second'],
                'memory_usage_gb': optimal_batch['peak_memory'],
                'recommended': optimal_batch['samples_per_second'] > 5.0  # Threshold
            }
        
        # Overall recommendation
        if self.device.type == 'mps':
            # M2 Mac specific
            recommendations['optimal_config'] = 'Standard'
            recommendations['optimal_settings'] = {
                'batch_size': 16,
                'num_workers': 0,  # MPS works better with 0 workers
                'precision': '32-true',
                'gradient_checkpointing': False,
                'optimization_flags': [
                    'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0',
                    'export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0'
                ]
            }
        elif self.device.type == 'cuda':
            # NVIDIA GPU
            vram = self.device_info['vram_gb']
            if vram >= 24:
                recommendations['optimal_config'] = 'Large'
                recommendations['optimal_settings'] = {'batch_size': 32, 'precision': '16-mixed'}
            elif vram >= 12:
                recommendations['optimal_config'] = 'Standard'
                recommendations['optimal_settings'] = {'batch_size': 16, 'precision': '16-mixed'}
            else:
                recommendations['optimal_config'] = 'Small'
                recommendations['optimal_settings'] = {'batch_size': 8, 'precision': '32-true'}
        else:
            # CPU
            recommendations['optimal_config'] = 'Micro'
            recommendations['optimal_settings'] = {'batch_size': 4, 'num_workers': 4}
        
        self.optimization_results['recommendations'] = recommendations
        return recommendations
    
    def save_optimization_results(self, output_dir: Path):
        """Save optimization results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        
        # Save recommendations
        with open(output_dir / 'recommendations.json', 'w') as f:
            json.dump(self.optimization_results['recommendations'], f, indent=2)
        
        # Create visualization
        self._create_benchmark_plots(output_dir)
        
        print(f"\n📁 Results saved to: {output_dir}")
    
    def _create_benchmark_plots(self, output_dir: Path):
        """Create benchmark visualization plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'CARS-FASTGAN Optimization Results - {self.device_info["description"]}', fontsize=16)
        
        # Plot 1: Throughput by configuration
        for result in self.optimization_results['results']:
            config_name = result['config_name']
            batch_results = result['batch_results']
            
            if batch_results:
                batch_sizes = [r['batch_size'] for r in batch_results]
                throughputs = [r['samples_per_second'] for r in batch_results]
                ax1.plot(batch_sizes, throughputs, marker='o', label=config_name)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title('Training Throughput by Configuration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage
        for result in self.optimization_results['results']:
            config_name = result['config_name']
            batch_results = result['batch_results']
            
            if batch_results:
                batch_sizes = [r['batch_size'] for r in batch_results]
                memory_usage = [r['peak_memory'] for r in batch_results]
                ax2.plot(batch_sizes, memory_usage, marker='s', label=config_name)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('GPU/Memory Usage by Configuration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Optimal batch sizes
        configs = []
        optimal_batches = []
        for config, rec in self.optimization_results['recommendations'].items():
            if isinstance(rec, dict) and 'optimal_batch_size' in rec:
                configs.append(config)
                optimal_batches.append(rec['optimal_batch_size'])
        
        ax3.bar(configs, optimal_batches, alpha=0.7)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Optimal Batch Size')
        ax3.set_title('Recommended Batch Sizes')
        
        # Plot 4: Summary text
        ax4.axis('off')
        summary_text = f"Device: {self.device_info['description']}\n"
        summary_text += f"Type: {self.device_info['vendor']}\n"
        if self.device_info['vram_gb'] > 0:
            summary_text += f"VRAM: {self.device_info['vram_gb']:.1f} GB\n"
        summary_text += f"System RAM: {self.system_memory:.1f} GB\n\n"
        
        if 'optimal_config' in self.optimization_results['recommendations']:
            summary_text += f"Recommended Config: {self.optimization_results['recommendations']['optimal_config']}\n"
            settings = self.optimization_results['recommendations']['optimal_settings']
            summary_text += f"Batch Size: {settings.get('batch_size', 'N/A')}\n"
            summary_text += f"Precision: {settings.get('precision', '32-true')}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Optimization Summary')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


class TrainingManager:
    """Unified training management system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.experiments_dir = self.project_root / "experiments"
        self.logs_dir = self.experiments_dir / "logs"
        self.checkpoints_dir = self.experiments_dir / "checkpoints"
        
        # Ensure directories exist
        for dir_path in [self.experiments_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load optimization results if available
        self.optimization_cache = self._load_optimization_cache()
        
        print("🚀 CARS Training Manager")
        print(f"📁 Project root: {self.project_root}")
        print("=" * 60)
    
    def _load_optimization_cache(self) -> Optional[Dict]:
        """Load cached optimization results"""
        cache_file = self.project_root / "scripts/optimization_results/recommendations.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def create_experiment_name(self, base_name: Optional[str] = None) -> str:
        """Create unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            return f"{base_name}_{timestamp}"
        return f"cars_fastgan_{timestamp}"
    
    def get_model_config(self, model_size: str) -> Dict[str, Any]:
        """Get model configuration by size"""
        configs = {
            'micro': {
                'generator': {'ngf': 32, 'n_layers': 3},
                'discriminator': {'ndf': 32, 'n_layers': 2}
            },
            'small': {
                'generator': {'ngf': 48, 'n_layers': 3},
                'discriminator': {'ndf': 48, 'n_layers': 3}
            },
            'standard': {
                'generator': {'ngf': 64, 'n_layers': 4},
                'discriminator': {'ndf': 64, 'n_layers': 3}
            },
            'large': {
                'generator': {'ngf': 96, 'n_layers': 4},
                'discriminator': {'ndf': 96, 'n_layers': 4}
            },
            'xlarge': {
                'generator': {'ngf': 128, 'n_layers': 5},
                'discriminator': {'ndf': 128, 'n_layers': 4}
            }
        }
        
        return configs.get(model_size.lower(), configs['standard'])
    
    def get_experiment_presets(self) -> Dict[str, Dict]:
        """Get predefined experiment configurations"""
        return {
            'baseline': {
                'description': 'Baseline FASTGAN configuration',
                'model_size': 'standard',
                'batch_size': 8,
                'max_epochs': 1000,
                'loss': {
                    'gan_loss': 'hinge',
                    'feature_matching_weight': 10.0
                },
                'optimizer': {
                    'generator': {'lr': 0.0002},
                    'discriminator': {'lr': 0.0002}
                }
            },
            'improved': {
                'description': 'Improved configuration with better stability',
                'model_size': 'standard',
                'batch_size': 16,
                'max_epochs': 1500,
                'loss': {
                    'gan_loss': 'hinge',
                    'feature_matching_weight': 20.0
                },
                'optimizer': {
                    'generator': {'lr': 0.0001},
                    'discriminator': {'lr': 0.0004}
                    # Note: betas are already defined in config files
                },
                'training': {
                    'use_gradient_penalty': True,
                    'gradient_penalty_weight': 10.0,
                    'use_ema': True,
                    'ema_decay': 0.999
                }
            },
            'large': {
                'description': 'Large model for high-quality generation',
                'model_size': 'large',
                'batch_size': 8,
                'max_epochs': 2000,
                'loss': {
                    'gan_loss': 'hinge',
                    'feature_matching_weight': 20.0
                },
                'optimizer': {
                    'generator': {'lr': 0.0001},
                    'discriminator': {'lr': 0.0004}
                },
                'training': {
                    'use_gradient_penalty': True,
                    'gradient_penalty_weight': 10.0,
                    'use_ema': True,
                    'ema_decay': 0.999
                },
                'precision': '16-mixed'  # For GPU
            },
            'fast': {
                'description': 'Fast training for quick experiments',
                'model_size': 'micro',
                'batch_size': 32,
                'max_epochs': 500,
                'loss': {
                    'gan_loss': 'lsgan',
                    'feature_matching_weight': 5.0
                },
                'optimizer': {
                    'generator': {'lr': 0.0002},
                    'discriminator': {'lr': 0.0002}
                }
            }
        }
    
    def launch_training(
        self,
        data_path: str,
        experiment_name: Optional[str] = None,
        model_size: str = 'standard',
        batch_size: Optional[int] = None,
        max_epochs: int = 1000,
        preset: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        device: str = 'auto',
        num_workers: int = 4,
        resume_checkpoint: Optional[str] = None,
        additional_config: Optional[Dict] = None,
        dry_run: bool = False
    ) -> bool:
        """Launch training with specified configuration"""
        
        # Validate data path
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"❌ Data path not found: {data_path}")
            return False
        
        # Handle improved preset specially - use the config file directly
        if preset == 'improved':
            # Get preset config for default values
            preset_config = self.get_experiment_presets()['improved']
            batch_size = batch_size or preset_config.get('batch_size', 16)
            max_epochs = preset_config.get('max_epochs', 1500)
            
            # Create experiment name
            if experiment_name is None:
                experiment_name = self.create_experiment_name('improved')
            
            # Build command using the improved config file
            cmd = ["python", "main.py"]
            cmd.extend([
                f"experiment_name={experiment_name}",
                f"data_path={str(data_path.resolve())}",
                "model=fastgan_improved",  # Override the model config (not +model)
                f"data.batch_size={batch_size}",
                f"data.num_workers={num_workers}",
                f"max_epochs={max_epochs}",
                "gradient_clip_val=0",  # Disable gradient clipping for manual optimization
                "enable_progress_bar=false"  # Disable progress bar to avoid bugs
            ])
            
            if use_wandb:
                cmd.append(f"use_wandb={use_wandb}")
                if wandb_project:
                    cmd.append(f"wandb.project={wandb_project}")
            
            if device != 'auto':
                if device == 'gpu' or device == 'cuda':
                    cmd.extend(['accelerator=gpu', 'devices=1'])
                elif device == 'mps':
                    cmd.extend(['accelerator=mps', 'devices=1'])
                elif device == 'cpu':
                    cmd.extend(['accelerator=cpu', 'devices=1'])
            
            if resume_checkpoint:
                cmd.append(f"resume_from_checkpoint={resume_checkpoint}")
            
            # Print configuration summary
            print(f"📋 Using preset: improved - {preset_config['description']}")
            print(f"\n📋 Training Configuration:")
            print(f"   Experiment: {experiment_name}")
            print(f"   Data: {data_path}")
            print(f"   Preset: improved (using fastgan_improved.yaml)")
            print(f"   Batch size: {batch_size}")
            print(f"   Epochs: {max_epochs}")
            print(f"   Device: {device}")
            
            if dry_run:
                print(f"\n🔍 Dry run - Command:")
                print(" ".join(cmd[:3]))
                for c in cmd[3:]:
                    print(f"    {c} \\")
                return True
            
            # Launch training
            print(f"\n🚀 Launching training...")
            print("=" * 60)
            
            try:
                result = subprocess.run(cmd, cwd=self.project_root, check=True)
                print("\n✅ Training completed successfully!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Training failed with exit code {e.returncode}")
                if e.returncode == 1:
                    print("\n💡 If you see Hydra override errors, try:")
                    print("   export HYDRA_FULL_ERROR=1")
                    print("   Then run the command again for detailed error messages")
                return False
            except KeyboardInterrupt:
                print("\n⏸️  Training interrupted by user")
                return False
        
        # For all other presets and configurations, use the regular flow
        # Create experiment name
        if experiment_name is None:
            experiment_name = self.create_experiment_name(preset or model_size)
        
        # Build configuration
        config = {
            'experiment_name': experiment_name,
            'data_path': str(data_path.resolve()),
            'max_epochs': max_epochs
        }
        
        if use_wandb:
            config['use_wandb'] = use_wandb
            if wandb_project:
                config['wandb'] = {'project': wandb_project}
        
        # Apply preset if specified (but not improved, which was handled above)
        if preset and preset in self.get_experiment_presets():
            preset_config = self.get_experiment_presets()[preset]
            model_size = preset_config.get('model_size', model_size)
            batch_size = batch_size or preset_config.get('batch_size', 8)
            max_epochs = preset_config.get('max_epochs', max_epochs)
            config['max_epochs'] = max_epochs
            
            # Apply preset configurations
            if 'loss' in preset_config:
                config['model'] = config.get('model', {})
                config['model']['loss'] = preset_config['loss']
            
            if 'optimizer' in preset_config:
                config['model'] = config.get('model', {})
                config['model']['optimizer'] = preset_config['optimizer']
            
            if 'training' in preset_config:
                config['model'] = config.get('model', {})
                config['model']['training'] = preset_config['training']
            
            if 'precision' in preset_config:
                config['precision'] = preset_config['precision']
            
            print(f"📋 Using preset: {preset} - {preset_config['description']}")
        
        # Get model configuration
        model_config = self.get_model_config(model_size)
        config['model'] = config.get('model', {})
        config['model'].update(model_config)
        
        # Data configuration
        config['data'] = {
            'batch_size': batch_size or 8,
            'num_workers': num_workers
        }
        
        # Device configuration
        if device != 'auto':
            if device == 'gpu' or device == 'cuda':
                config['accelerator'] = 'gpu'
                config['devices'] = 1
            elif device == 'mps':
                config['accelerator'] = 'mps'
                config['devices'] = 1
            elif device == 'cpu':
                config['accelerator'] = 'cpu'
                config['devices'] = 1
        
        # Resume from checkpoint
        if resume_checkpoint:
            config['resume_from_checkpoint'] = resume_checkpoint
        
        # Apply additional configuration
        if additional_config:
            self._merge_configs(config, additional_config)
        
        # Build command
        cmd = self.build_training_command(config)
        
        # Print configuration summary
        print(f"\n📋 Training Configuration:")
        print(f"   Experiment: {experiment_name}")
        print(f"   Data: {data_path}")
        print(f"   Model: {model_size}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Epochs: {max_epochs}")
        print(f"   Device: {device}")
        
        if dry_run:
            print(f"\n🔍 Dry run - Command:")
            print(" ".join(cmd[:3]))
            for c in cmd[3:]:
                print(f"    {c} \\")
            return True
        
        # Launch training
        print(f"\n🚀 Launching training...")
        print("=" * 60)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            print("\n✅ Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted by user")
            return False
    
    def build_training_command(self, config: Dict[str, Any]) -> List[str]:
        """Build training command with proper Hydra overrides"""
        cmd = ["python", "main.py"]
        
        # Add configuration overrides
        for key, value in config.items():
            if key == 'experiment_name':
                cmd.append(f"experiment_name={value}")
            elif key == 'data_path':
                cmd.append(f"data_path={value}")
            elif key == 'model':
                self._add_model_overrides(cmd, value)
            elif key == 'data':
                for sub_key, sub_value in value.items():
                    cmd.append(f"data.{sub_key}={sub_value}")
            elif key == 'training':
                for sub_key, sub_value in value.items():
                    if sub_key == 'max_epochs':
                        cmd.append(f"max_epochs={sub_value}")
                    else:
                        cmd.append(f"training.{sub_key}={sub_value}")
            elif key in ['max_epochs', 'use_wandb', 'accelerator', 'devices', 'precision']:
                cmd.append(f"{key}={value}")
            elif key == 'callbacks':
                for cb_key, cb_value in value.items():
                    for param_key, param_value in cb_value.items():
                        cmd.append(f"callbacks.{cb_key}.{param_key}={param_value}")
        
        return cmd
    
    def _add_model_overrides(self, cmd: List[str], model_config: Dict[str, Any]):
        """Add model configuration overrides handling special cases"""
        for sub_key, sub_value in model_config.items():
            if isinstance(sub_value, dict):
                for sub_sub_key, sub_sub_value in sub_value.items():
                    # Handle special cases like lists and complex values
                    if isinstance(sub_sub_value, list):
                        # Skip lists like betas - they should be in config files
                        # or convert to string format Hydra can understand
                        if sub_sub_key == 'betas':
                            # Skip betas as they're already in the config file
                            continue
                        elif sub_sub_key == 'feature_layers':
                            # Feature layers can be passed as a string
                            layers_str = ','.join(map(str, sub_sub_value))
                            cmd.append(f"model.{sub_key}.{sub_sub_key}=[{layers_str}]")
                    elif isinstance(sub_sub_value, bool):
                        # Convert boolean to lowercase string
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={str(sub_sub_value).lower()}")
                    else:
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={sub_sub_value}")
            else:
                cmd.append(f"model.{sub_key}={sub_value}")
    
    def launch_multiple_experiments(
        self,
        data_path: str,
        experiments: List[str],
        base_config: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """Launch multiple experiments sequentially"""
        results = {}
        
        print(f"\n🚀 Launching {len(experiments)} experiments...")
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"📊 Experiment {i}/{len(experiments)}: {experiment}")
            print(f"{'='*60}")
            
            # Determine if it's a preset or model size
            presets = self.get_experiment_presets()
            
            if experiment in presets:
                success = self.launch_training(
                    data_path=data_path,
                    preset=experiment,
                    **(base_config or {})
                )
            else:
                success = self.launch_training(
                    data_path=data_path,
                    model_size=experiment,
                    **(base_config or {})
                )
            
            results[experiment] = success
            
            if not success:
                print(f"\n⚠️  Experiment {experiment} failed!")
                continue_prompt = input("Continue with remaining experiments? (y/n): ")
                if continue_prompt.lower() != 'y':
                    break
        
        return results
    
    def optimize_and_launch(
        self,
        data_path: str,
        device: str = 'auto',
        auto_launch: bool = True
    ) -> bool:
        """Run optimization and launch with optimal settings"""
        print("\n🔬 Running hardware optimization...")
        
        optimizer = HardwareOptimizer(device)
        recommendations = optimizer.benchmark_configurations()
        
        # Save results
        opt_dir = self.project_root / "scripts/optimization_results"
        optimizer.save_optimization_results(opt_dir)
        
        if not auto_launch:
            print("\n✅ Optimization complete! Check results in:")
            print(f"   {opt_dir}")
            return True
        
        # Get optimal configuration
        optimal_config = recommendations.get('optimal_config', 'standard')
        optimal_settings = recommendations.get('optimal_settings', {})
        
        print(f"\n📋 Optimal configuration: {optimal_config}")
        print(f"   Batch size: {optimal_settings.get('batch_size', 8)}")
        print(f"   Precision: {optimal_settings.get('precision', '32-true')}")
        
        if 'optimization_flags' in optimal_settings:
            print("\n🚩 Setting optimization flags:")
            for flag in optimal_settings['optimization_flags']:
                print(f"   - {flag}")
                # Set environment variables
                parts = flag.replace('export ', '').split('=')
                if len(parts) == 2:
                    os.environ[parts[0]] = parts[1]
        
        # Launch training with optimal settings
        return self.launch_training(
            data_path=data_path,
            model_size=optimal_config.lower(),
            batch_size=optimal_settings.get('batch_size', 8),
            device=device,
            experiment_name=f"optimized_{optimal_config.lower()}_{optimizer.device_info['vendor'].lower()}"
        )
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def show_available_checkpoints(self) -> List[Path]:
        """Show available checkpoints"""
        checkpoints = list(self.checkpoints_dir.glob("**/*.ckpt"))
        
        if checkpoints:
            print("\n📁 Available checkpoints:")
            for i, ckpt in enumerate(sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"   {i+1}. {ckpt.name} ({size_mb:.1f} MB, {mod_time})")
        else:
            print("\n📁 No checkpoints found")
        
        return checkpoints


def main():
    """Main function for training manager"""
    parser = argparse.ArgumentParser(
        description='CARS Training Manager - Unified training launcher with optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    
    # Training configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--model_size', type=str, default='standard',
                       choices=['micro', 'small', 'standard', 'large', 'xlarge'],
                       help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum training epochs')
    parser.add_argument('--preset', type=str, default=None,
                       choices=['baseline', 'improved', 'large', 'fast'],
                       help='Use predefined experiment preset')
    
    # Multiple experiments
    parser.add_argument('--experiments', nargs='+', type=str,
                       help='Run multiple experiments (e.g., baseline improved large)')
    
    # Optimization
    parser.add_argument('--auto_optimize', action='store_true',
                       help='Run optimization before training')
    parser.add_argument('--optimize_only', action='store_true',
                       help='Only run optimization without training')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'gpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='cars-fastgan',
                       help='W&B project name')
    
    # Resume/Debug
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without starting training')
    parser.add_argument('--show_checkpoints', action='store_true',
                       help='Show available checkpoints and exit')
    
    args = parser.parse_args()
    
    # Create training manager
    manager = TrainingManager()
    
    # Show checkpoints if requested
    if args.show_checkpoints:
        manager.show_available_checkpoints()
        return
    
    # Run optimization only
    if args.optimize_only:
        optimizer = HardwareOptimizer(args.device)
        recommendations = optimizer.benchmark_configurations()
        opt_dir = Path("scripts/optimization_results")
        optimizer.save_optimization_results(opt_dir)
        
        print("\n✅ Optimization complete!")
        print(f"📁 Results saved to: {opt_dir}")
        return
    
    # Auto-optimize and launch
    if args.auto_optimize:
        success = manager.optimize_and_launch(
            data_path=args.data_path,
            device=args.device
        )
        return
    
    # Launch multiple experiments
    if args.experiments:
        results = manager.launch_multiple_experiments(
            data_path=args.data_path,
            experiments=args.experiments,
            base_config={
                'use_wandb': args.use_wandb,
                'wandb_project': args.wandb_project,
                'device': args.device,
                'num_workers': args.num_workers
            }
        )
        
        success_count = sum(results.values())
        print(f"\n✅ Completed {success_count}/{len(results)} experiments successfully")
        return
    
    # Single training launch
    success = manager.launch_training(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        preset=args.preset,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        device=args.device,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        dry_run=args.dry_run
    )
    
    if success and not args.dry_run:
        print("\n✨ Training completed successfully!")
    elif not success and not args.dry_run:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()