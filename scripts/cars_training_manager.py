#!/usr/bin/env python3
"""
CARS Training Manager - Unified training launcher with optimization and configuration management

This script replaces:
- launch_training.py
- launch_improved_training.py
- launch_training_fixed.py
- optimize_for_m2.py (optimization features)

Usage:
    # Launch training with auto-optimization
    python cars_training_manager.py --data_path data/processed --auto_optimize
    
    # Launch with specific configuration
    python cars_training_manager.py --data_path data/processed --model_size standard --batch_size 16
    
    # Run hardware optimization only
    python cars_training_manager.py --optimize_only --device mps
    
    # Launch multiple experiments
    python cars_training_manager.py --data_path data/processed --experiments baseline improved large
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
        else:
            return torch.device(device)
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            'type': self.device.type,
            'description': 'Unknown',
            'vram_gb': 0,
            'compute_capability': None,
            'vendor': 'Unknown',
            'optimization_hints': []
        }
        
        if self.device.type == 'cuda':
            # NVIDIA GPU info
            info['description'] = torch.cuda.get_device_name(0)
            info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['compute_capability'] = torch.cuda.get_device_capability(0)
            info['vendor'] = 'NVIDIA'
            
            # Optimization hints based on GPU generation
            compute_major, compute_minor = info['compute_capability']
            if compute_major >= 8:  # Ampere and newer (RTX 30xx, A100, etc.)
                info['optimization_hints'] = [
                    'Supports bfloat16 - consider using precision="bf16-mixed"',
                    'Efficient at larger batch sizes',
                    'TensorFloat-32 (TF32) enabled by default'
                ]
            elif compute_major >= 7:  # Turing/Volta (RTX 20xx, V100, etc.)
                info['optimization_hints'] = [
                    'Good float16 performance - use precision="16-mixed"',
                    'Tensor Cores available for acceleration'
                ]
            else:
                info['optimization_hints'] = [
                    'Limited mixed precision support',
                    'Consider smaller batch sizes'
                ]
                
        elif self.device.type == 'mps':
            # Apple Silicon GPU
            info['description'] = 'Apple Silicon GPU (Metal Performance Shaders)'
            info['vendor'] = 'Apple'
            # MPS doesn't expose VRAM directly, estimate based on system
            if self.system_memory >= 32:
                info['vram_gb'] = self.system_memory * 0.75  # Unified memory
            else:
                info['vram_gb'] = self.system_memory * 0.5
            info['optimization_hints'] = [
                'Unified memory architecture',
                'Good performance with batch sizes 8-32',
                'Use precision="32-true" for stability'
            ]
            
        elif self.device.type == 'cpu':
            info['description'] = 'CPU'
            info['vendor'] = 'CPU'
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                info['description'] = cpu_info.get('brand_raw', 'CPU')
            except:
                pass
            info['optimization_hints'] = [
                'Limited to smaller batch sizes',
                'Consider using smaller models',
                'Enable CPU optimizations (MKL, OpenMP)'
            ]
        
        # Check for Intel GPU (if available in future PyTorch versions)
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                info['type'] = 'xpu'
                info['description'] = 'Intel GPU (XPU)'
                info['vendor'] = 'Intel'
                info['vram_gb'] = torch.xpu.get_device_properties(0).total_memory / (1024**3)
                info['optimization_hints'] = [
                    'Use Intel Extension for PyTorch (IPEX)',
                    'Good performance with batch sizes 4-16',
                    'Consider precision="16-mixed" for newer Intel GPUs'
                ]
        except ImportError:
            pass
        
        # Check for AMD GPU (ROCm)
        if self.device.type == 'cuda' and 'AMD' in info['description']:
            info['vendor'] = 'AMD'
            info['optimization_hints'] = [
                'ROCm backend detected',
                'Performance may vary from NVIDIA',
                'Stick to power-of-2 batch sizes',
                'Mixed precision support depends on GPU generation'
            ]
        
        return info
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def benchmark_configurations(self, image_size: int = 512) -> Dict:
        """Benchmark different model configurations adapted for device type"""
        print("\n📊 Benchmarking model configurations...")
        
        # Import here to avoid circular dependencies
        sys.path.append(str(Path(__file__).parent.parent))
        from src.models.fastgan import FastGAN
        
        # Adapt configurations based on device
        configs = self._get_device_adapted_configs()
        batch_sizes = self._get_device_adapted_batch_sizes()
        
        results = []
        
        for config in configs:
            print(f"\n🔧 Testing {config['name']} configuration...")
            config_results = []
            
            for batch_size in batch_sizes:
                try:
                    # Clear cache based on device type
                    self._clear_device_cache()
                    
                    baseline_memory = self.get_memory_usage()
                    
                    # Create model
                    model = FastGAN(
                        latent_dim=256,
                        ngf=config['ngf'],
                        ndf=config['ndf'],
                        generator_layers=config['g_layers'],
                        discriminator_layers=config['d_layers'],
                        image_size=image_size,
                        channels=1
                    ).to(self.device)
                    
                    # Apply device-specific optimizations
                    model = self._apply_device_optimizations(model)
                    
                    # Test forward and backward pass
                    start_time = time.time()
                    
                    z = torch.randn(batch_size, 256, device=self.device)
                    fake_images = model.generator(z)
                    real_images = torch.randn(batch_size, 1, image_size, image_size, device=self.device)
                    
                    real_pred = model.discriminator(real_images)
                    fake_pred = model.discriminator(fake_images)
                    
                    if isinstance(real_pred, tuple):
                        real_pred = real_pred[0]
                    if isinstance(fake_pred, tuple):
                        fake_pred = fake_pred[0]
                    
                    loss = real_pred.mean() + fake_pred.mean()
                    loss.backward()
                    
                    # Synchronize for accurate timing on GPU
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    total_time = time.time() - start_time
                    peak_memory = self.get_memory_usage()
                    
                    # Get VRAM usage for GPUs
                    vram_usage = 0
                    if self.device.type == 'cuda':
                        vram_usage = torch.cuda.max_memory_allocated() / (1024**3)
                        torch.cuda.reset_peak_memory_stats()
                    elif self.device.type == 'mps':
                        # MPS doesn't provide direct memory stats yet
                        vram_usage = peak_memory - baseline_memory
                    
                    config_results.append({
                        'batch_size': batch_size,
                        'time': total_time,
                        'memory': peak_memory - baseline_memory,
                        'vram': vram_usage,
                        'samples_per_second': batch_size / total_time,
                        'success': True
                    })
                    
                    print(f"   Batch {batch_size:2d}: ✅ {total_time:.3f}s, "
                          f"RAM: {peak_memory - baseline_memory:.2f}GB, "
                          f"VRAM: {vram_usage:.2f}GB, "
                          f"{batch_size / total_time:.1f} samples/s")
                    
                    del model, fake_images, real_images, z
                    
                except Exception as e:
                    print(f"   Batch {batch_size:2d}: ❌ {str(e)[:50]}...")
                    config_results.append({
                        'batch_size': batch_size,
                        'success': False,
                        'error': str(e)
                    })
                    
                    try:
                        del model, fake_images, real_images, z
                    except:
                        pass
                    
                    self._clear_device_cache()
            
            results.append({
                'config': config,
                'results': config_results
            })
        
        self.optimization_results = results
        recommendations = self._analyze_results(results)
        
        # Add device-specific recommendations
        self._add_device_recommendations(recommendations)
        
        return recommendations
    
    def _get_device_adapted_configs(self) -> List[Dict]:
        """Get model configurations adapted for device type"""
        base_configs = [
            {"name": "Micro", "ngf": 32, "ndf": 32, "g_layers": 3, "d_layers": 2},
            {"name": "Small", "ngf": 48, "ndf": 48, "g_layers": 3, "d_layers": 3},
            {"name": "Standard", "ngf": 64, "ndf": 64, "g_layers": 4, "d_layers": 3},
            {"name": "Large", "ngf": 96, "ndf": 96, "g_layers": 4, "d_layers": 4},
            {"name": "XLarge", "ngf": 128, "ndf": 128, "g_layers": 5, "d_layers": 4}
        ]
        
        # Adapt based on device capabilities
        if self.device_info['vram_gb'] < 4:
            # Low VRAM - limit to smaller models
            return base_configs[:3]
        elif self.device_info['vram_gb'] < 8:
            # Medium VRAM
            return base_configs[:4]
        else:
            # High VRAM - all configs
            return base_configs
    
    def _get_device_adapted_batch_sizes(self) -> List[int]:
        """Get batch sizes adapted for device type"""
        if self.device.type == 'cpu':
            return [1, 2, 4, 8]
        elif self.device_info['vram_gb'] < 4:
            return [1, 2, 4, 8, 16]
        elif self.device_info['vram_gb'] < 8:
            return [1, 2, 4, 8, 16, 24, 32]
        elif self.device_info['vram_gb'] < 16:
            return [1, 2, 4, 8, 16, 24, 32, 48, 64]
        else:
            # High VRAM GPUs
            return [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    def _clear_device_cache(self):
        """Clear device cache based on device type"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        elif self.device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        # Intel XPU
        elif hasattr(torch, 'xpu') and self.device.type == 'xpu':
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
    
    def _apply_device_optimizations(self, model):
        """Apply device-specific optimizations to model"""
        if self.device_info['vendor'] == 'NVIDIA':
            # Enable TF32 for Ampere GPUs
            if self.device_info['compute_capability'][0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
        elif self.device_info['vendor'] == 'Intel':
            # Apply Intel optimizations if IPEX is available
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model)
            except ImportError:
                pass
                
        elif self.device_info['vendor'] == 'AMD':
            # ROCm specific optimizations
            # Currently limited, but can add as PyTorch ROCm support improves
            pass
            
        return model
    
    def _add_device_recommendations(self, recommendations: Dict):
        """Add device-specific recommendations"""
        if 'optimal' in recommendations:
            optimal_config = recommendations[recommendations['optimal']]
            
            # Add precision recommendations
            if self.device_info['vendor'] == 'NVIDIA':
                compute_major = self.device_info['compute_capability'][0]
                if compute_major >= 8:
                    optimal_config['recommended_precision'] = 'bf16-mixed'
                elif compute_major >= 7:
                    optimal_config['recommended_precision'] = '16-mixed'
                else:
                    optimal_config['recommended_precision'] = '32-true'
                    
            elif self.device_info['vendor'] == 'Apple':
                optimal_config['recommended_precision'] = '32-true'
                
            elif self.device_info['vendor'] == 'Intel':
                optimal_config['recommended_precision'] = '16-mixed'
                
            elif self.device_info['vendor'] == 'AMD':
                optimal_config['recommended_precision'] = '16-mixed'
                
            else:  # CPU
                optimal_config['recommended_precision'] = '32-true'
            
            # Add other optimization flags
            optimal_config['optimization_flags'] = self.device_info['optimization_hints']
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results and provide recommendations"""
        recommendations = {}
        
        for config_result in results:
            config = config_result['config']
            config_name = config['name']
            
            # Find optimal batch size
            successful_results = [r for r in config_result['results'] if r.get('success', False)]
            
            if successful_results:
                # Balance between speed and memory
                scores = []
                for r in successful_results:
                    # Score based on samples/second and memory efficiency
                    memory_score = 1.0 / (1.0 + r['memory'])  # Lower memory is better
                    speed_score = r['samples_per_second'] / 100.0  # Normalized speed
                    combined_score = 0.7 * speed_score + 0.3 * memory_score
                    scores.append((r, combined_score))
                
                optimal = max(scores, key=lambda x: x[1])[0]
                
                recommendations[config_name] = {
                    'batch_size': optimal['batch_size'],
                    'memory_usage': optimal['memory'],
                    'samples_per_second': optimal['samples_per_second'],
                    'model_params': config
                }
        
        # Find overall best configuration
        if recommendations:
            best_config = max(
                recommendations.items(),
                key=lambda x: x[1]['samples_per_second'] / (1.0 + x[1]['memory_usage'])
            )
            recommendations['optimal'] = best_config[0]
        
        return recommendations
    
    def save_optimization_results(self, output_dir: Path):
        """Save optimization results and create visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        
        # Create visualization
        self._create_optimization_plots(output_dir)
        
        print(f"\n✅ Optimization results saved to: {output_dir}")
    
    def _create_optimization_plots(self, output_dir: Path):
        """Create optimization visualization plots"""
        if not self.optimization_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Configuration Optimization Results', fontsize=16)
        
        # Plot 1: Batch size vs throughput for each config
        ax1 = axes[0, 0]
        for config_result in self.optimization_results:
            config_name = config_result['config']['name']
            successful = [r for r in config_result['results'] if r.get('success', False)]
            
            if successful:
                batch_sizes = [r['batch_size'] for r in successful]
                throughputs = [r['samples_per_second'] for r in successful]
                ax1.plot(batch_sizes, throughputs, marker='o', label=config_name)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Batch size vs memory for each config
        ax2 = axes[0, 1]
        for config_result in self.optimization_results:
            config_name = config_result['config']['name']
            successful = [r for r in config_result['results'] if r.get('success', False)]
            
            if successful:
                batch_sizes = [r['batch_size'] for r in successful]
                memory_usage = [r['memory'] for r in successful]
                ax2.plot(batch_sizes, memory_usage, marker='s', label=config_name)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model size comparison
        ax3 = axes[1, 0]
        model_names = []
        model_params = []
        max_throughput = []
        
        for config_result in self.optimization_results:
            config = config_result['config']
            config_name = config['name']
            successful = [r for r in config_result['results'] if r.get('success', False)]
            
            if successful:
                model_names.append(config_name)
                # Estimate parameters
                params = (config['ngf']**2 + config['ndf']**2) * (config['g_layers'] + config['d_layers'])
                model_params.append(params / 1000)  # In thousands
                max_throughput.append(max(r['samples_per_second'] for r in successful))
        
        bars = ax3.bar(model_names, max_throughput, alpha=0.7)
        ax3.set_ylabel('Max Samples/Second')
        ax3.set_title('Maximum Throughput by Model Size')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add parameter count on bars
        for bar, params in zip(bars, model_params):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{params:.0f}K', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Efficiency heatmap
        ax4 = axes[1, 1]
        configs = [r['config']['name'] for r in self.optimization_results]
        batch_sizes = sorted(set(bs for r in self.optimization_results 
                               for res in r['results'] 
                               if res.get('success', False) 
                               for bs in [res['batch_size']]))
        
        efficiency_matrix = np.zeros((len(configs), len(batch_sizes)))
        
        for i, config_result in enumerate(self.optimization_results):
            for result in config_result['results']:
                if result.get('success', False) and result['batch_size'] in batch_sizes:
                    j = batch_sizes.index(result['batch_size'])
                    # Efficiency score
                    efficiency = result['samples_per_second'] / (1.0 + result['memory'])
                    efficiency_matrix[i, j] = efficiency
        
        im = ax4.imshow(efficiency_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(batch_sizes)))
        ax4.set_xticklabels(batch_sizes)
        ax4.set_yticks(range(len(configs)))
        ax4.set_yticklabels(configs)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Model Configuration')
        ax4.set_title('Training Efficiency Heatmap')
        plt.colorbar(im, ax=ax4, label='Efficiency Score')
        
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
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            cmd.append(f"model.{sub_key}.{sub_sub_key}={sub_sub_value}")
                    else:
                        cmd.append(f"model.{sub_key}={sub_value}")
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
        
        # Create experiment name
        if experiment_name is None:
            experiment_name = self.create_experiment_name(preset or model_size)
        
        # Build configuration
        config = {
            'experiment_name': experiment_name,
            'data_path': str(data_path.resolve()),
            'max_epochs': max_epochs,
            'use_wandb': use_wandb
        }
        
        # Apply preset if specified
        if preset and preset in self.get_experiment_presets():
            preset_config = self.get_experiment_presets()[preset]
            model_size = preset_config.get('model_size', model_size)
            batch_size = batch_size or preset_config.get('batch_size', 8)
            max_epochs = preset_config.get('max_epochs', max_epochs)
            
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
            if device == 'gpu':
                config['accelerator'] = 'gpu'
                config['devices'] = 1
            elif device == 'mps':
                config['accelerator'] = 'mps'
                config['devices'] = 1
            elif device == 'cpu':
                config['accelerator'] = 'cpu'
                config['devices'] = 1
        
        # W&B configuration
        if use_wandb and wandb_project:
            config['wandb'] = {'project': wandb_project}
        
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
    
    def launch_multiple_experiments(
        self,
        data_path: str,
        experiments: List[str],
        base_config: Optional[Dict] = None,
        sequential: bool = True
    ) -> Dict[str, bool]:
        """Launch multiple experiments"""
        results = {}
        
        print(f"\n🔬 Launching {len(experiments)} experiments...")
        
        for exp_name in experiments:
            print(f"\n{'='*60}")
            print(f"Experiment: {exp_name}")
            print(f"{'='*60}")
            
            # Get preset or use as model size
            presets = self.get_experiment_presets()
            if exp_name in presets:
                success = self.launch_training(
                    data_path=data_path,
                    preset=exp_name,
                    additional_config=base_config
                )
            else:
                # Assume it's a model size
                success = self.launch_training(
                    data_path=data_path,
                    model_size=exp_name,
                    experiment_name=f"{exp_name}_experiment",
                    additional_config=base_config
                )
            
            results[exp_name] = success
            
            if not sequential:
                print("⚠️  Non-sequential mode not implemented yet")
                break
            
            if not success and sequential:
                print(f"\n❌ Experiment {exp_name} failed, stopping sequence")
                break
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 Experiment Summary:")
        for exp, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {exp}: {status}")
        
        return results
    
    def optimize_and_launch(
        self,
        data_path: str,
        device: str = 'auto',
        target_metric: str = 'efficiency'
    ) -> bool:
        """Run optimization and launch with optimal settings"""
        print("\n🔧 Running optimization before training...")
        
        # Run optimization
        optimizer = HardwareOptimizer(device)
        recommendations = optimizer.benchmark_configurations()
        
        # Save optimization results
        opt_dir = self.project_root / "scripts/optimization_results"
        optimizer.save_optimization_results(opt_dir)
        
        # Get optimal configuration
        if 'optimal' in recommendations:
            optimal_config = recommendations['optimal']
            optimal_settings = recommendations[optimal_config]
            
            print(f"\n✅ Optimal configuration: {optimal_config}")
            print(f"   Batch size: {optimal_settings['batch_size']}")
            print(f"   Memory usage: {optimal_settings['memory_usage']:.2f} GB")
            if 'vram' in optimal_settings:
                print(f"   VRAM usage: {optimal_settings.get('vram', 0):.2f} GB")
            print(f"   Throughput: {optimal_settings['samples_per_second']:.1f} samples/s")
            
            # Print device-specific recommendations
            if 'recommended_precision' in optimal_settings:
                print(f"   Recommended precision: {optimal_settings['recommended_precision']}")
            if 'optimization_flags' in optimal_settings:
                print(f"\n   Device-specific optimizations:")
                for flag in optimal_settings['optimization_flags']:
                    print(f"   - {flag}")
            
            # Build additional config with device optimizations
            additional_config = {}
            if 'recommended_precision' in optimal_settings:
                additional_config['precision'] = optimal_settings['recommended_precision']
            
            # Launch training with optimal settings
            return self.launch_training(
                data_path=data_path,
                model_size=optimal_config.lower(),
                batch_size=optimal_settings['batch_size'],
                experiment_name=f"optimized_{optimal_config.lower()}_{optimizer.device_info['vendor'].lower()}",
                device=device,
                additional_config=additional_config
            )
        else:
            print("❌ No optimal configuration found")
            return False
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def show_available_checkpoints(self) -> List[Path]:
        """Show available checkpoints"""
        checkpoints = list(self.checkpoints_dir.glob("*.ckpt"))
        
        if checkpoints:
            print("\n📁 Available checkpoints:")
            for i, ckpt in enumerate(sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"   {i+1}. {ckpt.name} ({size_mb:.1f} MB, {mod_time})")
        else:
            print("\n📁 No checkpoints found")
        
        return checkpoints
    
    def create_config_report(self, config: Dict) -> str:
        """Create a formatted configuration report"""
        report = []
        report.append("# Training Configuration Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        def format_dict(d: Dict, indent: int = 0) -> List[str]:
            lines = []
            for key, value in d.items():
                prefix = "  " * indent + "- "
                if isinstance(value, dict):
                    lines.append(f"{prefix}**{key}**:")
                    lines.extend(format_dict(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return lines
        
        report.extend(format_dict(config))
        
        return "\n".join(report)


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
        print("📊 Check logs: experiments/logs/")
        print("💾 Check checkpoints: experiments/checkpoints/")


if __name__ == "__main__":
    main()