#!/usr/bin/env python3
"""
CARS-FASTGAN Training Manager
Comprehensive training orchestration with hardware optimization
Enhanced with detailed error reporting for debugging
"""

import os
import sys
import json
import argparse
import subprocess
import psutil
import GPUtil
import torch
import torch.backends.mps
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import time
import yaml
from dataclasses import dataclass, field
import tempfile
import shutil
import traceback  # Added for detailed error reporting

# Add src to path - CRITICAL FOR IMPORTS
sys.path.append(str(Path(__file__).parent.parent))

# Rich console imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Initialize console
console = Console()


@dataclass
class HardwareInfo:
    """Hardware information container"""
    device: torch.device
    device_name: str
    vram_gb: float
    system_ram_gb: float
    num_gpus: int
    gpu_names: List[str] = field(default_factory=list)


class HardwareOptimizer:
    """Optimize training configurations for available hardware"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.system_memory = psutil.virtual_memory().total / (1024**3)
        self.device_info = self._get_device_info()
        self.optimization_results = {}
        
        # Display hardware info
        console.print(Panel.fit(
            f"[bold cyan]🖥️  Hardware Optimizer[/bold cyan]\n"
            f"   Device: {self.device}\n"
            f"   Device Info: {self.device_info['description']}\n"
            f"   VRAM: {self.device_info['vram_gb']:.1f} GB\n"
            f"   System RAM: {self.system_memory:.1f} GB",
            title="Hardware Configuration"
        ))
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate torch device"""
        if device != 'auto':
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'is_built') and torch.backends.mps.is_built():
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
        """Benchmark different model configurations with enhanced error reporting"""
        print("\n🔬 Benchmarking configurations...")
        
        # Import with error handling
        try:
            from src.models.fastgan import FastGANGenerator, FastGANDiscriminator
            print("✅ Successfully imported FastGAN models")
        except ImportError as e:
            print(f"❌ Failed to import FastGAN models: {e}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Python path: {sys.path[:3]}...")
            traceback.print_exc()
            return {}
        
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
                    # Test model with detailed error catching
                    result = self._benchmark_single_config(
                        config['ngf'], config['ndf'], config['n_layers'], 
                        batch_size, image_size=512
                    )
                    
                    if result['success']:
                        config_results.append(result)
                        print(f"   ✅ Batch size {batch_size}: {result['samples_per_second']:.1f} samples/sec")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        print(f"   ❌ Batch size {batch_size}: Failed - {error_msg}")
                        if 'traceback' in result:
                            print(f"      Traceback: {result['traceback']}")
                        break
                        
                except Exception as e:
                    print(f"   ❌ Batch size {batch_size}: Exception - {str(e)}")
                    print(f"      Exception type: {type(e).__name__}")
                    traceback.print_exc()
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
        """Benchmark a single configuration with detailed error reporting"""
        import torch.nn as nn
        
        try:
            # Import models with error handling
            try:
                from src.models.fastgan import FastGANGenerator, FastGANDiscriminator
            except ImportError as e:
                return {
                    'batch_size': batch_size,
                    'error': f"Import error: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'success': False
                }
            
            # Create models with detailed error catching
            try:
                generator = FastGANGenerator(
                    latent_dim=256,
                    ngf=ngf,
                    n_layers=n_layers,
                    image_size=image_size,
                    channels=1
                ).to(self.device)
            except Exception as e:
                return {
                    'batch_size': batch_size,
                    'error': f"Generator creation failed: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'success': False
                }
            
            try:
                discriminator = FastGANDiscriminator(
                    ndf=ndf,
                    n_layers=n_layers,
                    image_size=image_size,
                    channels=1
                ).to(self.device)
            except Exception as e:
                return {
                    'batch_size': batch_size,
                    'error': f"Discriminator creation failed: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'success': False
                }
            
            # Test data
            noise = torch.randn(batch_size, 256).to(self.device)
            real_images = torch.randn(batch_size, 1, image_size, image_size).to(self.device)
            
            # Warmup
            for _ in range(3):
                _ = generator(noise)
                d_out = discriminator(real_images)
                # Handle tuple output
                if isinstance(d_out, (tuple, list)):
                    _ = d_out[0]
            
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
                    d_out = discriminator(fake_images)
                    # Handle tuple output from multi-scale discriminator
                    if isinstance(d_out, (tuple, list)):
                        _ = d_out[0]  # Just access to ensure computation
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            forward_time = (time.time() - start_time) / 10
            
            # Time backward pass
            fake_images = generator(noise)
            d_fake = discriminator(fake_images)
            
            # Handle discriminator output - could be tuple or tensor
            if isinstance(d_fake, (tuple, list)):
                # Multi-scale discriminator returns tuple of outputs
                # Use the first output (usually the main prediction)
                d_fake = d_fake[0]
            
            # Ensure we have a tensor and compute loss
            if d_fake.dim() > 1:
                # If output has multiple dimensions, flatten and take mean
                loss = d_fake.view(-1).mean()
            else:
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
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle CUDA OOM specifically
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'batch_size': batch_size,
                'error': 'CUDA out of memory',
                'error_type': 'OutOfMemoryError',
                'success': False
            }
        except RuntimeError as e:
            # Many CUDA errors manifest as RuntimeError
            error_msg = str(e)
            if 'out of memory' in error_msg.lower() or 'oom' in error_msg.lower():
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                return {
                    'batch_size': batch_size,
                    'error': 'Out of memory',
                    'error_type': 'OutOfMemoryError',
                    'success': False
                }
            else:
                return {
                    'batch_size': batch_size,
                    'error': error_msg,
                    'error_type': 'RuntimeError',
                    'traceback': traceback.format_exc(),
                    'success': False
                }
        except Exception as e:
            error_details = {
                'batch_size': batch_size,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'success': False
            }
            
            # Add additional debugging info
            if hasattr(e, 'args') and e.args:
                error_details['error_args'] = str(e.args)
            
            return error_details
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations"""
        recommendations = {}
        
        if not self.optimization_results.get('results'):
            print("⚠️  No benchmark results to analyze")
            return recommendations
        
        for result in self.optimization_results['results']:
            config_name = result['config_name']
            batch_results = result['batch_results']
            
            if not batch_results:
                print(f"⚠️  No successful benchmarks for {config_name} configuration")
                continue
            
            # Find optimal batch size (best samples/sec)
            optimal_batch = max(batch_results, key=lambda x: x['samples_per_second'])
            
            recommendations[config_name] = {
                'optimal_batch_size': optimal_batch['batch_size'],
                'samples_per_second': optimal_batch['samples_per_second'],
                'memory_usage': optimal_batch['peak_memory'],
                'ngf': result['ngf'],
                'ndf': result['ndf'],
                'n_layers': result['n_layers']
            }
        
        # Find overall best configuration
        if recommendations:
            best_config = max(recommendations.items(), 
                            key=lambda x: x[1]['samples_per_second'])
            recommendations['best_config'] = best_config[0]
            recommendations['best_batch_size'] = best_config[1]['optimal_batch_size']
        
        return recommendations
    
    def save_optimization_results(self, output_dir: Path):
        """Save optimization results and plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        
        # Save recommendations
        recommendations = self._analyze_results()
        with open(output_dir / 'recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Create visualization with error handling
        try:
            self._create_plots(output_dir)
        except Exception as e:
            print(f"⚠️  Failed to create plots: {e}")
            traceback.print_exc()
        
        print(f"\n📁 Results saved to: {output_dir}")
    
    def _create_plots(self, output_dir: Path):
        """Create optimization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        # Plot 1: Throughput comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for result in self.optimization_results['results']:
            if result['batch_results']:
                batch_sizes = [r['batch_size'] for r in result['batch_results']]
                throughputs = [r['samples_per_second'] for r in result['batch_results']]
                ax1.plot(batch_sizes, throughputs, marker='o', 
                        label=f"{result['config_name']} ({result['ngf']})")
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Memory usage
        for result in self.optimization_results['results']:
            if result['batch_results']:
                batch_sizes = [r['batch_size'] for r in result['batch_results']]
                memory = [r['peak_memory'] for r in result['batch_results']]
                ax2.plot(batch_sizes, memory, marker='s', 
                        label=f"{result['config_name']} ({result['ngf']})")
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'optimization_plots.png', dpi=150)
        plt.close()


class TrainingManager:
    """Main training orchestration manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.experiment_name = None
        self.data_path = None
        self.console = Console()
        
        # Print startup banner
        self.console.print(Panel.fit(
            "[bold green]🚀 CARS Training Manager[/bold green]\n"
            f"[dim]📁 Project root: {self.project_root}[/dim]",
            title="CARS-FASTGAN",
        ))
    
    def launch_training(
        self,
        data_path: str,
        experiment_name: Optional[str] = None,
        model_size: str = 'standard',
        batch_size: Optional[int] = None,
        max_epochs: int = 1000,
        preset: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = 'cars-fastgan',
        device: str = 'auto',
        num_workers: int = 4,
        resume_checkpoint: Optional[str] = None,
        dry_run: bool = False
    ) -> bool:
        """Launch training with specified configuration"""
        
        # Set paths
        self.data_path = Path(data_path).resolve()
        self.experiment_name = experiment_name or self._generate_experiment_name()
        
        # Check for optimization results and use them if available
        optimization_results = self._load_optimization_results()
        
        # Get configuration
        if preset:
            config = self._get_preset_config(preset)
        else:
            config = self._build_config(
                model_size=model_size,
                batch_size=batch_size,
                max_epochs=max_epochs,
                use_wandb=use_wandb,
                device=device,
                num_workers=num_workers
            )
        
        # Apply optimization results if available and batch_size not explicitly set
        if optimization_results and batch_size is None:
            optimized_config = self._apply_optimization_results(config, optimization_results)
            if optimized_config:
                print(f"\n📊 Using optimized configuration from hardware benchmarks")
                print(f"   Model: {optimized_config.get('model_size', model_size)}")
                print(f"   Batch size: {optimized_config.get('batch_size', config['batch_size'])}")
                config.update(optimized_config)
        
        # Build command
        cmd = self.build_launch_command(config)
        
        # Add resume checkpoint if provided
        if resume_checkpoint:
            cmd.append(f"resume_from_checkpoint={resume_checkpoint}")
        
        # Save configuration
        self.save_configuration(config)
        
        if dry_run:
            print("\n🔍 Dry run - Configuration:")
            print(json.dumps(config, indent=2))
            print("\n📋 Command that would be executed:")
            print(" ".join(cmd))
            return True
        
        # Launch training
        return self._execute_training(cmd)
    
    def _execute_training(self, cmd: List[str]) -> bool:
        """Execute training command with error handling"""
        print("\n📋 Training Configuration:")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Data: {self.data_path}")
        print(f"   Command: {' '.join(cmd[:3])}...")
        
        print("\n🚀 Launching training...")
        print("=" * 60)
        
        try:
            # Change to project root
            os.chdir(self.project_root)
            
            # Run training
            result = subprocess.run(cmd, check=True)
            
            print("\n✅ Training completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            traceback.print_exc()
            return False
    
    def build_launch_command(self, config: Dict[str, Any]) -> List[str]:
        """Build the launch command for training"""
        cmd = [
            "python", "main.py",
            f"experiment_name={self.experiment_name}",
            f"data_path={self.data_path}",
            f"max_epochs={config['max_epochs']}",
            # Disable gradient clipping for manual optimization
            "gradient_clip_val=0",
        ]
        
        # Use improved model config for improved preset
        if config.get('preset_name') == 'improved':
            cmd.append("model=fastgan_improved")
        
        # Add model configuration
        self._add_model_config_to_cmd(cmd, config)
        
        # Add data configuration
        cmd.extend([
            f"data.batch_size={config['batch_size']}",
            f"data.num_workers={config.get('num_workers', 4)}",
        ])
        
        # Add training configuration
        if 'training' in config:
            for key, value in config['training'].items():
                # These go under model.training, not just training
                if key in ['gradient_penalty_weight', 'ema_decay']:
                    cmd.append(f"model.training.{key}={value}")
                else:
                    cmd.append(f"training.{key}={value}")
        
        # Add W&B configuration
        if config.get('use_wandb', False):
            cmd.extend([
                "use_wandb=true",
                f"wandb.project={config.get('wandb_project', 'cars-fastgan')}",
            ])
        
        # Add device configuration
        if config.get('device'):
            cmd.append(f"accelerator={config['device']}")
        
        # Add any extra overrides (with + prefix for new keys)
        if 'extra_overrides' in config:
            cmd.extend(config['extra_overrides'])
        
        return cmd
    
    def _add_model_config_to_cmd(self, cmd: List[str], config: Dict[str, Any]):
        """Add model configuration to command"""
        model_config = config.get('model', {})
        
        # Handle nested model configuration
        for sub_key, sub_value in model_config.items():
            if isinstance(sub_value, dict):
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, list):
                        # Handle list values (e.g., layers)
                        layers_str = ",".join(map(str, sub_sub_value))
                        cmd.append(f"model.{sub_key}.{sub_sub_key}=[{layers_str}]")
                    elif isinstance(sub_sub_value, bool):
                        # Convert boolean to lowercase string
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={str(sub_sub_value).lower()}")
                    else:
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={sub_sub_value}")
            else:
                cmd.append(f"model.{sub_key}={sub_value}")
    
    def optimize_and_launch(
        self,
        data_path: str,
        device: str = 'auto',
        auto_launch: bool = True
    ) -> bool:
        """Run optimization and launch with optimal settings"""
        print("\n🔬 Running hardware optimization...")
        
        try:
            optimizer = HardwareOptimizer(device)
            recommendations = optimizer.benchmark_configurations()
            
            # Debug: print recommendations
            print(f"\nRecommendations: {recommendations}")
            
            # Save results
            opt_dir = self.project_root / "scripts/optimization_results"
            optimizer.save_optimization_results(opt_dir)
            
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            print(f"   Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False
        
        if not auto_launch:
            print("\n✅ Optimization complete!")
            return True
        
        # Launch with optimal settings
        if not recommendations:
            print("\n⚠️  No valid recommendations found!")
            return False
        
        best_config = recommendations.get('best_config', 'standard')
        best_batch = recommendations.get('best_batch_size', 8)
        
        print(f"\n📋 Optimal configuration: {best_config}")
        print(f"   Batch size: {best_batch}")
        print(f"   Precision: 16-mixed")
        
        # Create experiment name
        device_name = device.replace(':', '_') if device != 'auto' else optimizer.device_info['vendor'].lower()
        experiment_name = f"optimized_{best_config.lower()}_{device_name}"
        
        # Launch training with optimal settings
        # The launch_training method will now automatically use the optimization results
        return self.launch_training(
            data_path=data_path,
            experiment_name=experiment_name,
            model_size=best_config.lower(),
            device=device
        )
    
    def save_configuration(self, config: Dict[str, Any]):
        """Save training configuration"""
        config_dir = self.project_root / "experiments" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _generate_experiment_name(self, base_name: Optional[str] = None) -> str:
        """Generate experiment name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            return f"{base_name}_{timestamp}"
        return f"cars_fastgan_{timestamp}"
    
    def _load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """Load optimization results if available"""
        opt_file = self.project_root / "scripts/optimization_results/recommendations.json"
        
        if opt_file.exists():
            try:
                with open(opt_file, 'r') as f:
                    recommendations = json.load(f)
                print(f"📁 Found optimization results from {opt_file}")
                return recommendations
            except Exception as e:
                print(f"⚠️  Failed to load optimization results: {e}")
                return None
        return None
    
    def _apply_optimization_results(self, config: Dict[str, Any], optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization results to configuration"""
        optimized_config = {}
        
        # Get the model size from config
        model_size = config.get('model_size', 'standard')
        
        # Look for optimization results for this model size
        model_key = model_size.capitalize()  # Convert to match optimization keys
        
        if model_key in optimization_results:
            opt_data = optimization_results[model_key]
            optimized_config['batch_size'] = opt_data.get('optimal_batch_size', config['batch_size'])
            
            # Also check if this is the best overall config
            if optimization_results.get('best_config') == model_key:
                print(f"   ✨ This is the optimal configuration for your hardware!")
        
        # If we have a best_config recommendation and no specific model was requested
        elif 'best_config' in optimization_results and config.get('model_size') == 'standard':
            best_config = optimization_results['best_config']
            best_batch = optimization_results.get('best_batch_size', 8)
            
            print(f"   💡 Switching to optimal model: {best_config}")
            optimized_config['model_size'] = best_config.lower()
            optimized_config['batch_size'] = best_batch
            
            # Update model config
            optimized_config['model'] = self.get_model_config(best_config.lower())
        
        return optimized_config
    
    def _get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get predefined configuration preset"""
        presets = self.get_experiment_presets()
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        preset_config = presets[preset].copy()
        
        # Extract extra overrides before building config
        extra_overrides = preset_config.pop('extra_overrides', [])
        
        # Convert to full configuration
        config = self._build_config(
            model_size=preset_config.pop('model_size', 'standard'),
            batch_size=preset_config.pop('batch_size', 8),
            max_epochs=preset_config.pop('max_epochs', 1000),
            **preset_config
        )
        
        # Add preset name to config for reference
        config['preset_name'] = preset
        
        # Add back extra overrides
        if extra_overrides:
            config['extra_overrides'] = extra_overrides
        
        return config
    
    def _build_config(
        self,
        model_size: str = 'standard',
        batch_size: Optional[int] = None,
        max_epochs: int = 1000,
        use_wandb: bool = False,
        device: str = 'auto',
        num_workers: int = 4,
        **kwargs
    ) -> Dict[str, Any]:
        """Build complete configuration"""
        model_config = self.get_model_config(model_size)
        
        # Default batch size based on model size
        if batch_size is None:
            batch_size = {
                'micro': 32,
                'small': 24,
                'standard': 16,
                'large': 12,
                'xlarge': 8
            }.get(model_size.lower(), 16)
        
        config = {
            'model_size': model_size,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'use_wandb': use_wandb,
            'device': device,
            'num_workers': num_workers,
            'model': model_config,
        }
        
        # Add any additional kwargs
        config.update(kwargs)
        
        return config
    
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
                }
            },
            'fast': {
                'description': 'Fast training for quick experiments',
                'model_size': 'micro',
                'batch_size': 32,
                'max_epochs': 500,
                'training': {
                    'log_images_every_n_epochs': 10,
                    'val_check_interval': 50
                }
            },
            'high_quality': {
                'description': 'High quality with large model',
                'model_size': 'large',
                'batch_size': 8,
                'max_epochs': 2000,
                'loss': {
                    'feature_matching_weight': 30.0,
                    'perceptual_weight': 10.0
                }
            }
        }
    
    def show_available_checkpoints(self):
        """Show available checkpoints"""
        checkpoint_dir = self.project_root / "experiments" / "checkpoints"
        
        if not checkpoint_dir.exists():
            print("No checkpoints directory found.")
            return
        
        checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
        
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        print("\n📁 Available Checkpoints:")
        print("=" * 60)
        
        for i, ckpt in enumerate(checkpoints, 1):
            # Get file stats
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(ckpt.stat().st_mtime)
            
            print(f"\n{i}. {ckpt.name}")
            print(f"   Path: {ckpt}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CARS-FASTGAN Training Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run hardware optimization and auto-launch
  python cars_training_manager.py --auto_optimize --data_path data/processed
  
  # Launch with specific preset
  python cars_training_manager.py --preset improved --data_path data/processed
  
  # Launch multiple experiments
  python cars_training_manager.py --experiments micro standard large --data_path data/processed
  
  # Custom configuration
  python cars_training_manager.py --model_size large --batch_size 16 --max_epochs 2000
        """
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared data directory')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--preset', type=str, choices=['baseline', 'improved', 'fast', 'high_quality'],
                       help='Use a predefined configuration preset')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='standard',
                       choices=['micro', 'small', 'standard', 'large', 'xlarge'],
                       help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-determined if not specified)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum training epochs')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cuda:0, cuda:1, mps, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Optimization
    parser.add_argument('--auto_optimize', action='store_true',
                       help='Run hardware optimization and auto-launch with best settings')
    parser.add_argument('--optimize_only', action='store_true',
                       help='Only run optimization without launching training')
    
    # Multiple experiments
    parser.add_argument('--experiments', nargs='+',
                       help='Launch multiple experiments (e.g., --experiments micro standard large)')
    
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