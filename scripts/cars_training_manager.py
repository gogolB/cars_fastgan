#!/usr/bin/env python3
"""
CARS-FASTGAN Training Manager
Comprehensive training orchestration with hardware optimization
Enhanced with proper preset override handling and detailed logging
"""

import os
import sys
import json
import argparse
import subprocess
import psutil
import torch
import torch.backends.mps
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import time
import traceback
from dataclasses import dataclass, field

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import GPU utilities
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Rich console imports (optional but recommended)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False
    # Fallback printing
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()


class ConfigTracker:
    """Track configuration values and their sources"""
    
    def __init__(self):
        self.values = {}
        self.sources = {}
        
    def set_value(self, key: str, value: Any, source: str):
        """Set a configuration value and track its source"""
        old_value = self.values.get(key)
        old_source = self.sources.get(key, "default")
        
        self.values[key] = value
        self.sources[key] = source
        
        # Log the override
        if old_value is not None and old_value != value:
            console.print(f"[yellow]📝 Override: {key}[/yellow]")
            console.print(f"   [dim]Previous: {old_value} (from {old_source})[/dim]")
            console.print(f"   [green]New: {value} (from {source})[/green]")
        elif old_value is None:
            console.print(f"[blue]📌 Setting: {key} = {value} (from {source})[/blue]")
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.values.get(key, default)
    
    def get_source(self, key: str) -> str:
        """Get the source of a configuration value"""
        return self.sources.get(key, "default")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.values.copy()
    
    def print_summary(self):
        """Print a summary of all configuration values and sources"""
        console.print("\n[cyan]📊 Configuration Summary:[/cyan]")
        console.print("=" * 60)
        for key in sorted(self.values.keys()):
            console.print(f"{key}: {self.values[key]} [dim](from {self.sources[key]})[/dim]")
        console.print("=" * 60)


class HardwareOptimizer:
    """Optimize training configurations for available hardware"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.system_memory = psutil.virtual_memory().total / (1024**3)
        self.device_info = self._get_device_info()
        self.optimization_results = {}
        
        # Display hardware info
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold cyan]🖥️  Hardware Optimizer[/bold cyan]\n"
                f"   Device: {self.device}\n"
                f"   Device Info: {self.device_info['description']}\n"
                f"   VRAM: {self.device_info['vram_gb']:.1f} GB\n"
                f"   System RAM: {self.system_memory:.1f} GB",
                title="Hardware Configuration"
            ))
        else:
            print("🖥️  Hardware Optimizer")
            print(f"   Device: {self.device}")
            print(f"   Device Info: {self.device_info['description']}")
            print(f"   VRAM: {self.device_info['vram_gb']:.1f} GB")
            print(f"   System RAM: {self.system_memory:.1f} GB")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate torch device"""
        if device != 'auto':
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            'device': str(self.device),
            'vram_gb': 0,
            'description': 'Unknown',
            'vendor': 'Unknown'
        }
        
        if self.device.type == 'cuda':
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        info['vram_gb'] = gpu.memoryTotal / 1024
                        info['description'] = gpu.name
                        info['vendor'] = 'NVIDIA'
                except:
                    pass
            
            if info['vram_gb'] == 0:  # Fallback
                props = torch.cuda.get_device_properties(0)
                info['vram_gb'] = props.total_memory / (1024**3)
                info['description'] = props.name
                info['vendor'] = 'NVIDIA'
                
        elif self.device.type == 'mps':
            info['description'] = 'Apple Silicon GPU'
            info['vendor'] = 'Apple'
            # Estimate based on system
            if self.system_memory >= 32:
                info['vram_gb'] = 24  # M1 Max/Ultra
            elif self.system_memory >= 16:
                info['vram_gb'] = 16  # M1 Pro
            else:
                info['vram_gb'] = 8   # M1
                
        else:
            info['description'] = 'CPU'
            info['vendor'] = 'CPU'
            info['vram_gb'] = self.system_memory * 0.8  # Can use most system RAM
        
        return info
    
    def benchmark_configurations(self, configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Benchmark different model configurations"""
        if configs is None:
            configs = self._get_default_configs()
        
        results = []
        
        for config in configs:
            console.print(f"\n[yellow]Testing {config['name']}...[/yellow]")
            result = self._benchmark_config(config)
            results.append(result)
            
            if result['batch_results']:
                optimal = result['batch_results'][-1]
                console.print(f"   ✓ Max batch size: {optimal['batch_size']}")
                console.print(f"   ✓ Throughput: {optimal['throughput']:.1f} img/s")
        
        self.optimization_results = {
            'device_info': self.device_info,
            'system_memory_gb': self.system_memory,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(results)
        }
        
        return self.optimization_results['recommendations']
    
    def _get_default_configs(self) -> List[Dict[str, Any]]:
        """Get default configurations to test"""
        return [
            {'name': 'Micro', 'ngf': 32, 'ndf': 32, 'n_layers': 3},
            {'name': 'Small', 'ngf': 48, 'ndf': 48, 'n_layers': 3},
            {'name': 'Standard', 'ngf': 64, 'ndf': 64, 'n_layers': 4},
            {'name': 'Large', 'ngf': 96, 'ndf': 96, 'n_layers': 4},
        ]
    
    def _benchmark_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single configuration"""
        from src.models.fastgan import FastGAN
        
        result = {
            'config_name': config['name'],
            'ngf': config['ngf'],
            'ndf': config['ndf'],
            'batch_results': []
        }
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            try:
                # Create model
                model = FastGAN(
                    img_size=512,
                    channels=1,
                    latent_dim=256,
                    ngf=config['ngf'],
                    ndf=config['ndf'],
                    n_layers=config['n_layers']
                ).to(self.device)
                
                # Test forward pass
                start_time = time.time()
                
                # Warmup
                for _ in range(3):
                    noise = torch.randn(batch_size, 256).to(self.device)
                    with torch.no_grad():
                        _ = model.generator(noise)
                
                # Actual benchmark
                num_iterations = 10
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
                bench_start = time.time()
                for _ in range(num_iterations):
                    noise = torch.randn(batch_size, 256).to(self.device)
                    with torch.no_grad():
                        _ = model.generator(noise)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                bench_time = time.time() - bench_start
                
                throughput = (batch_size * num_iterations) / bench_time
                
                # Get memory usage
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()
                else:
                    peak_memory = psutil.Process().memory_info().rss / (1024**3)
                
                result['batch_results'].append({
                    'batch_size': batch_size,
                    'throughput': throughput,
                    'peak_memory': peak_memory,
                    'time_per_batch': bench_time / num_iterations
                })
                
                del model
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Hit memory limit
                console.print(f"   [red]✗ Batch size {batch_size} failed: OOM[/red]")
                break
        
        return result
    
    def _generate_recommendations(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate recommendations based on benchmark results"""
        recommendations = {}
        
        # Find configuration with best throughput
        best_throughput = 0
        best_config = None
        best_batch = None
        
        for result in results:
            if result['batch_results']:
                # Get the largest working batch size
                optimal = result['batch_results'][-1]
                if optimal['throughput'] > best_throughput:
                    best_throughput = optimal['throughput']
                    best_config = result['config_name']
                    best_batch = optimal['batch_size']
        
        recommendations['optimal_config'] = best_config
        recommendations['optimal_settings'] = {
            'batch_size': best_batch,
            'precision': '16-mixed' if self.device.type == 'cuda' else '32-true'
        }
        
        # Device-specific optimizations
        if self.device.type == 'mps':
            recommendations['optimal_settings']['optimization_flags'] = [
                'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7',
                'export PYTORCH_ENABLE_MPS_FALLBACK=1'
            ]
        
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
        
        console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")


class CARSTrainingManager:
    """Main training orchestration class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.experiments_dir = self.project_root / "experiments"
        self.configs_dir = self.project_root / "configs"
        
        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config tracker
        self.config_tracker = ConfigTracker()
        
        # Print header
        self._print_header()
    
    def _print_header(self):
        """Print welcome header"""
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold cyan]🚀 CARS Training Manager[/bold cyan]\n"
                f"[dim]📁 Project root: {self.project_root}[/dim]",
                border_style="cyan"
            ))
        else:
            print("╭────────────── CARS-FASTGAN ──────────────╮")
            print("│ 🚀 CARS Training Manager                 │")
            print(f"│ 📁 Project root: {self.project_root} │")
            print("╰──────────────────────────────────────────╯")
    
    def get_experiment_presets(self) -> Dict[str, Dict]:
        """Get predefined experiment configurations from configs/training_presets.json"""
        # Load from the official training_presets.json file
        presets_file = self.configs_dir / "training_presets.json"
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    presets = data.get('presets', {})
                    console.print(f"[green]✓ Loaded presets from {presets_file}[/green]")
                    return presets
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load presets from {presets_file}: {e}[/yellow]")
                console.print("[yellow]Using fallback presets[/yellow]")
        else:
            console.print(f"[yellow]Warning: Presets file not found at {presets_file}[/yellow]")
            console.print("[yellow]Using fallback presets[/yellow]")
        
        # Fallback to minimal presets if file not found
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
            'small_dataset': {
                'description': 'Optimized for small datasets like CARS microscopy',
                'model_size': 'standard',
                'batch_size': 8,
                'max_epochs': 2000,
                'check_val_every_n_epoch': 1,
                'log_images_every_n_epochs': 1,
                'loss': {
                    'gan_loss': 'hinge',
                    'feature_matching_weight': 20.0
                },
                'optimizer': {
                    'generator': {'lr': 0.0001},
                    'discriminator': {'lr': 0.0004}
                }
            }
        }
    
    def get_model_config(self, model_size: str) -> Dict[str, Any]:
        """Get model configuration by size"""
        # First try to load from training_presets.json
        presets_file = self.configs_dir / "training_presets.json"
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    model_configs = data.get('model_configs', {})
                    if model_size.lower() in model_configs:
                        return model_configs[model_size.lower()]
            except Exception:
                pass
        
        # Fallback to hardcoded configs
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
        dry_run: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        log_images_every_n_epochs: Optional[int] = None
    ) -> bool:
        """Launch training with specified configuration"""
        
        console.print("\n[cyan]🔧 Building Configuration[/cyan]")
        console.print("Priority: command-line > preset > optimization > defaults")
        console.print("=" * 60)
        
        # Reset config tracker for this launch
        self.config_tracker = ConfigTracker()
        
        # Validate data path
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            console.print(f"[red]❌ Data path not found: {data_path}[/red]")
            return False
        
        # Step 1: Load default values from config files
        self.config_tracker.set_value('model_size', model_size, 'default')
        self.config_tracker.set_value('batch_size', 16, 'default')  # Default batch size
        self.config_tracker.set_value('max_epochs', 1000, 'default')
        self.config_tracker.set_value('check_val_every_n_epoch', 10, 'config_file')  # From config file
        self.config_tracker.set_value('log_images_every_n_epochs', 25, 'config_file')  # From config file
        
        # Step 2: Apply optimization results if available
        optimization_results = self._load_optimization_results()
        if optimization_results:
            optimal_config = optimization_results.get('optimal_config', '').lower()
            optimal_settings = optimization_results.get('optimal_settings', {})
            
            if optimal_config:
                self.config_tracker.set_value('model_size', optimal_config, 'optimization')
            if 'batch_size' in optimal_settings:
                self.config_tracker.set_value('batch_size', optimal_settings['batch_size'], 'optimization')
        
        # Step 3: Apply preset if specified
        if preset:
            preset_config = self._get_preset_config(preset)
            
            # Apply preset values
            for key in ['model_size', 'batch_size', 'max_epochs', 'check_val_every_n_epoch', 'log_images_every_n_epochs']:
                if key in preset_config and preset_config[key] is not None:
                    self.config_tracker.set_value(key, preset_config[key], f'preset:{preset}')
            
            # Store other preset configs for later use
            preset_config['preset_name'] = preset
        else:
            preset_config = {}
        
        # Step 4: Apply command-line overrides (highest priority)
        if model_size != 'standard':  # Only if not default
            self.config_tracker.set_value('model_size', model_size, 'command_line')
        if batch_size is not None:
            self.config_tracker.set_value('batch_size', batch_size, 'command_line')
        if max_epochs != 1000:  # Only if not default
            self.config_tracker.set_value('max_epochs', max_epochs, 'command_line')
        if check_val_every_n_epoch is not None:
            self.config_tracker.set_value('check_val_every_n_epoch', check_val_every_n_epoch, 'command_line')
        if log_images_every_n_epochs is not None:
            self.config_tracker.set_value('log_images_every_n_epochs', log_images_every_n_epochs, 'command_line')
        
        # Print configuration summary
        self.config_tracker.print_summary()
        
        # Build final configuration
        config = {
            'experiment_name': experiment_name or self._generate_experiment_name(preset or self.config_tracker.get_value('model_size')),
            'data_path': str(data_path),
            'model_size': self.config_tracker.get_value('model_size'),
            'batch_size': self.config_tracker.get_value('batch_size'),
            'max_epochs': self.config_tracker.get_value('max_epochs'),
            'check_val_every_n_epoch': self.config_tracker.get_value('check_val_every_n_epoch'),
            'log_images_every_n_epochs': self.config_tracker.get_value('log_images_every_n_epochs'),
            'use_wandb': use_wandb,
            'device': device,
            'num_workers': num_workers,
        }
        
        # Get model configuration
        model_config = self.get_model_config(config['model_size'])
        config['model'] = model_config
        
        # Add preset-specific configurations
        if preset_config:
            for key in ['loss', 'optimizer', 'training', 'callbacks']:
                if key in preset_config:
                    config[key] = preset_config[key]
            if 'preset_name' in preset_config:
                config['preset_name'] = preset_config['preset_name']
        
        # Apply additional config if provided
        if additional_config:
            self._merge_configs(config, additional_config)
        
        # Build and execute command
        cmd = self._build_launch_command(config)
        
        # Add resume checkpoint if provided
        if resume_checkpoint:
            cmd.append(f"+resume_from_checkpoint={resume_checkpoint}")
        
        # Save configuration
        self._save_configuration(config)
        
        if dry_run:
            console.print("\n[yellow]🔍 Dry run - Configuration:[/yellow]")
            console.print(json.dumps(config, indent=2))
            console.print("\n[yellow]📋 Command that would be executed:[/yellow]")
            console.print(" ".join(cmd))
            return True
        
        # Launch training
        return self._execute_training(cmd)
    
    def _get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get predefined configuration preset"""
        presets = self.get_experiment_presets()
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        # Return the full preset config
        return presets[preset].copy()
    
    def _build_launch_command(self, config: Dict[str, Any]) -> List[str]:
        """Build the training command with all overrides"""
        cmd = [
            "python", "main.py",
            f"experiment_name={config['experiment_name']}",
            f"data_path={config['data_path']}",
            f"max_epochs={config['max_epochs']}",
            "gradient_clip_val=0",  # Disable gradient clipping for manual optimization
        ]
        
        # CRITICAL: Add top-level config overrides FIRST
        if 'check_val_every_n_epoch' in config and config['check_val_every_n_epoch'] is not None:
            cmd.append(f"check_val_every_n_epoch={config['check_val_every_n_epoch']}")
        
        if 'log_images_every_n_epochs' in config and config['log_images_every_n_epochs'] is not None:
            cmd.append(f"log_images_every_n_epochs={config['log_images_every_n_epochs']}")
        
        # Handle special presets
        if config.get('preset_name') == 'improved':
            cmd.append("model=fastgan_improved")
        elif config.get('preset_name') == 'small_dataset':
            cmd.append("model=fastgan_improved")
            cmd.append("model.training.ema_decay=0.95")
        
        # Add model configuration
        if 'model' in config:
            for model_key, model_value in config['model'].items():
                if isinstance(model_value, dict):
                    for sub_key, sub_value in model_value.items():
                        cmd.append(f"model.{model_key}.{sub_key}={sub_value}")
                else:
                    cmd.append(f"model.{model_key}={model_value}")
        
        # Add data configuration
        cmd.append(f"data.batch_size={config['batch_size']}")
        cmd.append(f"data.num_workers={config.get('num_workers', 4)}")
        
        # Add other configurations
        if 'loss' in config:
            for key, value in config['loss'].items():
                cmd.append(f"model.loss.{key}={value}")
        
        if 'optimizer' in config:
            for opt_type in ['generator', 'discriminator']:
                if opt_type in config['optimizer']:
                    for key, value in config['optimizer'][opt_type].items():
                        cmd.append(f"model.optimizer.{opt_type}.{key}={value}")
        
        if 'training' in config:
            for key, value in config['training'].items():
                cmd.append(f"model.training.{key}={value}")
        
        if config.get('use_wandb'):
            cmd.append("use_wandb=true")
            if 'wandb_project' in config:
                cmd.append(f"wandb.project={config['wandb_project']}")
        
        # Device configuration
        device = config.get('device', 'auto')
        if device != 'auto':
            if device in ['gpu', 'cuda']:
                cmd.extend(['accelerator=gpu', 'devices=1'])
            elif device == 'mps':
                cmd.extend(['accelerator=mps', 'devices=1'])
            elif device == 'cpu':
                cmd.extend(['accelerator=cpu', 'devices=1'])
        
        return cmd
    
    def _execute_training(self, cmd: List[str]) -> bool:
        """Execute the training command"""
        console.print("\n[green]📋 Training Configuration:[/green]")
        console.print(f"   Command: {' '.join(cmd[:3])}...")
        
        console.print("\n[green]🚀 Launching training...[/green]")
        console.print("=" * 60)
        
        try:
            # Change to project root
            os.chdir(self.project_root)
            
            # Run training
            result = subprocess.run(cmd, check=True)
            
            console.print("\n[green]✅ Training completed successfully![/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"\n[red]❌ Training failed with exit code {e.returncode}[/red]")
            return False
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Training interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
            traceback.print_exc()
            return False
    
    def optimize_and_launch(
        self,
        data_path: str,
        device: str = 'auto',
        auto_launch: bool = True
    ) -> bool:
        """Run optimization and optionally launch training"""
        console.print("\n[cyan]🔬 Running hardware optimization...[/cyan]")
        
        try:
            optimizer = HardwareOptimizer(device)
            recommendations = optimizer.benchmark_configurations()
            
            # Save results
            opt_dir = self.project_root / "scripts/optimization_results"
            optimizer.save_optimization_results(opt_dir)
            
            if not auto_launch:
                console.print("\n[green]✅ Optimization complete![/green]")
                return True
            
            # Launch with optimal settings
            optimal_config = recommendations.get('optimal_config', 'standard')
            optimal_settings = recommendations.get('optimal_settings', {})
            
            return self.launch_training(
                data_path=data_path,
                model_size=optimal_config.lower(),
                batch_size=optimal_settings.get('batch_size', 8),
                device=device,
                experiment_name=f"optimized_{optimal_config.lower()}"
            )
            
        except Exception as e:
            console.print(f"\n[red]❌ Optimization failed: {e}[/red]")
            traceback.print_exc()
            return False
    
    def launch_multiple_experiments(
        self,
        data_path: str,
        experiments: List[str],
        base_config: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """Launch multiple experiments sequentially"""
        results = {}
        
        console.print(f"\n[cyan]🚀 Launching {len(experiments)} experiments...[/cyan]")
        
        for i, experiment in enumerate(experiments, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]📊 Experiment {i}/{len(experiments)}: {experiment}[/bold]")
            console.print(f"{'='*60}")
            
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
                console.print(f"\n[yellow]⚠️  Experiment {experiment} failed![/yellow]")
                continue_prompt = input("Continue with remaining experiments? (y/n): ")
                if continue_prompt.lower() != 'y':
                    break
        
        return results
    
    def show_available_checkpoints(self) -> List[Path]:
        """Show available checkpoints"""
        checkpoint_dir = self.project_root / "experiments" / "checkpoints"
        
        if not checkpoint_dir.exists():
            console.print("[yellow]No checkpoints directory found.[/yellow]")
            return []
        
        checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
        
        if not checkpoints:
            console.print("[yellow]No checkpoints found.[/yellow]")
            return []
        
        console.print("\n[cyan]📁 Available Checkpoints:[/cyan]")
        console.print("=" * 60)
        
        for i, ckpt in enumerate(sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            console.print(f"{i+1}. {ckpt.name} ({size_mb:.1f} MB, {mod_time})")
        
        return checkpoints
    
    def _generate_experiment_name(self, base_name: Optional[str] = None) -> str:
        """Generate unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            return f"cars_fastgan_{base_name}_{timestamp}"
        return f"cars_fastgan_{timestamp}"
    
    def _load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """Load cached optimization results"""
        opt_file = self.project_root / "scripts/optimization_results/recommendations.json"
        
        if opt_file.exists():
            try:
                with open(opt_file, 'r') as f:
                    results = json.load(f)
                console.print(f"[dim]📁 Found optimization results from {opt_file}[/dim]")
                return results
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load optimization results: {e}[/yellow]")
        
        return None
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        config_dir = self.experiments_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f"{config['experiment_name']}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also save the sources for debugging
        sources_file = config_dir / f"{config['experiment_name']}_sources.json"
        with open(sources_file, 'w') as f:
            json.dump({
                'values': self.config_tracker.values,
                'sources': self.config_tracker.sources
            }, f, indent=2)
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value


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
  
  # Launch with small_dataset preset (recommended for CARS)
  python cars_training_manager.py --preset small_dataset --data_path data/processed
  
  # Launch multiple experiments
  python cars_training_manager.py --experiments micro standard large --data_path data/processed
  
  # Custom configuration with overrides
  python cars_training_manager.py --preset small_dataset --data_path data/processed \\
      --check_val_every_n_epoch 1 --log_images_every_n_epochs 1
        """
    )
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared data directory')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    # Get available presets from the config file
    temp_manager = CARSTrainingManager()
    available_presets = list(temp_manager.get_experiment_presets().keys())
    
    parser.add_argument('--preset', type=str, 
                       choices=available_presets,
                       help='Use a predefined configuration preset from configs/training_presets.json')
    
    # Training configuration
    parser.add_argument('--model_size', type=str, default='standard',
                       choices=['micro', 'small', 'standard', 'large', 'xlarge'],
                       help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-determined if not specified)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum training epochs')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None,
                       help='Run validation every N epochs')
    parser.add_argument('--log_images_every_n_epochs', type=int, default=None,
                       help='Log generated images every N epochs')
    
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
    
    # Parse once to get preset
    args, _ = parser.parse_known_args()
    
    # If preset is specified, show which presets are available
    if args.preset:
        console.print(f"\n[cyan]Using preset: {args.preset}[/cyan]")
        preset_desc = temp_manager.get_experiment_presets().get(args.preset, {}).get('description', '')
        if preset_desc:
            console.print(f"[dim]{preset_desc}[/dim]")
    
    # Parse fully
    args = parser.parse_args()
    
    # Create training manager
    manager = CARSTrainingManager()
    
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
        
        console.print("\n[green]✅ Optimization complete![/green]")
        console.print(f"[dim]📁 Results saved to: {opt_dir}[/dim]")
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
                'num_workers': args.num_workers,
                'check_val_every_n_epoch': args.check_val_every_n_epoch,
                'log_images_every_n_epochs': args.log_images_every_n_epochs
            }
        )
        
        success_count = sum(results.values())
        console.print(f"\n[green]✅ Completed {success_count}/{len(results)} experiments successfully[/green]")
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
        dry_run=args.dry_run,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_images_every_n_epochs=args.log_images_every_n_epochs
    )
    
    if success and not args.dry_run:
        console.print("\n[green]✨ Training completed successfully![/green]")
    elif not success and not args.dry_run:
        console.print("\n[red]❌ Training failed![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()