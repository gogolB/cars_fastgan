#!/usr/bin/env python3
"""
CARS-FASTGAN Training Manager
High-level training orchestration with presets, optimization, and configuration tracking

Features:
- Predefined training presets for common configurations
- Hardware optimization for finding optimal batch sizes
- Comprehensive configuration tracking and visualization
- Support for multiple sequential experiments
- Integration with Weights & Biases
- Enhanced loss configurations support
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime
import time
import yaml
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

# Try rich for better terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
    from rich.tree import Tree
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import hardware optimizer if available
try:
    from scripts.optimize_for_hardware import HardwareOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


class ConfigTracker:
    """Track configuration values and their sources"""
    
    SOURCE_COLORS = {
        'yaml_config': 'cyan',
        'optimization': 'green',
        'preset': 'yellow',
        'command_line': 'bright_red',
        'additional_config': 'magenta'
    }
    
    def __init__(self):
        self.values = {}
        self.sources = {}
        self.history = []
    
    def set_value(self, key: str, value: Any, source: str):
        """Set a configuration value and track its source"""
        old_value = self.values.get(key)
        old_source = self.sources.get(key)
        
        self.values[key] = value
        self.sources[key] = source
        
        # Track history
        self.history.append({
            'key': key,
            'old_value': old_value,
            'new_value': value,
            'old_source': old_source,
            'new_source': source,
            'timestamp': datetime.now()
        })
    
    def set_nested_values(self, config: Dict, source: str):
        """Set nested configuration values"""
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config)
        for key, value in flat_config.items():
            self.set_value(key, value, source)
    
    def get_value(self, key: str, default=None):
        """Get a configuration value"""
        return self.values.get(key, default)
    
    def get_source(self, key: str):
        """Get the source of a configuration value"""
        return self.sources.get(key)
    
    def get_values_by_source(self, source: str):
        """Get all values from a specific source"""
        return {k: v for k, v in self.values.items() if self.sources[k] == source}
    
    def get_overrides_from_base(self):
        """Get all values that override the base configuration"""
        return {k: v for k, v in self.values.items() if self.sources[k] != 'yaml_config'}
    
    def print_summary(self):
        """Print configuration summary"""
        if RICH_AVAILABLE:
            table = Table(title="Configuration Summary", show_header=True)
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Source", style="bright_black")
            
            # Sort by source priority
            source_priority = ['yaml_config', 'optimization', 'preset', 'additional_config', 'command_line']
            sorted_items = sorted(
                self.values.items(),
                key=lambda x: (source_priority.index(self.sources[x[0]]) 
                              if self.sources[x[0]] in source_priority else 999, x[0])
            )
            
            for key, value in sorted_items:
                source = self.sources[key]
                color = self.SOURCE_COLORS.get(source.split(':')[0], 'white')
                table.add_row(
                    key,
                    str(value),
                    f"[{color}]{source}[/{color}]"
                )
            
            console.print(table)
        else:
            print("\nConfiguration Summary:")
            for key, value in sorted(self.values.items()):
                print(f"  {key}: {value} (from {self.sources[key]})")
    
    def print_history(self, last_n: int = None):
        """Print configuration change history"""
        history_to_show = self.history[-last_n:] if last_n else self.history
        
        if RICH_AVAILABLE:
            console.print("\n[bold]Configuration History:[/bold]")
            for entry in history_to_show:
                old_val = entry['old_value'] if entry['old_value'] is not None else "not set"
                console.print(
                    f"  [{self.SOURCE_COLORS.get(entry['new_source'].split(':')[0], 'white')}]"
                    f"{entry['key']}[/]: {old_val} → {entry['new_value']} "
                    f"([dim]{entry['new_source']}[/dim])"
                )
        else:
            print("\nConfiguration History:")
            for entry in history_to_show:
                old_val = entry['old_value'] if entry['old_value'] is not None else "not set"
                print(f"  {entry['key']}: {old_val} -> {entry['new_value']} (from {entry['new_source']})")


class CARSTrainingManager:
    """Manage CARS-FASTGAN training with presets and optimization"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.configs_dir = self.project_root / "configs"
        self.experiments_dir = self.project_root / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Load base configuration
        self.base_config = self._load_base_config()
        
        # Initialize config tracker
        self.config_tracker = ConfigTracker()
        
    def _load_base_config(self) -> Dict:
        """Load base configuration from yaml files"""
        # Load main config
        main_config_path = self.configs_dir / "config.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Load model config
        model_config_path = self.configs_dir / "model/fastgan.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f) or {}
                config['model'] = model_config
        
        # Load data config
        data_config_path = self.configs_dir / "data/cars_dataset.yaml"
        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f) or {}
                config['data'] = data_config
        
        return config
    
    def _load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """Load hardware optimization results if available"""
        opt_file = self.project_root / "scripts/optimization_results/recommendations.json"
        
        if opt_file.exists():
            console.print(f"[dim]📁 Found optimization results from {opt_file}[/dim]")
            with open(opt_file, 'r') as f:
                results = json.load(f)
                return results.get('recommendations', {})
        return None
    
    def get_experiment_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available experiment presets"""
        presets_file = self.configs_dir / "training_presets.json"
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    # Return the presets dictionary from the JSON file
                    return data.get('presets', {})
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load presets from {presets_file}: {e}[/yellow]")
        
        # Default presets if file doesn't exist or fails to load
        return {
            "baseline": {
                "description": "Standard FASTGAN configuration",
                "model": {
                    "generator": {"ngf": 64, "n_layers": 4},
                    "discriminator": {"ndf": 64, "n_layers": 4}
                }
            },
            "improved": {
                "description": "Enhanced configuration for better stability",
                "model": {
                    "generator": {"ngf": 64, "n_layers": 4},
                    "discriminator": {"ndf": 64, "n_layers": 4},
                    "loss": {"feature_matching_weight": 20.0, "gan_loss": "hinge"},
                    "optimizer": {
                        "generator": {"lr": 0.0001},
                        "discriminator": {"lr": 0.0004}
                    },
                    "training": {"use_gradient_penalty": True, "gradient_penalty_weight": 10.0}
                }
            },
            "small_dataset": {
                "description": "Optimized for small datasets like CARS microscopy",
                "data": {"batch_size": 8},
                "model": {
                    "generator": {"ngf": 64, "n_layers": 4},
                    "discriminator": {"ndf": 64, "n_layers": 3},
                    "loss": {"feature_matching_weight": 20.0, "gan_loss": "hinge"},
                    "optimizer": {
                        "generator": {"lr": 0.0001},
                        "discriminator": {"lr": 0.0004}
                    },
                    "training": {"use_ema": True, "ema_decay": 0.95}
                },
                "max_epochs": 2000,
                "log_images_every_n_epochs": 10
            }
        }
    
    def get_model_config(self, model_size: str) -> Dict[str, Dict[str, Any]]:
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
                'discriminator': {'ndf': 128, 'n_layers': 5}
            }
        }
        return configs.get(model_size, configs['standard'])
    
    def _generate_experiment_name(self, base: str = "cars_fastgan") -> str:
        """Generate unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{timestamp}"
    
    def _build_command_with_config_file(self, config_file: Path) -> List[str]:
        """Build command using config file"""
        cmd = ["python", "main.py", "--config-file", str(config_file)]
        return cmd
    
    def _write_complete_config_file(self, wandb_project: Optional[str], resume_checkpoint: Optional[str]) -> Path:
        """Write complete configuration to a file for main.py"""
        config_dir = self.experiments_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_name = self.config_tracker.get_value('experiment_name')
        config_file = config_dir / f"{experiment_name}_config.yaml"
        
        # Build complete config from tracker
        complete_config = {}
        
        # Convert flat keys back to nested structure
        for key, value in self.config_tracker.values.items():
            parts = key.split('.')
            current = complete_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        
        # Add special parameters that aren't in the tracker
        if wandb_project:
            if 'wandb' not in complete_config:
                complete_config['wandb'] = {}
            complete_config['wandb']['project'] = wandb_project
        
        if resume_checkpoint:
            complete_config['resume_from_checkpoint'] = resume_checkpoint
        
        # Write to file
        with open(config_file, 'w') as f:
            yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)
        
        return config_file
    
    def _save_complete_configuration(self):
        """Save complete configuration with all tracking information"""
        config_dir = self.experiments_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_name = self.config_tracker.get_value('experiment_name')
        
        # Convert all values to JSON-serializable format
        def make_serializable(obj):
            """Convert OmegaConf objects and other non-serializable types to JSON-serializable format"""
            if isinstance(obj, (DictConfig, dict)):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, ListConfig)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                # For custom objects, try to convert to dict
                return str(obj)
            else:
                return obj
        
        # Save complete tracked configuration
        config_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'values': make_serializable(self.config_tracker.values),
            'sources': self.config_tracker.sources,
            'history': [
                {
                    'key': h['key'],
                    'old_value': make_serializable(h['old_value']),
                    'new_value': make_serializable(h['new_value']),
                    'old_source': h['old_source'],
                    'new_source': h['new_source'],
                    'timestamp': h['timestamp'].isoformat()
                }
                for h in self.config_tracker.history
            ]
        }
        
        tracking_file = config_dir / f"{experiment_name}_tracking.json"
        with open(tracking_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        console.print(f"\n[dim]💾 Configuration tracking saved to: {tracking_file}[/dim]")
    
    def _execute_training(self, cmd: List[str]) -> bool:
        """Execute training directly without subprocess"""
        console.print("\n[green]🚀 Launching training...[/green]")
        
        try:
            # Print statistics before launch
            source_counts = {}
            for source in self.config_tracker.sources.values():
                source_type = source.split(':')[0]
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            if RICH_AVAILABLE:
                stats_table = Table(title="Configuration Statistics", show_header=True)
                stats_table.add_column("Source", style="cyan")
                stats_table.add_column("Count", style="green")
                
                for source, count in sorted(source_counts.items()):
                    color = self.config_tracker.SOURCE_COLORS.get(source, 'white')
                    stats_table.add_row(f"[{color}]{source}[/{color}]", str(count))
                
                stats_table.add_row("[bold]Total[/bold]", f"[bold]{len(self.config_tracker.values)}[/bold]")
                console.print(stats_table)
            else:
                print("\nConfiguration Statistics:")
                for source, count in sorted(source_counts.items()):
                    print(f"  {source}: {count} values")
                print(f"  Total: {len(self.config_tracker.values)} values")
            
            # Import and call the training function directly
            console.print("\n[dim]Starting training directly (no subprocess)...[/dim]\n")
            
            # Get the config file path from the command
            config_file = None
            for i, arg in enumerate(cmd):
                if arg == "--config-file" and i + 1 < len(cmd):
                    config_file = cmd[i + 1]
                    break
            
            if not config_file:
                console.print("[red]❌ No config file found in command[/red]")
                return False
            
            # Import main and run with the config file
            import importlib.util
            spec = importlib.util.spec_from_file_location("main", self.project_root / "main.py")
            main_module = importlib.util.module_from_spec(spec)
            sys.modules["main"] = main_module
            spec.loader.exec_module(main_module)
            
            # Prepare args for main
            sys.argv = cmd
            
            # Run main
            console.print(f"\n[cyan]Starting training with config: {config_file}[/cyan]\n")
            main_module.main()
            
            console.print("\n[green]✅ Training completed successfully![/green]")
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Training interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]❌ Training failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def _merge_configs(self, config: Dict, additional_config: Dict):
        """Merge additional config into existing config"""
        for key, value in additional_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._merge_configs(config[key], value)
            else:
                config[key] = value
    
    def _add_model_overrides(self, cmd: List[str], model_config: Dict):
        """Add model configuration overrides to command"""
        for sub_key, sub_value in model_config.items():
            if isinstance(sub_value, dict):
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, list):
                        # Handle list parameters
                        if all(isinstance(x, (int, float)) for x in sub_sub_value):
                            layers_str = '[' + ','.join(str(x) for x in sub_sub_value) + ']'
                        else:
                            # String list
                            layers_str = '[' + ','.join(f'"{x}"' if isinstance(x, str) else str(x) for x in sub_sub_value) + ']'
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={layers_str}")
                    elif isinstance(sub_sub_value, bool):
                        # Convert boolean to lowercase string
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={str(sub_sub_value).lower()}")
                    else:
                        cmd.append(f"model.{sub_key}.{sub_sub_key}={sub_sub_value}")
            else:
                cmd.append(f"model.{sub_key}={sub_value}")
    
    def create_experiment_name(self, base_name: Optional[str] = None) -> str:
        """Create a unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            return f"{base_name}_{timestamp}"
        return f"cars_fastgan_{timestamp}"
    
    def _apply_preset(self, preset_name: str) -> Dict[str, Any]:
        """Apply a preset configuration"""
        presets = self.get_experiment_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset_config = presets[preset_name]
        
        # Apply preset values to tracker
        for key, value in preset_config.items():
            if key == 'description':
                continue
            if isinstance(value, dict):
                self.config_tracker.set_nested_values({key: value}, f'preset:{preset_name}')
            else:
                self.config_tracker.set_value(key, value, f'preset:{preset_name}')
        
        return preset_config
    
    def _apply_optimization_results(self, optimization_results: Dict[str, Any]):
        """Apply optimization results to configuration"""
        optimal_config = optimization_results.get('optimal_config', '').lower()
        optimal_settings = optimization_results.get('optimal_settings', {})
        
        if optimal_config:
            # Apply model size
            model_config = self.get_model_config(optimal_config)
            for key, value in model_config.get('generator', {}).items():
                self.config_tracker.set_value(f'model.generator.{key}', value, 'optimization')
            for key, value in model_config.get('discriminator', {}).items():
                self.config_tracker.set_value(f'model.discriminator.{key}', value, 'optimization')
        
        # Apply batch size
        if 'batch_size' in optimal_settings:
            self.config_tracker.set_value('data.batch_size', optimal_settings['batch_size'], 'optimization')
    
    def _apply_command_line_overrides(self, **kwargs):
        """Apply command-line overrides"""
        override_mapping = {
            'model_size': None,  # Special handling
            'batch_size': 'data.batch_size',
            'max_epochs': 'max_epochs',
            'check_val_every_n_epoch': 'check_val_every_n_epoch',
            'log_images_every_n_epochs': 'log_images_every_n_epochs',
            'num_workers': 'data.num_workers',
            'use_wandb': 'use_wandb',
            'data_path': 'data_path',
            'experiment_name': 'experiment_name',
            'device': None  # Special handling
        }
        
        for arg_name, config_key in override_mapping.items():
            if arg_name in kwargs and kwargs[arg_name] is not None:
                if arg_name == 'model_size':
                    # Apply model size configuration
                    model_config = self.get_model_config(kwargs[arg_name])
                    for key, value in model_config.get('generator', {}).items():
                        self.config_tracker.set_value(f'model.generator.{key}', value, 'command_line')
                    for key, value in model_config.get('discriminator', {}).items():
                        self.config_tracker.set_value(f'model.discriminator.{key}', value, 'command_line')
                elif arg_name == 'device':
                    # Device configuration is handled separately
                    pass
                elif config_key:
                    self.config_tracker.set_value(config_key, kwargs[arg_name], 'command_line')
    
    def launch_training(
        self,
        data_path: str,
        experiment_name: Optional[str] = None,
        model_size: str = 'standard',
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        preset: Optional[str] = None,
        use_wandb: Optional[bool] = None,
        wandb_project: Optional[str] = None,
        device: str = 'auto',
        num_workers: Optional[int] = None,
        resume_checkpoint: Optional[str] = None,
        additional_config: Optional[Dict] = None,
        dry_run: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        log_images_every_n_epochs: Optional[int] = None,
        # Track which arguments were explicitly provided
        _provided_args: Optional[set] = None,
        _additional_overrides: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Launch training with complete configuration tracking"""
        
        console.print("\n[cyan]🔧 Building Configuration with Full Tracking[/cyan]")
        console.print("=" * 60)
        
        # Create a fresh config tracker starting with base config
        self.config_tracker = ConfigTracker()
        self.config_tracker.set_nested_values(self.base_config, 'yaml_config')
        
        # Validate data path
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            console.print(f"[red]❌ Data path not found: {data_path}[/red]")
            return False
        
        # Apply optimization results if available
        optimization_results = self._load_optimization_results()
        if optimization_results:
            console.print("\n[green]🔧 Applying optimization results...[/green]")
            self._apply_optimization_results(optimization_results)
        
        # Apply preset if specified
        preset_config = {}
        if preset:
            console.print(f"\n[yellow]📋 Applying preset: {preset}[/yellow]")
            preset_config = self._apply_preset(preset)
            
            # Handle special presets that use enhanced config
            if preset_config.get('config_name') == 'fastgan_enhanced':
                # Load the enhanced config
                enhanced_config_path = self.configs_dir / "model/fastgan_enhanced.yaml"
                if enhanced_config_path.exists():
                    with open(enhanced_config_path, 'r') as f:
                        enhanced_config = yaml.safe_load(f) or {}
                        self.config_tracker.set_nested_values({'model': enhanced_config}, f'preset:{preset}:enhanced_config')
        
        # Apply command-line overrides - only for explicitly provided values
        console.print("\n[bright_red]⚡ Applying command-line overrides...[/bright_red]")
        
        # Build kwargs with only explicitly provided values
        override_kwargs = {}
        
        # Use _provided_args to check which were explicitly provided
        if _provided_args is None:
            _provided_args = set()
        
        # Only add to override_kwargs if explicitly provided
        if 'model_size' in _provided_args and model_size != 'standard':
            override_kwargs['model_size'] = model_size
        if 'batch_size' in _provided_args:
            override_kwargs['batch_size'] = batch_size
        if 'max_epochs' in _provided_args:
            override_kwargs['max_epochs'] = max_epochs
        if 'check_val_every_n_epoch' in _provided_args:
            override_kwargs['check_val_every_n_epoch'] = check_val_every_n_epoch
        if 'log_images_every_n_epochs' in _provided_args:
            override_kwargs['log_images_every_n_epochs'] = log_images_every_n_epochs
        if 'num_workers' in _provided_args:
            override_kwargs['num_workers'] = num_workers
        if 'use_wandb' in _provided_args:
            override_kwargs['use_wandb'] = use_wandb
        if 'device' in _provided_args and device != 'auto':
            override_kwargs['device'] = device
        
        # Always set these as they're required
        override_kwargs['data_path'] = str(data_path)
        override_kwargs['experiment_name'] = experiment_name or self._generate_experiment_name(preset or model_size)
        
        self._apply_command_line_overrides(**override_kwargs)
        
        # Apply additional dotted overrides with highest priority
        if _additional_overrides:
            console.print("\n[bright_red]⚡ Applying additional command-line overrides...[/bright_red]")
            for key, value in _additional_overrides.items():
                self.config_tracker.set_value(key, value, 'command_line')
                console.print(f"  [bright_red]{key} = {value}[/bright_red]")
        
        # Apply additional config if provided
        if additional_config:
            console.print("\n[magenta]📝 Applying additional configuration...[/magenta]")
            self.config_tracker.set_nested_values(additional_config, 'additional_config')
        
        # Print configuration summary
        console.print("\n[bold]📊 Final Configuration Summary:[/bold]")
        self.config_tracker.print_summary()
        
        # Print override history
        console.print("\n[bold]📜 Override History:[/bold]")
        self.config_tracker.print_history(last_n=20)  # Show last 20 changes
        
        # Get all overrides from base
        overrides = self.config_tracker.get_overrides_from_base()
        console.print(f"\n[bold]Total overrides from base configuration: {len(overrides)}[/bold]")
        
        # Build command with complete config file instead of overrides
        config_file = self._write_complete_config_file(wandb_project, resume_checkpoint)
        cmd = self._build_command_with_config_file(config_file)
        
        # Save complete configuration
        self._save_complete_configuration()
        
        if dry_run:
            console.print("\n[yellow]🔍 Dry run - Command that would be executed:[/yellow]")
            console.print(" ".join(cmd[:3]))
            for c in cmd[3:]:
                console.print(f"    {c} \\")
            
            console.print(f"\n[yellow]Total command line arguments: {len(cmd) - 2}[/yellow]")
            return True
        
        # Launch training
        return self._execute_training(cmd)
    
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
            print("\n✅ Optimization complete! Results saved.")
            return True
        
        # Launch with optimal settings
        optimal_config = recommendations['optimal_config']
        optimal_settings = recommendations['optimal_settings']
        
        print(f"\n🚀 Launching with optimal configuration: {optimal_config}")
        print(f"   Batch size: {optimal_settings['batch_size']}")
        
        return self.launch_training(
            data_path=data_path,
            model_size=optimal_config,
            batch_size=optimal_settings['batch_size'],
            device=device
        )


def main():
    parser = argparse.ArgumentParser(
        description='CARS-FASTGAN Training Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a preset configuration
  python cars_training_manager.py --preset improved --data_path data/processed
  
  # Use enhanced losses preset
  python cars_training_manager.py --preset enhanced_losses --data_path data/processed
  
  # Run hardware optimization first
  python cars_training_manager.py --auto_optimize --data_path data/processed
  
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
    
    # Training configuration - use None as default to detect if provided
    parser.add_argument('--model_size', type=str, default='standard',
                       choices=['micro', 'small', 'standard', 'large', 'xlarge'],
                       help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-determined if not specified)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum training epochs')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None,
                       help='Run validation every N epochs')
    parser.add_argument('--log_images_every_n_epochs', type=int, default=None,
                       help='Log generated images every N epochs')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cuda:0, cuda:1, mps, cpu)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')
    
    # Multiple experiments
    parser.add_argument('--experiments', nargs='+', type=str,
                       help='Run multiple experiments (presets or model sizes)')
    
    # Optimization
    parser.add_argument('--auto_optimize', action='store_true',
                       help='Run hardware optimization and launch with optimal settings')
    parser.add_argument('--optimize_only', action='store_true',
                       help='Only run optimization without training')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='W&B project name')
    
    # Other options
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without launching training')
    
    # Parse known args to handle additional overrides
    args, unknown = parser.parse_known_args()
    
    # Parse additional overrides (e.g., model.loss.use_ssim_loss=true)
    additional_overrides = {}
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert string values to appropriate types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            
            additional_overrides[key] = value
    
    # Track which arguments were explicitly provided
    provided_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            arg_name = arg[2:].split('=')[0]
            provided_args.add(arg_name.replace('-', '_'))
    
    # Create manager
    manager = CARSTrainingManager()
    
    # Handle different modes
    if args.optimize_only:
        if not OPTIMIZER_AVAILABLE:
            print("❌ Hardware optimizer not available!")
            print("   Make sure optimize_for_hardware.py is in scripts/")
            return
        
        success = manager.optimize_and_launch(
            data_path=args.data_path,
            device=args.device,
            auto_launch=False
        )
    elif args.auto_optimize:
        if not OPTIMIZER_AVAILABLE:
            print("❌ Hardware optimizer not available!")
            print("   Make sure optimize_for_hardware.py is in scripts/")
            return
        
        success = manager.optimize_and_launch(
            data_path=args.data_path,
            device=args.device,
            auto_launch=True
        )
    elif args.experiments:
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
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Experiment Summary:")
        for exp, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {exp}: {status}")
    else:
        # Single experiment
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
            log_images_every_n_epochs=args.log_images_every_n_epochs,
            _provided_args=provided_args,
            _additional_overrides=additional_overrides
        )
    
    if not args.dry_run:
        print("\n✨ Done!")


if __name__ == "__main__":
    main()