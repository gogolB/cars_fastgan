#!/usr/bin/env python3
"""
CARS-FASTGAN Training Manager - Complete Tracking Version
Tracks EVERY configuration value through the entire override chain

Priority chain (highest to lowest):
1. Command line arguments
2. Preset configurations  
3. Optimization results
4. Config files (loaded from YAML)
"""

import json
import subprocess
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import traceback
from omegaconf import OmegaConf, DictConfig, ListConfig
import yaml
from copy import deepcopy

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.tree import Tree
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    # Fallback print function
    class FakeConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FakeConsole()

# Import hardware optimizer if available
try:
    from hardware_optimizer import HardwareOptimizer
except ImportError:
    HardwareOptimizer = None


class ConfigTracker:
    """Track ALL configuration values and their sources with color coding"""
    
    # Color codes for different sources
    SOURCE_COLORS = {
        'yaml_config': 'blue',          # From YAML config files
        'optimization': 'green',        # From optimization results
        'preset': 'yellow',             # From preset configurations
        'command_line': 'bright_red'    # Command line overrides
    }
    
    # ANSI color codes for fallback
    ANSI_COLORS = {
        'yaml_config': '\033[94m',   # Blue
        'optimization': '\033[92m',  # Green
        'preset': '\033[93m',        # Yellow
        'command_line': '\033[91m'   # Red
    }
    ANSI_RESET = '\033[0m'
    
    def __init__(self):
        self.values = {}      # Flat dictionary of all values
        self.sources = {}     # Source for each value
        self.history = []     # Track all changes
        self.nested = {}      # Nested structure for visualization
    
    def set_value(self, key: str, value: Any, source: str):
        """Set a configuration value and track its source"""
        old_value = self.values.get(key)
        old_source = self.sources.get(key)
        
        # Convert OmegaConf objects to regular Python objects for comparison and storage
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value)
        
        self.values[key] = value
        self.sources[key] = source
        
        # Update nested structure
        self._update_nested(key, value, source)
        
        # Track the change
        if old_value != value:  # Only track actual changes
            self.history.append({
                'key': key,
                'old_value': old_value,
                'new_value': value,
                'old_source': old_source,
                'new_source': source,
                'timestamp': datetime.now()
            })
    
    def _update_nested(self, key: str, value: Any, source: str):
        """Update nested structure for tree visualization"""
        parts = key.split('.')
        current = self.nested
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {'_values': {}, '_source': source}
            current = current[part]
        
        # Set the final value
        if '_values' not in current:
            current['_values'] = {}
        current['_values'][parts[-1]] = (value, source)
    
    def set_nested_values(self, config: Dict, source: str, prefix: str = ''):
        """Recursively set values from nested dictionary"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and not key.startswith('_'):
                # Recurse into nested dictionaries
                self.set_nested_values(value, source, full_key)
            else:
                # Set the actual value
                self.set_value(full_key, value, source)
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.values.get(key, default)
    
    def print_summary(self):
        """Print configuration summary with color coding"""
        if RICH_AVAILABLE:
            # Create tree structure
            tree = Tree("📋 Configuration", guide_style="dim")
            self._build_tree(tree, self.nested)
            console.print(tree)
        else:
            # Fallback text output
            print("\n📋 Configuration")
            self._print_nested(self.nested)
    
    def _build_tree(self, tree: Tree, data: Dict, show_source: bool = True):
        """Build rich tree structure"""
        for key, value in sorted(data.items()):
            if key == '_values':
                for k, (v, source) in sorted(value.items()):
                    source_type = source.split(':')[0]
                    color = self.SOURCE_COLORS.get(source_type, 'white')
                    
                    if show_source:
                        label = f"{k} = {self._format_value(v)} [{color}]({source})[/{color}]"
                    else:
                        label = f"{k} = {self._format_value(v)}"
                    tree.add(label)
            elif key != '_source' and isinstance(value, dict):
                branch = tree.add(f"[bold]{key}[/bold]")
                self._build_tree(branch, value, show_source)
    
    def _print_nested(self, data: Dict, indent: int = 0):
        """Print nested structure without rich"""
        for key, value in sorted(data.items()):
            if key == '_values':
                for k, (v, source) in sorted(value.items()):
                    source_type = source.split(':')[0]
                    color = self.ANSI_COLORS.get(source_type, '')
                    spaces = '  ' * (indent + 1)
                    print(f"{spaces}├── {k} = {self._format_value(v)} {color}({source}){self.ANSI_RESET}")
            elif key != '_source' and isinstance(value, dict):
                spaces = '  ' * indent
                print(f"{spaces}├── {key}")
                self._print_nested(value, indent + 1)
    
    def _format_value(self, value: Any) -> str:
        """Format value for display"""
        if isinstance(value, (list, tuple)) and len(value) > 3:
            return f"[{value[0]}, {value[1]}, ..., {value[-1]}]"
        elif isinstance(value, dict) and len(value) > 3:
            keys = list(value.keys())
            return f"{{{keys[0]}: ..., {keys[-1]}: ...}}"
        elif isinstance(value, str) and len(value) > 50:
            return f"{value[:47]}..."
        else:
            return str(value)
    
    def print_history(self, last_n: Optional[int] = None):
        """Print configuration change history"""
        history_to_show = self.history[-last_n:] if last_n else self.history
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold]Configuration Override History ({len(history_to_show)} changes):[/bold]")
            for change in history_to_show:
                old_source_type = change['old_source'].split(':')[0] if change['old_source'] else 'unknown'
                new_source_type = change['new_source'].split(':')[0]
                
                old_color = self.SOURCE_COLORS.get(old_source_type, 'white')
                new_color = self.SOURCE_COLORS.get(new_source_type, 'white')
                
                old_val = str(change['old_value'])
                new_val = str(change['new_value'])
                
                if len(old_val) > 20:
                    old_val = old_val[:17] + "..."
                if len(new_val) > 20:
                    new_val = new_val[:17] + "..."
                
                console.print(
                    f"  {change['key']}: "
                    f"[{old_color}]{old_val}[/{old_color}] → "
                    f"[{new_color}]{new_val}[/{new_color}]"
                )
        else:
            print(f"\n=== Configuration Override History ({len(history_to_show)} changes) ===")
            for change in history_to_show:
                print(f"  {change['key']}: {change['old_value']} → {change['new_value']}")
    
    def get_overrides_from_base(self, base_source: str = 'yaml_config') -> Dict[str, Tuple[Any, str]]:
        """Get all values that override the base configuration"""
        overrides = {}
        for key, value in self.values.items():
            source = self.sources[key]
            if source != base_source:
                overrides[key] = (value, source)
        return overrides


class CARSTrainingManager:
    """CARS-FASTGAN Training Manager with complete configuration tracking"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.experiments_dir = self.project_root / "experiments"
        self.checkpoints_dir = self.experiments_dir / "checkpoints"
        self.configs_dir = self.project_root / "configs"
        self.config_tracker = ConfigTracker()
        
        # Load base configuration from YAML files
        self.base_config = self._load_complete_config()
        
        # Track ALL base configuration values
        console.print("\n[blue]📄 Loading and tracking base configuration from YAML files...[/blue]")
        self.config_tracker.set_nested_values(self.base_config, 'yaml_config')
        
        # Print banner
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold cyan]🚀 CARS-FASTGAN Training Manager[/bold cyan]\n"
                f"[dim]📁 Project root: {self.project_root}[/dim]\n"
                f"[dim]📊 Tracking {len(self.config_tracker.values)} configuration values[/dim]",
                border_style="cyan"
            ))
    
    def _load_complete_config(self) -> DictConfig:
        """Load complete configuration from YAML files using OmegaConf"""
        config_path = self.configs_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load with OmegaConf
        cfg = OmegaConf.load(config_path)
        
        # Resolve any interpolations
        OmegaConf.resolve(cfg)
        
        console.print(f"[green]✓ Loaded complete configuration from {config_path}[/green]")
        
        return cfg
    
    def _load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """Load optimization results if available"""
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
                "callbacks": {
                    "model_checkpoint": {
                        "monitor": "val/g_loss",
                        "filename": "fastgan-ep{epoch:04d}-g{val_g_loss:.3f}-d{val_d_loss:.3f}",
                        "save_top_k": 10,
                        "every_n_epochs": 50
                    },
                    "early_stopping": {
                        "monitor": "val/g_loss",
                        "patience": 500
                    }
                },
                "check_val_every_n_epoch": 1,
                "log_images_every_n_epochs": 1
            }
        }
    
    def get_model_config(self, model_size: str) -> Dict[str, Any]:
        """Get model configuration for given size"""
        # First try to load from training_presets.json
        presets_file = self.configs_dir / "training_presets.json"
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    model_configs = data.get('model_configs', {})
                    if model_size.lower() in model_configs:
                        return model_configs[model_size.lower()]
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load model configs: {e}[/yellow]")
        
        # Fallback to hardcoded configs
        configs = {
            'micro': {
                'generator': {'ngf': 32, 'n_layers': 3},
                'discriminator': {'ndf': 32, 'n_layers': 3}
            },
            'small': {
                'generator': {'ngf': 48, 'n_layers': 3},
                'discriminator': {'ndf': 48, 'n_layers': 3}
            },
            'standard': {
                'generator': {'ngf': 64, 'n_layers': 4},
                'discriminator': {'ndf': 64, 'n_layers': 4}
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
        _provided_args: Optional[set] = None
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
        
        # Apply optimal settings
        if 'batch_size' in optimal_settings:
            self.config_tracker.set_value('data.batch_size', optimal_settings['batch_size'], 'optimization')
        if 'precision' in optimal_settings:
            self.config_tracker.set_value('precision', optimal_settings['precision'], 'optimization')
    
    def _apply_preset(self, preset: str) -> Dict[str, Any]:
        """Apply preset configuration"""
        preset_config = self._get_preset_config(preset)
        
        # Apply preset values with proper tracking
        def apply_preset_values(config: Dict, prefix: str = ''):
            for key, value in config.items():
                if key in ['description', 'preset_name']:
                    continue
                    
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict) and key not in ['callbacks', 'gan_training', 'loss', 'optimizer', 'training']:
                    # Special handling for nested model configuration
                    if key == 'model':
                        # Handle model sub-configurations
                        if 'generator' in value:
                            for gkey, gvalue in value['generator'].items():
                                self.config_tracker.set_value(f'model.generator.{gkey}', gvalue, f'preset:{preset}')
                        if 'discriminator' in value:
                            for dkey, dvalue in value['discriminator'].items():
                                self.config_tracker.set_value(f'model.discriminator.{dkey}', dvalue, f'preset:{preset}')
                        if 'loss' in value:
                            for lkey, lvalue in value['loss'].items():
                                self.config_tracker.set_value(f'model.loss.{lkey}', lvalue, f'preset:{preset}')
                        if 'optimizer' in value:
                            for opt_type, opt_config in value['optimizer'].items():
                                for okey, ovalue in opt_config.items():
                                    self.config_tracker.set_value(f'model.optimizer.{opt_type}.{okey}', ovalue, f'preset:{preset}')
                        if 'training' in value:
                            for tkey, tvalue in value['training'].items():
                                self.config_tracker.set_value(f'model.training.{tkey}', tvalue, f'preset:{preset}')
                    elif key == 'data':
                        for dkey, dvalue in value.items():
                            self.config_tracker.set_value(f'data.{dkey}', dvalue, f'preset:{preset}')
                    else:
                        apply_preset_values(value, full_key)
                elif key == 'loss' and isinstance(value, dict):
                    # Top-level loss config
                    for lkey, lvalue in value.items():
                        self.config_tracker.set_value(f'model.loss.{lkey}', lvalue, f'preset:{preset}')
                elif key == 'optimizer' and isinstance(value, dict):
                    # Top-level optimizer config
                    for opt_type, opt_config in value.items():
                        for okey, ovalue in opt_config.items():
                            self.config_tracker.set_value(f'model.optimizer.{opt_type}.{okey}', ovalue, f'preset:{preset}')
                elif key == 'training' and isinstance(value, dict):
                    # Top-level training config
                    for tkey, tvalue in value.items():
                        self.config_tracker.set_value(f'model.training.{tkey}', tvalue, f'preset:{preset}')
                elif key == 'callbacks' and isinstance(value, dict):
                    # Callbacks configuration
                    self.config_tracker.set_value('callbacks', value, f'preset:{preset}')
                elif key == 'model_size':
                    # Apply model configuration based on size
                    model_config = self.get_model_config(value)
                    for mkey, mvalue in model_config.get('generator', {}).items():
                        self.config_tracker.set_value(f'model.generator.{mkey}', mvalue, f'preset:{preset}')
                    for mkey, mvalue in model_config.get('discriminator', {}).items():
                        self.config_tracker.set_value(f'model.discriminator.{mkey}', mvalue, f'preset:{preset}')
                else:
                    # Direct value
                    self.config_tracker.set_value(full_key, value, f'preset:{preset}')
        
        apply_preset_values(preset_config)
        preset_config['preset_name'] = preset
        return preset_config
    
    def _apply_command_line_overrides(self, **kwargs):
        """Apply command-line overrides - only for explicitly provided values"""
        # Map command line arguments to configuration keys
        arg_mappings = {
            'model_size': None,  # Special handling
            'batch_size': 'data.batch_size',
            'max_epochs': 'max_epochs',
            'check_val_every_n_epoch': 'check_val_every_n_epoch',
            'log_images_every_n_epochs': 'log_images_every_n_epochs',
            'num_workers': 'data.num_workers',
            'use_wandb': 'use_wandb',
            'device': 'device',
            'data_path': 'data_path',
            'experiment_name': 'experiment_name',
        }
        
        for arg, config_key in arg_mappings.items():
            if arg in kwargs:
                if arg == 'model_size':
                    # Apply model configuration
                    model_config = self.get_model_config(kwargs[arg])
                    for key, value in model_config.get('generator', {}).items():
                        self.config_tracker.set_value(f'model.generator.{key}', value, 'command_line')
                    for key, value in model_config.get('discriminator', {}).items():
                        self.config_tracker.set_value(f'model.discriminator.{key}', value, 'command_line')
                elif config_key:
                    self.config_tracker.set_value(config_key, kwargs[arg], 'command_line')
    
    def _write_complete_config_file(self, wandb_project: Optional[str], resume_checkpoint: Optional[str]) -> Path:
        """Write complete configuration to a YAML file"""
        # Create a complete configuration dictionary from tracked values
        complete_config = {}
        
        # Build nested structure from flat keys
        for key, value in self.config_tracker.values.items():
            parts = key.split('.')
            current = complete_config
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Convert values to appropriate types for YAML
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, (DictConfig, ListConfig)):
                value = OmegaConf.to_container(value)
            
            current[parts[-1]] = value
        
        # IMPORTANT: Ensure progress bar settings are preserved
        if 'enable_progress_bar' not in complete_config:
            complete_config['enable_progress_bar'] = True
        
        # Ensure callbacks.rich_progress_bar exists
        if 'callbacks' not in complete_config:
            complete_config['callbacks'] = {}
        if 'rich_progress_bar' not in complete_config['callbacks']:
            complete_config['callbacks']['rich_progress_bar'] = {'leave': True}
            
        # Add special values that might not be tracked
        if wandb_project and 'wandb' in complete_config:
            complete_config['wandb']['project'] = wandb_project
        
        if resume_checkpoint:
            complete_config['ckpt_path'] = str(resume_checkpoint)
        
        # Ensure experiment name is at top level
        experiment_name = self.config_tracker.get_value('experiment_name')
        complete_config['experiment_name'] = experiment_name
        
        # Write to file
        config_dir = self.experiments_dir / "run_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f"{experiment_name}_config.yaml"
        
        # Save with OmegaConf to handle special types properly
        OmegaConf.save(complete_config, config_file)
        
        console.print(f"[green]✓ Wrote complete configuration to: {config_file}[/green]")
        
        # Also save a human-readable version with sources
        annotated_file = config_dir / f"{experiment_name}_config_annotated.yaml"
        with open(annotated_file, 'w') as f:
            f.write("# CARS-FASTGAN Configuration with Sources\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Experiment: {experiment_name}\n\n")
            
            # Write config with source annotations
            self._write_annotated_config(f, complete_config, self.config_tracker.sources)
        
        return config_file
    
    def _write_annotated_config(self, f, config: Dict, sources: Dict[str, str], prefix: str = '', indent: int = 0):
        """Write config with source annotations"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            source = sources.get(full_key, 'unknown')
            
            spaces = '  ' * indent
            
            if isinstance(value, dict):
                f.write(f"{spaces}{key}:  # [{source}]\n")
                self._write_annotated_config(f, value, sources, full_key, indent + 1)
            else:
                f.write(f"{spaces}{key}: {value}  # [{source}]\n")
    
    def _build_command_with_overrides(self) -> List[str]:
        """Build command using Hydra overrides instead of config file"""
        cmd = ["python", "main.py"]
        
        # Get all overrides from base configuration
        overrides = self.config_tracker.get_overrides_from_base()
        
        # Add overrides as command-line arguments
        for key, (value, source) in overrides.items():
            # Skip complex nested structures that might cause issues
            if isinstance(value, (dict, list)) and key not in ['callbacks', 'wandb']:
                continue
                
            # Format the override
            if isinstance(value, bool):
                override = f"{key}={str(value).lower()}"
            elif isinstance(value, (int, float)):
                override = f"{key}={value}"
            elif isinstance(value, str):
                # Escape special characters if needed
                if ' ' in value or '=' in value:
                    override = f'{key}="{value}"'
                else:
                    override = f"{key}={value}"
            else:
                # Skip complex types
                continue
                
            cmd.append(override)
        
        return cmd
    
    def _build_command_with_config_file(self, config_file: Path) -> List[str]:
        """Build command using config file"""
        cmd = ["python", "main.py", "--config-file", str(config_file)]
        return cmd
    
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
                    'old_value': str(h['old_value'])[:100],  # Limit length
                    'new_value': str(h['new_value'])[:100],
                    'old_source': h['old_source'],
                    'new_source': h['new_source']
                }
                for h in self.config_tracker.history
            ],
            'statistics': {
                'total_values': len(self.config_tracker.values),
                'total_overrides': len(self.config_tracker.get_overrides_from_base()),
                'sources': {}
            }
        }
        
        # Count sources
        for source in self.config_tracker.sources.values():
            source_type = source.split(':')[0]
            config_data['statistics']['sources'][source_type] = \
                config_data['statistics']['sources'].get(source_type, 0) + 1
        
        config_file = config_dir / f"{experiment_name}_complete_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        console.print(f"[dim]💾 Saved complete configuration to {config_file}[/dim]")
    
    def _get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get predefined configuration preset"""
        presets = self.get_experiment_presets()
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]
    
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
            
            # Import the training function
            sys.path.insert(0, str(self.project_root))
            from main import train_with_config
            
            # Load the configuration
            cfg = OmegaConf.load(config_file)
            
            # Run training directly
            train_with_config(cfg)
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Training interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Training error: {e}[/red]")
            traceback.print_exc()
            return False
    
    def show_available_checkpoints(self):
        """Show available checkpoints"""
        console.print("\n[cyan]📦 Available Checkpoints:[/cyan]")
        
        checkpoints = []
        if self.checkpoints_dir.exists():
            for ckpt in self.checkpoints_dir.glob("**/*.ckpt"):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                checkpoints.append((ckpt, size_mb, mod_time))
        
        if checkpoints:
            for ckpt, size_mb, mod_time in sorted(checkpoints, key=lambda x: x[0].stat().st_mtime, reverse=True):
                console.print(f"  📁 {ckpt.name} ({size_mb:.1f} MB, {mod_time})")
        else:
            console.print("  [dim]No checkpoints found[/dim]")


def main():
    """Main entry point with improved argument parsing"""
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
    
    # Parse once to get namespace and track which args were explicitly provided
    args = parser.parse_args()
    
    # Track which arguments were explicitly provided on command line
    provided_args = set()
    
    # Check sys.argv to see what was actually provided
    import sys
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Extract the argument name
            arg_name = arg.split('=')[0][2:]  # Remove '--'
            # Map to internal names
            if arg_name in ['batch_size', 'max_epochs', 'check_val_every_n_epoch', 
                          'log_images_every_n_epochs', 'num_workers', 'model_size',
                          'device', 'use_wandb']:
                provided_args.add(arg_name.replace('-', '_'))
    
    # If preset is specified, show which presets are available
    if args.preset:
        console.print(f"\n[cyan]Using preset: {args.preset}[/cyan]")
        preset_desc = temp_manager.get_experiment_presets().get(args.preset, {}).get('description', '')
        if preset_desc:
            console.print(f"[dim]{preset_desc}[/dim]")
    
    # Create training manager
    manager = CARSTrainingManager()
    
    # Show checkpoints if requested
    if args.show_checkpoints:
        manager.show_available_checkpoints()
        return
    
    # Run optimization only
    if args.optimize_only:
        if HardwareOptimizer is None:
            console.print("[red]❌ Hardware optimizer not available. Please check installation.[/red]")
            return
            
        optimizer = HardwareOptimizer(args.device)
        recommendations = optimizer.benchmark_configurations()
        opt_dir = Path("scripts/optimization_results")
        optimizer.save_optimization_results(opt_dir)
        
        console.print("\n[green]✅ Optimization complete![/green]")
        console.print(f"[dim]📁 Results saved to: {opt_dir}[/dim]")
        return
    
    # Auto-optimize and launch
    if args.auto_optimize:
        # Implementation would go here
        console.print("[yellow]Auto-optimize feature not fully implemented yet[/yellow]")
        return
    
    # Launch multiple experiments
    if args.experiments:
        # Implementation would go here
        console.print("[yellow]Multiple experiments feature not fully implemented yet[/yellow]")
        return
    
    # Single training launch
    success = manager.launch_training(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        preset=args.preset,
        use_wandb=args.use_wandb if 'use_wandb' in provided_args else None,
        wandb_project=args.wandb_project,
        device=args.device,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        dry_run=args.dry_run,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_images_every_n_epochs=args.log_images_every_n_epochs,
        _provided_args=provided_args  # Pass the set of explicitly provided args
    )
    
    if success and not args.dry_run:
        console.print("\n[green]✨ Training completed successfully![/green]")
    elif not success and not args.dry_run:
        console.print("\n[red]❌ Training failed![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()