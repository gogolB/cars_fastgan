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
from omegaconf import OmegaConf, DictConfig
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
        
        # Convert DictConfig to regular dict for comparison
        if isinstance(value, DictConfig):
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
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = {'value': value, 'source': source}
    
    def set_nested_values(self, data: Union[Dict, DictConfig], source: str, prefix: str = ''):
        """Recursively set all values from a nested dictionary"""
        if isinstance(data, DictConfig):
            data = OmegaConf.to_container(data)
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Set the dict itself
                    self.set_value(full_key, value, source)
                    # Recurse into it
                    self.set_nested_values(value, source, full_key)
                else:
                    self.set_value(full_key, value, source)
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.values.get(key, default)
    
    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a configuration value"""
        return self.sources.get(key)
    
    def get_all_with_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all values that start with a given prefix"""
        return {k: v for k, v in self.values.items() if k.startswith(prefix)}
    
    def print_summary(self, filter_source: Optional[str] = None):
        """Print configuration summary with color-coded sources"""
        if RICH_AVAILABLE:
            # Create a tree view
            tree = Tree("📋 Configuration", style="bold white")
            self._build_tree(tree, self.nested, filter_source)
            console.print(tree)
            
            # Print override chain legend
            console.print("\n[bold]Override Priority Chain (highest → lowest):[/bold]")
            priority_order = ['command_line', 'preset', 'optimization', 'yaml_config']
            for source in priority_order:
                color = self.SOURCE_COLORS[source]
                console.print(f"  [{color}]■[/{color}] {source.replace('_', ' ').title()}")
            
            # Print statistics
            source_counts = {}
            for source in self.sources.values():
                source_type = source.split(':')[0]
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            console.print(f"\n[bold]Configuration Statistics:[/bold]")
            console.print(f"Total values tracked: {len(self.values)}")
            for source, count in source_counts.items():
                color = self.SOURCE_COLORS.get(source, 'white')
                console.print(f"  [{color}]{source}[/{color}]: {count} values")
        else:
            # Fallback to table view
            self._print_table_view(filter_source)
    
    def _build_tree(self, parent: Tree, data: Dict, filter_source: Optional[str], level: int = 0):
        """Build tree visualization"""
        for key, value in sorted(data.items()):
            if isinstance(value, dict) and 'value' in value and 'source' in value:
                # Leaf node
                source = value['source']
                source_type = source.split(':')[0]
                
                if filter_source and source_type != filter_source:
                    continue
                
                color = self.SOURCE_COLORS.get(source_type, 'white')
                val_str = str(value['value'])
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                
                parent.add(f"[cyan]{key}[/cyan] = {val_str} [{color}]({source})[/{color}]")
            else:
                # Branch node
                branch = parent.add(f"[bold cyan]{key}[/bold cyan]")
                self._build_tree(branch, value, filter_source, level + 1)
    
    def _print_table_view(self, filter_source: Optional[str] = None):
        """Print table view for non-rich environments"""
        print("\n=== Configuration Summary ===")
        print(f"{'Key':<40} {'Value':<30} {'Source'}")
        print("-" * 90)
        
        for key in sorted(self.values.keys()):
            source = self.sources[key]
            source_type = source.split(':')[0]
            
            if filter_source and source_type != filter_source:
                continue
            
            value = str(self.values[key])
            if len(value) > 30:
                value = value[:27] + "..."
            
            color = self.ANSI_COLORS.get(source_type, '')
            print(f"{key:<40} {value:<30} {color}{source}{self.ANSI_RESET}")
    
    def print_history(self, last_n: Optional[int] = None):
        """Print the configuration change history"""
        history_to_show = self.history[-last_n:] if last_n else self.history
        
        if not history_to_show:
            console.print("[dim]No configuration changes recorded[/dim]")
            return
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold]Configuration Override History ({len(history_to_show)} changes):[/bold]")
            for change in history_to_show:
                old_source_type = change['old_source'].split(':')[0] if change['old_source'] else 'default'
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
        config_file = self.configs_dir / "config.yaml"
        
        if not config_file.exists():
            console.print(f"[yellow]Warning: config.yaml not found at {config_file}[/yellow]")
            return OmegaConf.create({})
        
        try:
            # Load with OmegaConf to handle includes and interpolations
            cfg = OmegaConf.load(config_file)
            
            # Load additional configs based on defaults
            if 'defaults' in cfg:
                for default_item in cfg.defaults:
                    if isinstance(default_item, str) and default_item != '_self_':
                        # Simple default reference
                        self._load_default_config(cfg, default_item)
                    elif isinstance(default_item, dict):
                        # Structured default with path
                        for category, file_name in default_item.items():
                            self._load_default_config(cfg, file_name, category)
            
            # Resolve all interpolations
            OmegaConf.resolve(cfg)
            
            console.print(f"[green]✓ Loaded complete configuration from {config_file}[/green]")
            return cfg
            
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            traceback.print_exc()
            return OmegaConf.create({})
    
    def _load_default_config(self, cfg: DictConfig, file_name: str, category: Optional[str] = None):
        """Load a default configuration file"""
        # Try different paths
        possible_paths = [
            self.configs_dir / f"{file_name}.yaml",
            self.configs_dir / category / f"{file_name}.yaml" if category else None,
        ]
        
        # Also check in subdirectories if category not specified
        if not category:
            for subdir in ['data', 'model', 'training', 'evaluation']:
                possible_paths.append(self.configs_dir / subdir / f"{file_name}.yaml")
        
        for path in possible_paths:
            if path and path.exists():
                try:
                    sub_cfg = OmegaConf.load(path)
                    
                    # Determine where to merge this config
                    if category:
                        cfg[category] = OmegaConf.merge(cfg.get(category, {}), sub_cfg)
                    else:
                        # Infer from filename
                        if 'data' in file_name:
                            cfg.data = OmegaConf.merge(cfg.get('data', {}), sub_cfg)
                        elif 'model' in file_name:
                            cfg.model = OmegaConf.merge(cfg.get('model', {}), sub_cfg)
                        elif 'training' in file_name:
                            cfg.training = OmegaConf.merge(cfg.get('training', {}), sub_cfg)
                        elif 'evaluation' in file_name:
                            cfg.evaluation = OmegaConf.merge(cfg.get('evaluation', {}), sub_cfg)
                    
                    console.print(f"[dim]  ✓ Loaded {path}[/dim]")
                    break
                except Exception as e:
                    console.print(f"[yellow]  Warning: Error loading {path}: {e}[/yellow]")
    
    def get_experiment_presets(self) -> Dict[str, Dict]:
        """Get predefined experiment configurations from configs/training_presets.json"""
        presets_file = self.configs_dir / "training_presets.json"
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    return data.get('presets', {})
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load presets: {e}[/yellow]")
        
        return {}
    
    def get_model_config(self, model_size: str) -> Dict[str, Any]:
        """Get model configuration by size from training_presets.json"""
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
        
        # Fallback
        return {
            'generator': {'ngf': 64, 'n_layers': 4},
            'discriminator': {'ndf': 64, 'n_layers': 3}
        }
    
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
        log_images_every_n_epochs: Optional[int] = None
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
        
        # Check each parameter against its default
        if model_size != 'standard':
            override_kwargs['model_size'] = model_size
        if batch_size is not None:
            override_kwargs['batch_size'] = batch_size
        if max_epochs is not None:
            override_kwargs['max_epochs'] = max_epochs
        if check_val_every_n_epoch is not None:
            override_kwargs['check_val_every_n_epoch'] = check_val_every_n_epoch
        if log_images_every_n_epochs is not None:
            override_kwargs['log_images_every_n_epochs'] = log_images_every_n_epochs
        if num_workers is not None:
            override_kwargs['num_workers'] = num_workers
        if use_wandb is not None:
            override_kwargs['use_wandb'] = use_wandb
        if device != 'auto':
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
        
        if 'batch_size' in optimal_settings:
            self.config_tracker.set_value('data.batch_size', optimal_settings['batch_size'], 'optimization')
    
    def _apply_preset(self, preset: str) -> Dict[str, Any]:
        """Apply preset configuration"""
        preset_config = self._get_preset_config(preset)
        
        # Apply all preset values recursively
        def apply_preset_values(config: Dict, prefix: str = ''):
            for key, value in config.items():
                if key in ['description', 'preset_name']:
                    continue
                
                full_key = f"{prefix}.{key}" if prefix else key
                
                # Map certain preset keys to config keys
                key_mappings = {
                    'model_size': None,  # Special handling below
                    'batch_size': 'data.batch_size',
                    'max_epochs': 'max_epochs',
                    'check_val_every_n_epoch': 'check_val_every_n_epoch',
                    'log_images_every_n_epochs': 'log_images_every_n_epochs',
                }
                
                if key in key_mappings:
                    if key == 'model_size':
                        # Apply model configuration
                        model_config = self.get_model_config(value)
                        for mkey, mvalue in model_config.get('generator', {}).items():
                            self.config_tracker.set_value(f'model.generator.{mkey}', mvalue, f'preset:{preset}')
                        for mkey, mvalue in model_config.get('discriminator', {}).items():
                            self.config_tracker.set_value(f'model.discriminator.{mkey}', mvalue, f'preset:{preset}')
                    elif key_mappings[key]:
                        self.config_tracker.set_value(key_mappings[key], value, f'preset:{preset}')
                elif isinstance(value, dict):
                    # Nested configuration
                    if key == 'loss':
                        for lkey, lvalue in value.items():
                            self.config_tracker.set_value(f'model.loss.{lkey}', lvalue, f'preset:{preset}')
                    elif key == 'optimizer':
                        for opt_type, opt_config in value.items():
                            for okey, ovalue in opt_config.items():
                                self.config_tracker.set_value(f'model.optimizer.{opt_type}.{okey}', ovalue, f'preset:{preset}')
                    elif key == 'training':
                        for tkey, tvalue in value.items():
                            self.config_tracker.set_value(f'model.training.{tkey}', tvalue, f'preset:{preset}')
                    else:
                        apply_preset_values(value, full_key)
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
            elif isinstance(value, DictConfig):
                value = OmegaConf.to_container(value)
            
            current[parts[-1]] = value
        
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
    
    def _build_command_with_config_file(self, config_file: Path) -> List[str]:
        """Build command using config file"""
        cmd = ["python", "main.py", "--config-file", str(config_file)]
        return cmd
    
    def _save_complete_configuration(self):
        """Save complete configuration with all tracking information"""
        config_dir = self.experiments_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_name = self.config_tracker.get_value('experiment_name')
        
        # Save complete tracked configuration
        config_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'values': self.config_tracker.values,
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
        
        return presets[preset].copy()
    
    def _load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """Load cached optimization results"""
        opt_file = self.project_root / "scripts/optimization_results/recommendations.json"
        
        if opt_file.exists():
            try:
                with open(opt_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    def _generate_experiment_name(self, base_name: Optional[str] = None) -> str:
        """Generate unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            return f"{base_name}_{timestamp}"
        return f"cars_fastgan_{timestamp}"
    
    def _execute_training(self, cmd: List[str]) -> bool:
        """Execute the training command"""
        console.print(f"\n[green]🚀 Launching training...[/green]")
        console.print("=" * 60)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            console.print("\n[green]✅ Training completed successfully![/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"\n[red]❌ Training failed with exit code {e.returncode}[/red]")
            return False
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Training interrupted by user[/yellow]")
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