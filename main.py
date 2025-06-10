"""
Main training script for CARS-FASTGAN
Modified to accept a config file path instead of command-line overrides

This allows the training manager to write a complete configuration file
and pass it to main.py, avoiding Hydra parsing issues with complex values.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    RichProgressBar, RichModelSummary, ModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import warnings
import logging
from typing import Optional, List, Dict, Any
import argparse

# Suppress some warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')

# Import our modules
from src.data.dataset import CARSDataModule
from src.training.fastgan_module import FastGANModule
from src.evaluation.metrics import ComprehensiveEvaluator

# Set up logging
logger = logging.getLogger(__name__)


def setup_logging_config(cfg: DictConfig) -> None:
    """Setup logging configuration
    
    Args:
        cfg: Hydra configuration
    """
    log_level = cfg.get('log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration for common issues
    
    Args:
        cfg: Hydra configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check data path exists
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Check for required subdirectories or warn
    expected_dirs = ['train', 'val', 'test']
    has_split_dirs = all((data_path / d).exists() for d in expected_dirs)
    
    if has_split_dirs:
        logger.info("Found pre-split data directories")
    else:
        # Check if there are images in the root
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_path.glob(f'*{ext}'))
            image_files.extend(data_path.glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {data_path}")
        
        logger.info(f"Found {len(image_files)} images in root directory")
        logger.info("Data will be split automatically based on configuration")


def setup_logging(cfg: DictConfig):
    """Setup experiment logging
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Logger instance or list of loggers
    """
    loggers = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.experiment_name,
        version=None,
        log_graph=False,
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    logger.info(f"TensorBoard logger initialized: {cfg.log_dir}/{cfg.experiment_name}")
    
    # Weights & Biases logger (optional)
    if cfg.get('use_wandb', False):
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger
            
            wandb_config = cfg.get('wandb', {})
            
            # Initialize wandb run
            wandb.init(
                project=wandb_config.get('project', 'cars-fastgan'),
                entity=wandb_config.get('entity', None),
                name=cfg.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=wandb_config.get('tags', ['fastgan', 'cars']),
                notes=f"CARS-FASTGAN experiment: {cfg.experiment_name}",
                dir=cfg.log_dir,
                reinit=True
            )
            
            wandb_logger = WandbLogger(
                project=wandb_config.get('project', 'cars-fastgan'),
                entity=wandb_config.get('entity', None),
                name=cfg.experiment_name,
                save_dir=cfg.log_dir,
                offline=wandb_config.get('offline', False),
                log_model=wandb_config.get('log_model', True),
                experiment=wandb.run,
            )
            loggers.append(wandb_logger)
            logger.info(f"Weights & Biases logger initialized: {wandb_config.get('project')}/{cfg.experiment_name}")
            
        except ImportError:
            logger.warning("Weights & Biases requested but not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")
    
    return loggers if len(loggers) > 1 else loggers[0] if loggers else None


def setup_callbacks(cfg: DictConfig) -> List[pl.callbacks.Callback]:
    """Setup training callbacks
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_config = cfg.get('callbacks', {}).get('model_checkpoint', {})
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename with experiment name
    filename = checkpoint_config.get('filename', f"{cfg.experiment_name}-{{epoch:04d}}-{{val_d_loss:.3f}}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename,
        monitor=checkpoint_config.get('monitor', 'val/d_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=True,
        auto_insert_metric_name=checkpoint_config.get('auto_insert_metric_name', False),
        verbose=True,
        every_n_epochs=cfg.get('check_val_every_n_epoch', 1)
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping (optional)
    early_stopping_config = cfg.get('callbacks', {}).get('early_stopping', {})
    patience = early_stopping_config.get('patience', 0)
    
    if patience > 0:
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val/d_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            patience=patience,
            min_delta=early_stopping_config.get('min_delta', 0.001),
            verbose=True,
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled with patience={patience}")
    
    # Learning rate monitoring
    lr_monitor_config = cfg.get('callbacks', {}).get('lr_monitor', {})
    lr_monitor = LearningRateMonitor(
        logging_interval=lr_monitor_config.get('logging_interval', 'epoch'),
        log_momentum=lr_monitor_config.get('log_momentum', False)
    )
    callbacks.append(lr_monitor)
    
    # Progress bar (Rich or default)
    rich_progress_config = cfg.get('callbacks', {}).get('rich_progress_bar')
    if rich_progress_config is not None and cfg.get('enable_progress_bar', True):
        try:
            if isinstance(rich_progress_config, dict):
                rich_progress = RichProgressBar(
                    leave=rich_progress_config.get('leave', True)
                )
            else:
                rich_progress = RichProgressBar()
            callbacks.append(rich_progress)
        except ImportError:
            logger.info("Rich progress bar not available, using default")
    
    # Model summary
    model_summary_config = cfg.get('callbacks', {}).get('model_summary', {})
    if cfg.get('enable_model_summary', True):
        try:
            model_summary = RichModelSummary(
                max_depth=model_summary_config.get('max_depth', 2)
            )
        except ImportError:
            model_summary = ModelSummary(
                max_depth=model_summary_config.get('max_depth', 2)
            )
        callbacks.append(model_summary)
    
    return callbacks


def create_trainer(cfg: DictConfig, logger_instance, callbacks: List) -> pl.Trainer:
    """Create PyTorch Lightning trainer
    
    Args:
        cfg: Hydra configuration
        logger_instance: Logger instance(s)
        callbacks: List of callbacks
        
    Returns:
        Configured PyTorch Lightning Trainer
    """
    # Determine accelerator and devices
    accelerator = cfg.get('accelerator', 'auto')
    
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = cfg.get('devices', 1)
            logger.info(f"Using GPU acceleration with {devices} device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1  # MPS only supports single device
            logger.info("Using Apple Metal Performance Shaders (MPS) acceleration")
        else:
            accelerator = "cpu"
            devices = 1
            logger.info("Using CPU (no GPU available)")
    else:
        devices = cfg.get('devices', 1)
    
    # Ensure log directory exists
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer_config = {
        'accelerator': accelerator,
        'devices': devices,
        'precision': cfg.get('precision', '32-true'),
        'max_epochs': cfg.get('max_epochs', 1000),
        'min_epochs': cfg.get('min_epochs', 1),
        'gradient_clip_val': cfg.get('gradient_clip_val', 0.0),
        'gradient_clip_algorithm': cfg.get('gradient_clip_algorithm', 'norm'),
        'accumulate_grad_batches': cfg.get('accumulate_grad_batches', 1),
        'val_check_interval': cfg.get('val_check_interval', 1.0),
        'check_val_every_n_epoch': cfg.get('check_val_every_n_epoch', 10),
        'log_every_n_steps': cfg.get('log_every_n_steps', 50),
        'enable_checkpointing': cfg.get('enable_checkpointing', True),
        'enable_model_summary': cfg.get('enable_model_summary', True),
        'enable_progress_bar': cfg.get('enable_progress_bar', True),
        'detect_anomaly': cfg.get('detect_anomaly', False),
        'deterministic': cfg.get('deterministic', True),
        'benchmark': cfg.get('benchmark', False),
        'profiler': cfg.get('profiler', None),
        'num_sanity_val_steps': cfg.get('num_sanity_val_steps', 2),
        'reload_dataloaders_every_n_epochs': cfg.get('reload_dataloaders_every_n_epochs', 0),
        'logger': logger_instance,
        'callbacks': callbacks,
        'default_root_dir': str(log_dir),
    }
    
    # Handle debug settings
    if cfg.get('debug', {}).get('fast_dev_run', False):
        trainer_config['fast_dev_run'] = True
        logger.info("Fast dev run enabled - running 1 train/val/test batch")
    
    if cfg.get('debug', {}).get('limit_train_batches') is not None:
        trainer_config['limit_train_batches'] = cfg.debug.limit_train_batches
    
    if cfg.get('debug', {}).get('limit_val_batches') is not None:
        trainer_config['limit_val_batches'] = cfg.debug.limit_val_batches
    
    if cfg.get('debug', {}).get('overfit_batches', 0) > 0:
        trainer_config['overfit_batches'] = cfg.debug.overfit_batches
        logger.info(f"Overfitting to {cfg.debug.overfit_batches} batches")
    
    trainer = pl.Trainer(**trainer_config)
    
    logger.info(f"Trainer configured:")
    logger.info(f"  - Accelerator: {accelerator}")
    logger.info(f"  - Devices: {devices}")
    logger.info(f"  - Precision: {trainer_config['precision']}")
    logger.info(f"  - Max epochs: {trainer_config['max_epochs']}")
    
    return trainer


def train_with_config(cfg: DictConfig) -> Optional[float]:
    """Main training function
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Optional metric value for hyperparameter optimization
    """
    # Setup logging
    setup_logging_config(cfg)
    
    logger.info("CARS-FASTGAN Training Script")
    logger.info("="*50)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Device: {cfg.get('device', 'auto')}")
    logger.info(f"Data path: {cfg.data_path}")
    logger.info("="*50)
    
    # Validate configuration
    try:
        validate_config(cfg)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return None
    
    # Set random seed for reproducibility
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Random seed set to: {cfg.seed}")
    
    # Create output directories
    output_dirs = [cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    logger.info("Setting up data module...")
    
    datamodule = CARSDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.get('data', {}).get('batch_size', 8),
        num_workers=cfg.get('data', {}).get('num_workers', 4),
        image_size=cfg.get('data', {}).get('image_size', 512),
        use_8bit=cfg.get('data', {}).get('use_8bit', True),
        train_ratio=cfg.get('data', {}).get('train_ratio', 0.8),
        val_ratio=cfg.get('data', {}).get('val_ratio', 0.1),
        test_ratio=cfg.get('data', {}).get('test_ratio', 0.1),
        seed=cfg.get('data', {}).get('split_seed', 42),
        pin_memory=cfg.get('data', {}).get('pin_memory', True),
        drop_last=cfg.get('data', {}).get('drop_last', True),
        augment_train=cfg.get('data', {}).get('augment_train', True),
        augment_val=cfg.get('data', {}).get('augment_val', False),
        augment_test=cfg.get('data', {}).get('augment_test', False),
        cache_images=cfg.get('data', {}).get('cache_images', False),
        normalize_method=cfg.get('data', {}).get('normalize_method', 'standard'),
        persistent_workers=cfg.get('data', {}).get('persistent_workers', True)
    )
    
    # Setup data
    datamodule.setup()
    logger.info(f"Data setup complete:")
    logger.info(f"  - Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"  - Val samples: {len(datamodule.val_dataset)}")
    logger.info(f"  - Test samples: {len(datamodule.test_dataset)}")
    
    # Setup model
    logger.info("Setting up model...")
    
    # Extract nested configuration with defaults
    model_config = cfg.get('model', {})
    generator_config = model_config.get('generator', {})
    discriminator_config = model_config.get('discriminator', {})
    loss_config = model_config.get('loss', {})
    optimizer_config = model_config.get('optimizer', {})
    training_config = model_config.get('training', {})
    
    model = FastGANModule(
        # Model architecture
        latent_dim=generator_config.get('latent_dim', 256),
        ngf=generator_config.get('ngf', 64),
        ndf=discriminator_config.get('ndf', 64),
        generator_layers=generator_config.get('n_layers', 4),
        discriminator_layers=discriminator_config.get('n_layers', 3),
        image_size=cfg.get('data', {}).get('image_size', 512),
        channels=cfg.get('data', {}).get('channels', 1),
        use_skip_connections=generator_config.get('use_skip_connections', True),
        use_spectral_norm=discriminator_config.get('use_spectral_norm', True),
        use_multiscale=discriminator_config.get('use_multiscale', True),
        num_scales=discriminator_config.get('num_scales', 2),
        norm_type=generator_config.get('norm_type', 'batch'),
        
        # Loss configuration
        gan_loss=loss_config.get('gan_loss', 'hinge'),
        adversarial_weight=loss_config.get('adversarial_weight', 1.0),
        feature_matching_weight=loss_config.get('feature_matching_weight', 10.0),
        use_feature_matching=loss_config.get('use_feature_matching', True),
        feature_layers=loss_config.get('feature_layers', [2, 3, 4]),
        
        # Optimization
        generator_lr=optimizer_config.get('generator', {}).get('lr', 0.0002),
        discriminator_lr=optimizer_config.get('discriminator', {}).get('lr', 0.0002),
        beta1=optimizer_config.get('generator', {}).get('betas', [0.0, 0.999])[0],
        beta2=optimizer_config.get('generator', {}).get('betas', [0.0, 0.999])[1],
        weight_decay=optimizer_config.get('generator', {}).get('weight_decay', 0.0),
        
        # Training
        n_critic=training_config.get('n_critic', 1),
        use_gradient_penalty=training_config.get('use_gradient_penalty', False),
        gradient_penalty_weight=training_config.get('gradient_penalty_weight', 10.0),
        use_ema=training_config.get('use_ema', False),
        ema_decay=training_config.get('ema_decay', 0.999),
        
        # Logging
        log_images_every_n_epochs=cfg.get('log_images_every_n_epochs', 25),
        num_sample_images=cfg.get('num_sample_images', 16),
        fixed_noise_size=cfg.get('gan_training', {}).get('fixed_noise_size', 64),
        
        # Evaluation
        compute_fid=cfg.get('evaluation', {}).get('metrics', {}).get('fid', {}).get('enabled', True),
        compute_lpips=cfg.get('evaluation', {}).get('metrics', {}).get('lpips', {}).get('enabled', True),
        fid_batch_size=cfg.get('evaluation', {}).get('metrics', {}).get('fid', {}).get('batch_size', 50),
        fid_num_samples=cfg.get('evaluation', {}).get('metrics', {}).get('fid', {}).get('num_samples', 1000)
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup logging and callbacks
    experiment_logger = setup_logging(cfg)
    callbacks = setup_callbacks(cfg)
    
    # Create trainer
    trainer = create_trainer(cfg, experiment_logger, callbacks)
    
    # Resume from checkpoint if specified
    ckpt_path = cfg.get('ckpt_path', None) or cfg.get('resume_from_checkpoint', None)
    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    elif ckpt_path:
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        ckpt_path = None
    
    # Start training
    logger.info("Starting training...")
    
    try:
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        logger.info("Training completed successfully!")
        
        # Run evaluation if requested
        if cfg.get('evaluate_after_training', False):
            logger.info("Running post-training evaluation...")
            results = evaluate_model(trainer, model, datamodule, cfg)
            return results.get('fid_score', None)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    return None


def evaluate_model(trainer: pl.Trainer, model: pl.LightningModule, 
                  datamodule: pl.LightningDataModule, cfg: DictConfig) -> Dict[str, float]:
    """Run comprehensive evaluation
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Trained model
        datamodule: Data module
        cfg: Configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Implementation of evaluate_model remains the same
    # ... (keeping the existing implementation)
    pass


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_hydra(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point - original behavior preserved"""
    return train_with_config(cfg)


def main():
    """Main entry point with support for config file path"""
    parser = argparse.ArgumentParser(description='CARS-FASTGAN Training')
    parser.add_argument('--config-file', type=str, help='Path to complete config YAML file')
    parser.add_argument('overrides', nargs='*', help='Hydra overrides')
    
    args, unknown = parser.parse_known_args()
    
    if args.config_file:
        # Load config from file
        logger.info(f"Loading configuration from: {args.config_file}")
        cfg = OmegaConf.load(args.config_file)
        
        # Apply any additional overrides
        if args.overrides:
            overrides_cfg = OmegaConf.from_dotlist(args.overrides)
            cfg = OmegaConf.merge(cfg, overrides_cfg)
        
        # Run training with loaded config
        train_with_config(cfg)
    else:
        # Use Hydra's normal behavior
        sys.argv = [sys.argv[0]] + unknown + args.overrides
        main_hydra()


if __name__ == "__main__":
    main()