"""
Main training script for CARS-FASTGAN
Handles configuration loading, data setup, and training orchestration

This script provides:
- Hydra-based configuration management
- Automatic experiment tracking and logging
- Multi-backend support (CPU, CUDA, MPS)
- Comprehensive error handling and recovery
- Post-training evaluation options
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
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        has_images = any(list(data_path.glob(f'*{ext}')) for ext in image_extensions)
        
        if not has_images:
            raise ValueError(f"No images found in {data_path}")
        else:
            logger.info("Found images in root directory - will create splits automatically")
    
    # Validate model configuration
    if cfg.get('model', {}).get('generator', {}).get('n_layers', 4) < 1:
        raise ValueError("Generator must have at least 1 layer")
    
    if cfg.get('model', {}).get('discriminator', {}).get('n_layers', 3) < 1:
        raise ValueError("Discriminator must have at least 1 layer")
    
    # Validate training configuration
    if cfg.get('max_epochs', 1000) < 1:
        raise ValueError("max_epochs must be at least 1")
    
    if cfg.get('data', {}).get('batch_size', 8) < 1:
        raise ValueError("batch_size must be at least 1")


def setup_logging(cfg: DictConfig) -> List[pl.loggers.Logger]:
    """Setup experiment logging (TensorBoard, W&B, etc.)
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        List of configured loggers
    """
    loggers = []
    
    # Always use TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.get('experiment_name', 'fastgan_experiment'),
        version=cfg.get('experiment', {}).get('version', None),
        log_graph=False,  # Avoid graph logging issues
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    logger.info(f"TensorBoard logging to: {tb_logger.log_dir}")
    
    # Weights & Biases logger (optional)
    if cfg.get('use_wandb', False) and cfg.get('wandb', {}).get('project'):
        try:
            import wandb
            
            # Initialize wandb
            wandb_config = cfg.get('wandb', {})
            wandb_logger = WandbLogger(
                project=wandb_config.get('project', 'cars-fastgan'),
                entity=wandb_config.get('entity'),
                name=f"{cfg.get('experiment_name', 'fastgan')}_{cfg.get('experiment', {}).get('version', 'auto')}",
                tags=wandb_config.get('tags', ['fastgan', 'cars', 'microscopy']),
                save_dir=cfg.log_dir,
                log_model=wandb_config.get('log_model', True),
                offline=wandb_config.get('offline', False)
            )
            loggers.append(wandb_logger)
            logger.info("Weights & Biases logging enabled")
            
        except ImportError:
            logger.warning("W&B requested but not installed. Install with: pip install wandb")
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename=checkpoint_config.get('filename', 'fastgan-{epoch:04d}-{val_d_loss:.3f}'),
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
    try:
        rich_progress = RichProgressBar(
            leave=cfg.get('callbacks', {}).get('rich_progress_bar', {}).get('leave', True),
            theme=cfg.get('callbacks', {}).get('rich_progress_bar', {}).get('theme', None)
        )
        callbacks.append(rich_progress)
    except ImportError:
        logger.info("Rich progress bar not available, using default")
    
    # Model summary (Rich or default)
    try:
        model_summary = RichModelSummary(
            max_depth=cfg.get('callbacks', {}).get('model_summary', {}).get('max_depth', 2)
        )
        callbacks.append(model_summary)
    except ImportError:
        model_summary = ModelSummary(
            max_depth=cfg.get('callbacks', {}).get('model_summary', {}).get('max_depth', 2)
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
            logger.info("Using CPU (no GPU detected)")
    else:
        devices = cfg.get('devices', 1)
        logger.info(f"Using specified accelerator: {accelerator}")
    
    # Trainer configuration
    trainer_config = {
        "accelerator": accelerator,
        "devices": devices,
        "max_epochs": cfg.get('max_epochs', 1000),
        "min_epochs": cfg.get('min_epochs', 1),
        "precision": cfg.get('precision', '32-true'),
        "logger": logger_instance,
        "callbacks": callbacks,
        "enable_checkpointing": cfg.get('enable_checkpointing', True),
        "enable_model_summary": cfg.get('enable_model_summary', True),
        "enable_progress_bar": cfg.get('enable_progress_bar', True),
        "val_check_interval": cfg.get('val_check_interval', 1.0),
        "check_val_every_n_epoch": cfg.get('check_val_every_n_epoch', 1),
        "log_every_n_steps": cfg.get('log_every_n_steps', 50),
        "gradient_clip_val": cfg.get('gradient_clip_val', None),
        "gradient_clip_algorithm": cfg.get('gradient_clip_algorithm', 'norm'),
        "accumulate_grad_batches": cfg.get('accumulate_grad_batches', 1),
        "deterministic": cfg.get('deterministic', True),
        "benchmark": cfg.get('benchmark', False),
        "detect_anomaly": cfg.get('detect_anomaly', False),
        "num_sanity_val_steps": cfg.get('num_sanity_val_steps', 2),
        "reload_dataloaders_every_n_epochs": cfg.get('reload_dataloaders_every_n_epochs', 0),
    }
    
    # Add strategy for multi-GPU if needed
    if devices > 1:
        trainer_config["strategy"] = cfg.get('strategy', 'ddp')
    
    # Add profiler if specified
    profiler = cfg.get('profiler', None)
    if profiler:
        trainer_config["profiler"] = profiler
        logger.info(f"Profiling enabled: {profiler}")
    
    # Debug settings
    debug = cfg.get('debug', {})
    if isinstance(debug, dict):
        if debug.get('fast_dev_run', False):
            trainer_config["fast_dev_run"] = True
            logger.warning("Fast dev run enabled - only running a few batches")
        if debug.get('limit_train_batches'):
            trainer_config["limit_train_batches"] = debug['limit_train_batches']
        if debug.get('limit_val_batches'):
            trainer_config["limit_val_batches"] = debug['limit_val_batches']
        if debug.get('overfit_batches', 0) > 0:
            trainer_config["overfit_batches"] = debug['overfit_batches']
            logger.warning(f"Overfitting on {debug['overfit_batches']} batches")
    
    return pl.Trainer(**trainer_config)


def run_evaluation(
    model: FastGANModule, 
    datamodule: CARSDataModule, 
    cfg: DictConfig
) -> Dict[str, Any]:
    """Run comprehensive evaluation after training
    
    Args:
        model: Trained model
        datamodule: Data module
        cfg: Configuration
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("\n" + "="*50)
    logger.info("Running post-training evaluation...")
    logger.info("="*50)
    
    try:
        # Setup evaluation
        device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        evaluator = ComprehensiveEvaluator(device=device)
        
        # Get real images from validation set
        val_loader = datamodule.val_dataloader()
        real_images = []
        
        num_samples = cfg.get('evaluation', {}).get('metrics', {}).get('fid', {}).get('num_samples', 1000)
        batch_size = cfg.get('data', {}).get('batch_size', 8)
        
        for batch in val_loader:
            real_images.append(batch['image'])
            if len(real_images) * batch_size >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
        # Generate fake images
        model.eval()
        fake_images = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                noise = torch.randn(current_batch_size, model.hparams.latent_dim, device=model.device)
                
                if model.use_ema and model.ema_generator is not None:
                    fake_batch = model.ema_generator(noise)
                else:
                    fake_batch = model.model.generator(noise)
                    
                fake_images.append(fake_batch.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)
        
        # Run evaluation
        results = evaluator.evaluate(real_images, fake_images)
        
        # Save results
        results_path = Path(cfg.output_dir) / "evaluation_results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                serializable_results[key] = float(value)
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        OmegaConf.save(serializable_results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Optional metric value for hyperparameter optimization
    """
    # Setup logging
    setup_logging_config(cfg)
    
    logger.info("CARS-FASTGAN Training Script")
    logger.info("="*50)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Device: {cfg.device}")
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
    ckpt_path = cfg.get('resume_from_checkpoint', None)
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
        if cfg.get('evaluation', {}).get('schedule', {}).get('full_evaluation_at_end', False):
            results = run_evaluation(model, datamodule, cfg)
            
            # Print key results
            if results:
                if 'fid_score' in results and results['fid_score'] is not None:
                    logger.info(f"Final FID Score: {results['fid_score']:.3f}")
                if 'is_mean' in results and results['is_mean'] is not None:
                    logger.info(f"Final IS Score: {results['is_mean']:.3f} ± {results.get('is_std', 0):.3f}")
                
                # Return FID score for hyperparameter optimization
                return results.get('fid_score')
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("\nTraining script completed!")
    return None


if __name__ == "__main__":
    main()