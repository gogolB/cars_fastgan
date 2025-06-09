"""
Main training script for CARS-FASTGAN
Handles configuration loading, data setup, and training orchestration
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.dataset import CARSDataModule
from src.training.fastgan_module import FastGANModule
from src.evaluation.metrics import ComprehensiveEvaluator


def setup_logging(cfg: DictConfig) -> pl.loggers.Logger:
    """Setup logging based on configuration"""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.get('experiment_name', 'fastgan_experiment'),
        version=cfg.get('experiment', {}).get('version', None)
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if cfg.use_wandb and cfg.get('wandb', {}).get('project'):
        try:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.get('wandb', {}).get('entity'),
                name=f"{cfg.get('experiment_name', 'fastgan')}_{cfg.get('experiment', {}).get('version', 'auto')}",
                tags=cfg.get('wandb', {}).get('tags', []),
                save_dir=cfg.log_dir
            )
            loggers.append(wandb_logger)
            print("Weights & Biases logging enabled")
        except Exception as e:
            print(f"Warning: Could not initialize Weights & Biases: {e}")
    
    return loggers if len(loggers) > 1 else loggers[0]


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename=cfg.get('callbacks', {}).get('model_checkpoint', {}).get('filename', 'fastgan-{epoch:02d}'),
        monitor=cfg.get('callbacks', {}).get('model_checkpoint', {}).get('monitor', 'val/d_loss'),
        mode=cfg.get('callbacks', {}).get('model_checkpoint', {}).get('mode', 'min'),
        save_top_k=cfg.get('callbacks', {}).get('model_checkpoint', {}).get('save_top_k', 3),
        auto_insert_metric_name=cfg.get('callbacks', {}).get('model_checkpoint', {}).get('auto_insert_metric_name', False),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_patience = cfg.get('callbacks', {}).get('early_stopping', {}).get('patience', 0)
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=cfg.get('callbacks', {}).get('early_stopping', {}).get('monitor', 'val/d_loss'),
            mode=cfg.get('callbacks', {}).get('early_stopping', {}).get('mode', 'min'),
            patience=early_stopping_patience,
            min_delta=cfg.get('callbacks', {}).get('early_stopping', {}).get('min_delta', 0.01),
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.get('callbacks', {}).get('lr_monitor', {}).get('logging_interval', 'epoch'),
        log_momentum=cfg.get('callbacks', {}).get('lr_monitor', {}).get('log_momentum', False)
    )
    callbacks.append(lr_monitor)
    
    # Rich progress bar (if available)
    try:
        from pytorch_lightning.callbacks import RichProgressBar
        rich_progress = RichProgressBar(
            leave=cfg.get('callbacks', {}).get('rich_progress_bar', {}).get('leave', True)
        )
        callbacks.append(rich_progress)
    except ImportError:
        print("Rich progress bar not available, using default")
    
    # Model summary
    try:
        from pytorch_lightning.callbacks import RichModelSummary
        model_summary = RichModelSummary(
            max_depth=cfg.get('callbacks', {}).get('model_summary', {}).get('max_depth', 2)
        )
        callbacks.append(model_summary)
    except ImportError:
        from pytorch_lightning.callbacks import ModelSummary
        model_summary = ModelSummary(
            max_depth=cfg.get('callbacks', {}).get('model_summary', {}).get('max_depth', 2)
        )
        callbacks.append(model_summary)
    
    return callbacks


def create_trainer(cfg: DictConfig, logger, callbacks: list) -> pl.Trainer:
    """Create PyTorch Lightning trainer"""
    
    # Determine accelerator and devices
    accelerator = cfg.get('accelerator', 'auto')
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = 1
    else:
        devices = cfg.get('devices', 1)
    
    print(f"Using accelerator: {accelerator}, devices: {devices}")
    
    # Trainer arguments
    trainer_args = {
        "accelerator": accelerator,
        "devices": devices,
        "max_epochs": cfg.get('max_epochs', 1000),
        "min_epochs": cfg.get('min_epochs', 1),
        "precision": cfg.get('precision', '32-true'),
        "logger": logger,
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
    }
    
    # Add profiler if specified
    profiler = cfg.get('profiler', None)
    if profiler:
        trainer_args["profiler"] = profiler
    
    # Debug settings
    debug = cfg.get('debug', {})
    if isinstance(debug, dict):
        if debug.get('fast_dev_run', False):
            trainer_args["fast_dev_run"] = True
        if debug.get('limit_train_batches'):
            trainer_args["limit_train_batches"] = debug['limit_train_batches']
        if debug.get('limit_val_batches'):
            trainer_args["limit_val_batches"] = debug['limit_val_batches']
        if debug.get('overfit_batches', 0) > 0:
            trainer_args["overfit_batches"] = debug['overfit_batches']
    elif debug:  # If debug is True (boolean)
        trainer_args["fast_dev_run"] = True
    
    return pl.Trainer(**trainer_args)


def run_evaluation(
    model: FastGANModule, 
    datamodule: CARSDataModule, 
    cfg: DictConfig
) -> dict:
    """Run comprehensive evaluation after training"""
    print("\n" + "="*50)
    print("Running post-training evaluation...")
    print("="*50)
    
    # Setup evaluation
    evaluator = ComprehensiveEvaluator(device=cfg.device)
    
    # Get real images from validation set
    val_loader = datamodule.val_dataloader()
    real_images = []
    
    for batch in val_loader:
        real_images.append(batch['image'])
        if len(real_images) * cfg.data.batch_size >= cfg.evaluation.fid_num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)
    real_images = real_images[:cfg.evaluation.fid_num_samples]
    
    # Generate fake images
    model.eval()
    fake_images = []
    
    with torch.no_grad():
        for i in range(0, cfg.evaluation.fid_num_samples, cfg.evaluation.fid_batch_size):
            batch_size = min(cfg.evaluation.fid_batch_size, cfg.evaluation.fid_num_samples - i)
            noise = torch.randn(batch_size, model.hparams.latent_dim, device=model.device)
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
            serializable_results[key] = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                                       for k, v in value.items()}
        else:
            serializable_results[key] = value
    
    OmegaConf.save(serializable_results, results_path)
    print(f"Evaluation results saved to: {results_path}")
    
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    print("CARS-FASTGAN Training Script")
    print("="*50)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Device: {cfg.device}")
    print(f"Data path: {cfg.data_path}")
    print("="*50)
    
    # Set random seed
    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)
    
    # Create output directories
    for dir_path in [cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    print("Setting up data module...")
    datamodule = CARSDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        image_size=cfg.data.image_size,
        use_8bit=cfg.data.use_8bit,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.data.split_seed,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
        augment_train=cfg.data.augment_train,
        augment_val=cfg.data.augment_val,
        augment_test=cfg.data.augment_test
    )
    
    # Setup data
    datamodule.setup()
    print(f"Data setup complete:")
    print(f"  - Train samples: {len(datamodule.train_dataset)}")
    print(f"  - Val samples: {len(datamodule.val_dataset)}")
    print(f"  - Test samples: {len(datamodule.test_dataset)}")
    
    # Setup model
    print("Setting up model...")
    model = FastGANModule(
        # Model architecture
        latent_dim=cfg.model.generator.latent_dim,
        ngf=cfg.model.generator.ngf,
        ndf=cfg.model.discriminator.ndf,
        generator_layers=cfg.model.generator.n_layers,
        discriminator_layers=cfg.model.discriminator.n_layers,
        image_size=cfg.data.image_size,
        channels=cfg.data.channels,
        use_skip_connections=cfg.model.generator.use_skip_connections,
        use_spectral_norm=cfg.model.discriminator.use_spectral_norm,
        use_multiscale=cfg.model.discriminator.use_multiscale,
        num_scales=cfg.model.discriminator.num_scales,
        norm_type=cfg.model.generator.norm_type,
        
        # Loss configuration
        gan_loss=cfg.get('model', {}).get('loss', {}).get('gan_loss', 'hinge'),
        adversarial_weight=cfg.get('model', {}).get('loss', {}).get('adversarial_weight', 1.0),
        feature_matching_weight=cfg.get('model', {}).get('loss', {}).get('feature_matching_weight', 10.0),
        use_feature_matching=cfg.get('model', {}).get('loss', {}).get('use_feature_matching', True),
        feature_layers=cfg.get('model', {}).get('loss', {}).get('feature_layers', [2, 3, 4]),
        
        # Optimization
        generator_lr=cfg.get('model', {}).get('optimizer', {}).get('generator', {}).get('lr', 0.0002),
        discriminator_lr=cfg.get('model', {}).get('optimizer', {}).get('discriminator', {}).get('lr', 0.0002),
        beta1=cfg.get('model', {}).get('optimizer', {}).get('generator', {}).get('betas', [0.0, 0.999])[0],
        beta2=cfg.get('model', {}).get('optimizer', {}).get('generator', {}).get('betas', [0.0, 0.999])[1],
        weight_decay=cfg.get('model', {}).get('optimizer', {}).get('generator', {}).get('weight_decay', 0.0),
        
        # Training
        n_critic=cfg.get('model', {}).get('training', {}).get('n_critic', 1),
        use_gradient_penalty=cfg.get('model', {}).get('training', {}).get('use_gradient_penalty', False),
        gradient_penalty_weight=cfg.get('model', {}).get('training', {}).get('gradient_penalty_weight', 10.0),
        use_ema=cfg.get('model', {}).get('training', {}).get('use_ema', False),
        ema_decay=cfg.get('model', {}).get('training', {}).get('ema_decay', 0.999),
        
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
    
    # Setup logging and callbacks
    logger = setup_logging(cfg)
    callbacks = setup_callbacks(cfg)
    
    # Create trainer
    trainer = create_trainer(cfg, logger, callbacks)
    
    # Start training
    print("Starting training...")
    try:
        trainer.fit(model, datamodule)
        print("Training completed successfully!")
        
        # Run evaluation if requested
        if cfg.get('evaluation', {}).get('full_evaluation_at_end', False):
            results = run_evaluation(model, datamodule, cfg)
            
            # Print key results
            if 'fid_score' in results and results['fid_score'] is not None:
                print(f"Final FID Score: {results['fid_score']:.3f}")
            if 'is_mean' in results and results['is_mean'] is not None:
                print(f"Final IS Score: {results['is_mean']:.3f} ± {results['is_std']:.3f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    print("\nTraining script completed!")


if __name__ == "__main__":
    main()