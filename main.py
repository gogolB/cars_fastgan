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
        name=cfg.experiment.name,
        version=cfg.experiment.version
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if cfg.use_wandb and cfg.wandb.project:
        try:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=f"{cfg.experiment.name}_{cfg.experiment.version or 'auto'}",
                tags=cfg.wandb.tags,
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
        filename=cfg.callbacks.model_checkpoint.filename,
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        auto_insert_metric_name=cfg.callbacks.model_checkpoint.auto_insert_metric_name,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.callbacks.early_stopping.patience > 0:
        early_stopping = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            mode=cfg.callbacks.early_stopping.mode,
            patience=cfg.callbacks.early_stopping.patience,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.callbacks.lr_monitor.logging_interval,
        log_momentum=cfg.callbacks.lr_monitor.log_momentum
    )
    callbacks.append(lr_monitor)
    
    # Rich progress bar (if available)
    try:
        from pytorch_lightning.callbacks import RichProgressBar
        rich_progress = RichProgressBar(
            leave=cfg.callbacks.rich_progress_bar.leave
        )
        callbacks.append(rich_progress)
    except ImportError:
        print("Rich progress bar not available, using default")
    
    # Model summary
    try:
        from pytorch_lightning.callbacks import RichModelSummary
        model_summary = RichModelSummary(
            max_depth=cfg.callbacks.model_summary.max_depth
        )
        callbacks.append(model_summary)
    except ImportError:
        from pytorch_lightning.callbacks import ModelSummary
        model_summary = ModelSummary(
            max_depth=cfg.callbacks.model_summary.max_depth
        )
        callbacks.append(model_summary)
    
    return callbacks


def create_trainer(cfg: DictConfig, logger, callbacks: list) -> pl.Trainer:
    """Create PyTorch Lightning trainer"""
    
    # Determine accelerator and devices
    if cfg.accelerator == "auto":
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
        accelerator = cfg.accelerator
        devices = cfg.devices
    
    print(f"Using accelerator: {accelerator}, devices: {devices}")
    
    # Trainer arguments
    trainer_args = {
        "accelerator": accelerator,
        "devices": devices,
        "max_epochs": cfg.max_epochs,
        "min_epochs": cfg.min_epochs,
        "precision": cfg.precision,
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": cfg.enable_checkpointing,
        "enable_model_summary": cfg.enable_model_summary,
        "enable_progress_bar": cfg.enable_progress_bar,
        "val_check_interval": cfg.val_check_interval,
        "check_val_every_n_epoch": cfg.check_val_every_n_epoch,
        "log_every_n_steps": cfg.log_every_n_steps,
        "gradient_clip_val": cfg.gradient_clip_val,
        "gradient_clip_algorithm": cfg.gradient_clip_algorithm,
        "accumulate_grad_batches": cfg.accumulate_grad_batches,
        "deterministic": cfg.deterministic,
        "benchmark": cfg.benchmark,
        "detect_anomaly": cfg.detect_anomaly,
    }
    
    # Add profiler if specified
    if cfg.profiler:
        trainer_args["profiler"] = cfg.profiler
    
    # Debug settings
    if cfg.debug.fast_dev_run:
        trainer_args["fast_dev_run"] = True
    if cfg.debug.limit_train_batches:
        trainer_args["limit_train_batches"] = cfg.debug.limit_train_batches
    if cfg.debug.limit_val_batches:
        trainer_args["limit_val_batches"] = cfg.debug.limit_val_batches
    if cfg.debug.overfit_batches > 0:
        trainer_args["overfit_batches"] = cfg.debug.overfit_batches
    
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
        num_workers=cfg.data.num_workers,
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
        gan_loss=cfg.model.loss.gan_loss,
        adversarial_weight=cfg.model.loss.adversarial_weight,
        feature_matching_weight=cfg.model.loss.feature_matching_weight,
        use_feature_matching=cfg.model.loss.use_feature_matching,
        feature_layers=cfg.model.loss.feature_layers,
        
        # Optimization
        generator_lr=cfg.model.optimizer.generator.lr,
        discriminator_lr=cfg.model.optimizer.discriminator.lr,
        beta1=cfg.model.optimizer.generator.betas[0],
        beta2=cfg.model.optimizer.generator.betas[1],
        weight_decay=cfg.model.optimizer.generator.weight_decay,
        
        # Training
        n_critic=cfg.model.training.n_critic,
        use_gradient_penalty=cfg.model.training.use_gradient_penalty,
        gradient_penalty_weight=cfg.model.training.gradient_penalty_weight,
        use_ema=cfg.model.training.use_ema,
        ema_decay=cfg.model.training.ema_decay,
        
        # Logging
        log_images_every_n_epochs=cfg.log_images_every_n_epochs,
        num_sample_images=cfg.num_sample_images,
        fixed_noise_size=cfg.gan_training.fixed_noise_size,
        
        # Evaluation
        compute_fid=cfg.evaluation.metrics.fid.enabled,
        compute_lpips=cfg.evaluation.metrics.lpips.enabled,
        fid_batch_size=cfg.evaluation.metrics.fid.batch_size,
        fid_num_samples=cfg.evaluation.metrics.fid.num_samples
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
        if cfg.evaluation.full_evaluation_at_end:
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