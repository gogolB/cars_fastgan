"""
PyTorch Lightning Module for FASTGAN Training
Fixed version with proper TensorBoard logging and metric tracking

Key improvements:
- Proper step-based logging with global_step
- Consistent metric naming
- Better loss tracking
- Gradient norm monitoring
- Proper image logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, StepLR
import torchvision
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path
import copy
import logging

from ..models.fastgan import FastGAN, FastGANGenerator
from ..evaluation.metrics import FIDScore, LPIPSScore

logger = logging.getLogger(__name__)


class FastGANModule(pl.LightningModule):
    """PyTorch Lightning module for FASTGAN training with fixed logging"""
    
    def __init__(
        self,
        # Model parameters
        latent_dim: int = 256,
        ngf: int = 64,
        ndf: int = 64,
        generator_layers: int = 4,
        discriminator_layers: int = 3,
        image_size: int = 512,
        channels: int = 1,
        use_skip_connections: bool = True,
        use_spectral_norm: bool = True,
        use_multiscale: bool = True,
        num_scales: int = 2,
        norm_type: str = "batch",
        
        # Loss parameters
        gan_loss: str = "hinge",
        adversarial_weight: float = 1.0,
        feature_matching_weight: float = 10.0,
        use_feature_matching: bool = True,
        feature_layers: List[int] = [2, 3, 4],
        
        # Optimization parameters
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        beta1: float = 0.0,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        
        # Training parameters
        n_critic: int = 1,
        use_gradient_penalty: bool = False,
        gradient_penalty_weight: float = 10.0,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        
        # Logging parameters
        log_images_every_n_epochs: int = 25,
        num_sample_images: int = 16,
        fixed_noise_size: int = 64,
        log_every_n_steps: int = 50,
        
        # Evaluation parameters
        compute_fid: bool = True,
        compute_lpips: bool = True,
        fid_batch_size: int = 50,
        fid_num_samples: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = FastGAN(
            latent_dim=latent_dim,
            ngf=ngf,
            ndf=ndf,
            generator_layers=generator_layers,
            discriminator_layers=discriminator_layers,
            image_size=image_size,
            channels=channels,
            use_skip_connections=use_skip_connections,
            use_spectral_norm=use_spectral_norm,
            use_multiscale=use_multiscale,
            num_scales=num_scales,
            norm_type=norm_type
        )
        
        # Loss configuration
        self.gan_loss = gan_loss
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.use_feature_matching = use_feature_matching
        self.feature_layers = feature_layers
        
        # Training configuration
        self.n_critic = n_critic
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # EMA for generator
        self.use_ema = use_ema
        if use_ema:
            self.ema_generator = self._create_ema_model()
            self.ema_decay = ema_decay
        else:
            self.ema_generator = None
        
        # Fixed noise for consistent sampling
        self.register_buffer(
            "fixed_noise", 
            torch.randn(fixed_noise_size, latent_dim)
        )
        
        # Evaluation metrics
        self.compute_fid = compute_fid
        self.compute_lpips = compute_lpips
        
        if compute_fid:
            try:
                self.fid_metric = FIDScore()
            except ImportError:
                logger.warning("FID computation requested but pytorch-fid not available")
                self.compute_fid = False
                
        if compute_lpips:
            try:
                self.lpips_metric = LPIPSScore()
            except ImportError:
                logger.warning("LPIPS computation requested but lpips not available")
                self.compute_lpips = False
        
        # Training tracking
        self.automatic_optimization = False  # Manual optimization for GANs
        
        # Track training progress
        self.global_train_step = 0
        self.last_logged_epoch = 0
        
    def _create_ema_model(self) -> FastGANGenerator:
        """Create EMA version of generator"""
        ema_generator = FastGANGenerator(
            latent_dim=self.hparams.latent_dim,
            ngf=self.hparams.ngf,
            n_layers=self.hparams.generator_layers,
            image_size=self.hparams.image_size,
            channels=self.hparams.channels,
            use_skip_connections=self.hparams.use_skip_connections,
            norm_type=self.hparams.norm_type
        )
        ema_generator.load_state_dict(self.model.generator.state_dict())
        ema_generator.eval()
        
        for param in ema_generator.parameters():
            param.requires_grad = False
            
        return ema_generator
    
    def _update_ema(self):
        """Update EMA generator"""
        if self.ema_generator is None:
            return
            
        with torch.no_grad():
            for ema_param, param in zip(self.ema_generator.parameters(), 
                                       self.model.generator.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def adversarial_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute adversarial loss based on loss type"""
        if self.gan_loss == "hinge":
            if target_is_real:
                return F.relu(1.0 - pred).mean()
            else:
                return F.relu(1.0 + pred).mean()
        elif self.gan_loss == "bce":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return F.binary_cross_entropy_with_logits(pred, target)
        elif self.gan_loss == "wgan":
            return -pred.mean() if target_is_real else pred.mean()
        else:
            raise ValueError(f"Unknown GAN loss: {self.gan_loss}")
    
    def feature_matching_loss(
        self, 
        fake_features: List[torch.Tensor], 
        real_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature matching loss"""
        loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(fake_features)
    
    def gradient_penalty(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
        interpolated.requires_grad_(True)
        
        d_interpolated = self.model.discriminator(interpolated)
        if isinstance(d_interpolated, tuple):
            d_interpolated = d_interpolated[0]
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step with proper logging"""
        real_images = batch['image']
        batch_size = real_images.size(0)
        
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Update global step counter
        self.global_train_step += 1
        
        # Sample noise
        noise = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        # Generate fake images
        fake_images = self.model.generator(noise)
        
        # ============================================
        # Train Discriminator
        # ============================================
        opt_d.zero_grad()
        
        # Real images
        if self.model.discriminator.use_multiscale:
            real_pred, real_scale_preds, real_features = self.model.discriminator(
                real_images, return_features=True
            )
        else:
            real_output = self.model.discriminator(real_images, return_features=True)
            if isinstance(real_output, tuple) and len(real_output) == 2:
                real_pred, real_features = real_output
                real_scale_preds = []
            else:
                real_pred = real_output
                real_features = []
                real_scale_preds = []
        
        # Fake images (detached)
        if self.model.discriminator.use_multiscale:
            fake_pred, fake_scale_preds, fake_features = self.model.discriminator(
                fake_images.detach(), return_features=True
            )
        else:
            fake_output = self.model.discriminator(fake_images.detach(), return_features=True)
            if isinstance(fake_output, tuple) and len(fake_output) == 2:
                fake_pred, fake_features = fake_output
                fake_scale_preds = []
            else:
                fake_pred = fake_output
                fake_features = []
                fake_scale_preds = []
        
        # Discriminator loss
        d_loss_real = self.adversarial_loss(real_pred, True)
        d_loss_fake = self.adversarial_loss(fake_pred, False)
        d_loss = d_loss_real + d_loss_fake
        
        # Multi-scale discriminator loss
        for real_scale, fake_scale in zip(real_scale_preds, fake_scale_preds):
            d_loss += self.adversarial_loss(real_scale, True)
            d_loss += self.adversarial_loss(fake_scale, False)
        
        # Gradient penalty
        if self.use_gradient_penalty:
            gp = self.gradient_penalty(real_images, fake_images)
            d_loss += self.gradient_penalty_weight * gp
            self.log("train/gradient_penalty", gp, 
                    on_step=True, on_epoch=False, prog_bar=False, 
                    logger=True, sync_dist=True)
        
        self.manual_backward(d_loss)
        
        # Gradient clipping for discriminator
        if self.trainer.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), 
                self.trainer.gradient_clip_val
            )
        
        # Log discriminator gradient norm
        d_grad_norm = self._get_gradient_norm(self.model.discriminator)
        self.log("train/d_grad_norm", d_grad_norm,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
        opt_d.step()
        
        # ============================================
        # Train Generator
        # ============================================
        if batch_idx % self.n_critic == 0:
            opt_g.zero_grad()
            
            # Generate new fake images
            if self.model.discriminator.use_multiscale:
                fake_pred_g, fake_scale_preds_g, fake_features_g = self.model.discriminator(
                    fake_images, return_features=True
                )
            else:
                fake_output_g = self.model.discriminator(fake_images, return_features=True)
                if isinstance(fake_output_g, tuple) and len(fake_output_g) == 2:
                    fake_pred_g, fake_features_g = fake_output_g
                    fake_scale_preds_g = []
                else:
                    fake_pred_g = fake_output_g
                    fake_features_g = []
                    fake_scale_preds_g = []
            
            # Generator adversarial loss
            if self.gan_loss == "hinge" or self.gan_loss == "wgan":
                g_loss_adv = -fake_pred_g.mean()
            else:
                g_loss_adv = self.adversarial_loss(fake_pred_g, True)
            
            # Multi-scale generator loss
            for fake_scale in fake_scale_preds_g:
                if self.gan_loss == "hinge" or self.gan_loss == "wgan":
                    g_loss_adv += -fake_scale.mean()
                else:
                    g_loss_adv += self.adversarial_loss(fake_scale, True)
            
            g_loss = self.adversarial_weight * g_loss_adv
            
            # Feature matching loss
            if self.use_feature_matching and real_features and fake_features_g:
                # Select only the specified layers for feature matching
                if self.feature_layers:
                    real_features_selected = [real_features[i] for i in self.feature_layers 
                                            if i < len(real_features)]
                    fake_features_selected = [fake_features_g[i] for i in self.feature_layers 
                                            if i < len(fake_features_g)]
                else:
                    real_features_selected = real_features
                    fake_features_selected = fake_features_g
                
                if real_features_selected and fake_features_selected:
                    fm_loss = self.feature_matching_loss(fake_features_selected, real_features_selected)
                    g_loss += self.feature_matching_weight * fm_loss
                    self.log("train/feature_matching_loss", fm_loss,
                            on_step=True, on_epoch=False, prog_bar=False,
                            logger=True, sync_dist=True)
            
            self.manual_backward(g_loss)
            
            # Gradient clipping for generator
            if self.trainer.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.generator.parameters(), 
                    self.trainer.gradient_clip_val
                )
            
            # Log generator gradient norm
            g_grad_norm = self._get_gradient_norm(self.model.generator)
            self.log("train/g_grad_norm", g_grad_norm,
                    on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, sync_dist=True)
            
            opt_g.step()
            
            # Update EMA
            if self.use_ema and self.ema_generator is not None:
                self._update_ema()
            
            # Log generator losses
            self.log("train/g_loss", g_loss,
                    on_step=True, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=True)
            self.log("train/g_loss_adv", g_loss_adv,
                    on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, sync_dist=True)
        
        # ============================================
        # Logging
        # ============================================
        
        # Log discriminator losses
        self.log("train/d_loss", d_loss,
                on_step=True, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)
        self.log("train/d_loss_real", d_loss_real,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("train/d_loss_fake", d_loss_fake,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
        # Log learning rates
        self.log("train/lr_g", opt_g.param_groups[0]['lr'],
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("train/lr_d", opt_d.param_groups[0]['lr'],
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
        # Log discriminator predictions (for monitoring training balance)
        self.log("train/d_real_mean", real_pred.mean(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("train/d_fake_mean", fake_pred.mean(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
        # Log image statistics
        self.log("train/real_img_mean", real_images.mean(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("train/fake_img_mean", fake_images.mean(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
    def _get_gradient_norm(self, model: nn.Module) -> float:
        """Calculate gradient norm for a model"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        real_images = batch['image']
        batch_size = real_images.size(0)
        
        # Generate fake images
        noise = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        with torch.no_grad():
            if self.use_ema and self.ema_generator is not None:
                fake_images = self.ema_generator(noise)
            else:
                fake_images = self.model.generator(noise)
        
        # Compute discriminator predictions
        real_pred = self.model.discriminator(real_images)
        fake_pred = self.model.discriminator(fake_images)
        
        if isinstance(real_pred, tuple):
            real_pred = real_pred[0]
        if isinstance(fake_pred, tuple):
            fake_pred = fake_pred[0]
        
        # Validation losses
        val_d_loss_real = self.adversarial_loss(real_pred, True)
        val_d_loss_fake = self.adversarial_loss(fake_pred, False)
        val_d_loss = val_d_loss_real + val_d_loss_fake
        
        if self.gan_loss == "hinge" or self.gan_loss == "wgan":
            val_g_loss = -fake_pred.mean()
        else:
            val_g_loss = self.adversarial_loss(fake_pred, True)
        
        # Log validation losses
        self.log("val/d_loss", val_d_loss,
                on_step=False, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)
        self.log("val/g_loss", val_g_loss,
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("val/d_loss_real", val_d_loss_real,
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("val/d_loss_fake", val_d_loss_fake,
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
        
        return {
            "val_d_loss": val_d_loss,
            "val_g_loss": val_g_loss,
            "real_images": real_images,
            "fake_images": fake_images
        }
    
    def on_validation_epoch_end(self):
        """Log images at end of validation epoch"""
        if self.current_epoch - self.last_logged_epoch >= self.hparams.log_images_every_n_epochs:
            self._log_sample_images()
            self.last_logged_epoch = self.current_epoch
    
    

    def _log_sample_images(self):
        """Log sample images to tensorboard with diagnostics"""
        with torch.no_grad():
            # Generate samples with fixed noise
            num_samples = min(16, len(self.fixed_noise))
            
            # Generate from BOTH generators
            if self.use_ema and self.ema_generator is not None:
                fake_images_ema = self.ema_generator(self.fixed_noise[:num_samples])
                print(f"\n[DIAGNOSTIC] Epoch {self.current_epoch}")
                print(f"EMA Generator raw output - min: {fake_images_ema.min():.3f}, max: {fake_images_ema.max():.3f}, mean: {fake_images_ema.mean():.3f}")
            else:
                fake_images_ema = None
                
            # Always generate from regular generator too
            fake_images_regular = self.model.generator(self.fixed_noise[:num_samples])
            print(f"Regular Generator raw output - min: {fake_images_regular.min():.3f}, max: {fake_images_regular.max():.3f}, mean: {fake_images_regular.mean():.3f}")
            
            # Use EMA if available, otherwise regular
            fake_images = fake_images_ema if fake_images_ema is not None else fake_images_regular
            
            # Denormalize to [0, 1] for visualization
            fake_images_display = (fake_images + 1) / 2
            fake_images_display = torch.clamp(fake_images_display, 0, 1)
            
            print(f"After denormalization - min: {fake_images_display.min():.3f}, max: {fake_images_display.max():.3f}, mean: {fake_images_display.mean():.3f}")
            
            # Also generate with NEW random noise to see if it's the fixed noise causing issues
            new_noise = torch.randn_like(self.fixed_noise[:4])
            if self.use_ema and self.ema_generator is not None:
                fake_new = self.ema_generator(new_noise)
            else:
                fake_new = self.model.generator(new_noise)
            fake_new_display = torch.clamp((fake_new + 1) / 2, 0, 1)
            print(f"New random noise - min: {fake_new_display.min():.3f}, max: {fake_new_display.max():.3f}, mean: {fake_new_display.mean():.3f}")
            
            # Create comparison grid
            if fake_images_ema is not None:
                # Show both EMA and regular
                ema_display = torch.clamp((fake_images_ema + 1) / 2, 0, 1)
                regular_display = torch.clamp((fake_images_regular + 1) / 2, 0, 1)
                
                # Stack for comparison: top row = EMA, bottom row = regular
                comparison = torch.cat([
                    ema_display[:4],
                    regular_display[:4],
                    fake_new_display
                ], dim=0)
                
                comparison_grid = torchvision.utils.make_grid(comparison, nrow=4)
                
                if self.logger and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_image(
                        "comparison/ema_vs_regular_vs_new", 
                        comparison_grid, 
                        self.global_step
                    )
            
            # Create main grid
            grid = torchvision.utils.make_grid(
                fake_images_display, 
                nrow=4, 
                normalize=False,
                value_range=(0, 1)
            )
            
            # Log to tensorboard
            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image(
                    "generated_images", 
                    grid, 
                    self.global_step
                )
                
                # Also log individual samples for better inspection
                for i in range(min(4, num_samples)):
                    self.logger.experiment.add_image(
                        f"generated_samples/sample_{i}", 
                        fake_images_display[i], 
                        self.global_step
                    )
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log epoch number
        self.log("epoch", float(self.current_epoch),
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Generator optimizer
        g_optimizer = Adam(
            self.model.generator.parameters(),
            lr=self.hparams.generator_lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
        )
        
        # Discriminator optimizer
        d_optimizer = Adam(
            self.model.discriminator.parameters(),
            lr=self.hparams.discriminator_lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
        )
        
        optimizers = [g_optimizer, d_optimizer]
        
        # Optional: Add schedulers
        schedulers = []
        
        # Example: Linear warmup and decay
        if hasattr(self.hparams, 'use_scheduler') and self.hparams.use_scheduler:
            g_scheduler = {
                'scheduler': LinearLR(
                    g_optimizer,
                    start_factor=0.1,
                    total_iters=self.hparams.warmup_epochs
                ),
                'interval': 'epoch',
                'frequency': 1
            }
            d_scheduler = {
                'scheduler': LinearLR(
                    d_optimizer,
                    start_factor=0.1,
                    total_iters=self.hparams.warmup_epochs
                ),
                'interval': 'epoch',
                'frequency': 1
            }
            schedulers = [g_scheduler, d_scheduler]
        
        if schedulers:
            return optimizers, schedulers
        else:
            return optimizers
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state"""
        # Save EMA state
        if self.use_ema and self.ema_generator is not None:
            checkpoint['ema_generator_state_dict'] = self.ema_generator.state_dict()
        
        # Save global step
        checkpoint['global_train_step'] = self.global_train_step
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state"""
        # Load EMA state
        if self.use_ema and 'ema_generator_state_dict' in checkpoint:
            if self.ema_generator is None:
                self.ema_generator = self._create_ema_model()
            self.ema_generator.load_state_dict(checkpoint['ema_generator_state_dict'])
        
        # Load global step
        if 'global_train_step' in checkpoint:
            self.global_train_step = checkpoint['global_train_step']