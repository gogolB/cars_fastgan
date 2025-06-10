"""
PyTorch Lightning Module for FASTGAN Training
Fixed version with proper TensorBoard logging and metric tracking

Key improvements:
- Proper step-based logging with global_step
- Consistent metric naming
- Better loss tracking
- Gradient norm monitoring
- Proper image logging
- Advanced loss support
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

# Try to import advanced losses
try:
    from ..losses.advanced_losses import AdvancedLossManager
    ADVANCED_LOSSES_AVAILABLE = True
except ImportError:
    ADVANCED_LOSSES_AVAILABLE = False
    logging.info("Advanced losses not available")

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
        
        # Advanced loss parameters
        use_perceptual_loss: bool = False,
        perceptual_weight: float = 10.0,
        perceptual_layers: List[str] = None,
        use_lpips_loss: bool = False,
        lpips_weight: float = 10.0,
        use_ssim_loss: bool = False,
        ssim_weight: float = 1.0,
        use_tv_loss: bool = False,
        tv_weight: float = 1e-4,
        use_focal_freq_loss: bool = False,
        focal_freq_weight: float = 1.0,
        focal_freq_alpha: float = 1.0,
        use_mode_seeking: bool = False,
        mode_seeking_weight: float = 0.1,
        mode_seeking_freq: int = 5,
        
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
        
        # Initialize advanced losses if available and enabled
        if ADVANCED_LOSSES_AVAILABLE:
            advanced_loss_config = {
                'use_perceptual_loss': use_perceptual_loss,
                'perceptual_weight': perceptual_weight,
                'perceptual_layers': perceptual_layers,
                'use_lpips_loss': use_lpips_loss,
                'lpips_weight': lpips_weight,
                'use_ssim_loss': use_ssim_loss,
                'ssim_weight': ssim_weight,
                'use_tv_loss': use_tv_loss,
                'tv_weight': tv_weight,
                'use_focal_freq_loss': use_focal_freq_loss,
                'focal_freq_weight': focal_freq_weight,
                'focal_freq_alpha': focal_freq_alpha,
                'use_mode_seeking': use_mode_seeking,
                'mode_seeking_weight': mode_seeking_weight,
                'mode_seeking_freq': mode_seeking_freq,
                'device': getattr(self, 'device', 'cuda')
            }
            
            # Check if any advanced losses are enabled
            if any(k.startswith('use_') and v for k, v in advanced_loss_config.items()):
                self.advanced_loss_manager = AdvancedLossManager(
                    advanced_loss_config,
                    device=getattr(self, 'device', 'cuda')
                )
                logger.info("Advanced loss manager initialized")
            else:
                self.advanced_loss_manager = None
        else:
            self.advanced_loss_manager = None
        
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
        self.last_computed_metrics_epoch = 0
        
        # Validation outputs storage
        self.validation_step_outputs = []
        
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
    
    def on_train_start(self):
        """Move advanced losses to correct device if needed"""
        if self.advanced_loss_manager is not None:
            self.advanced_loss_manager.to(self.device)
    
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
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def _get_gradient_norm(self, model: nn.Module) -> float:
        """Get gradient norm for monitoring"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        """Training step with fixed logging and advanced losses"""
        real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_size = real_images.size(0)
        
        # Get optimizers
        opt_d, opt_g = self.optimizers()
        
        # ============================================
        # DISCRIMINATOR UPDATE
        # ============================================
        
        for _ in range(self.n_critic):
            opt_d.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_images = self.model.generator(z)
            
            # Discriminator predictions
            if self.hparams.use_multiscale:
                real_output = self.model.discriminator(real_images, return_features=True)
                fake_output = self.model.discriminator(fake_images.detach(), return_features=True)
                
                if len(real_output) == 3:
                    real_pred, real_scale_preds, real_features = real_output
                    fake_pred, fake_scale_preds, fake_features = fake_output
                elif len(real_output) == 2:
                    real_pred, real_features = real_output
                    fake_pred, fake_features = fake_output
                    real_scale_preds = []
                    fake_scale_preds = []
                else:
                    real_pred = real_output
                    fake_pred = fake_output
                    real_features = []
                    fake_features = []
                    real_scale_preds = []
                    fake_scale_preds = []
            else:
                real_output = self.model.discriminator(real_images, return_features=False)
                fake_output = self.model.discriminator(fake_images.detach(), return_features=False)
                
                if isinstance(real_output, tuple) and len(real_output) == 2:
                    real_pred, real_features = real_output
                    fake_pred, fake_features = fake_output
                else:
                    real_pred = real_output
                    fake_pred = fake_output
                    real_features = []
                    fake_features = []
                real_scale_preds = []
                fake_scale_preds = []
            
            # Discriminator loss
            d_loss_real = self.adversarial_loss(real_pred, True)
            d_loss_fake = self.adversarial_loss(fake_pred, False)
            
            # Multi-scale losses
            for real_scale, fake_scale in zip(real_scale_preds, fake_scale_preds):
                d_loss_real += self.adversarial_loss(real_scale, True)
                d_loss_fake += self.adversarial_loss(fake_scale, False)
            
            d_loss = d_loss_real + d_loss_fake
            
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
        # GENERATOR UPDATE
        # ============================================
        
        opt_g.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        fake_images = self.model.generator(z)
        
        # Prepare for mode seeking loss if enabled
        z2 = None
        fake_images2 = None
        if hasattr(self, 'advanced_loss_manager') and self.advanced_loss_manager is not None:
            if self.advanced_loss_manager.config.get('use_mode_seeking', False):
                z2 = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
                fake_images2 = self.model.generator(z2)
        
        # Discriminator predictions for generator
        if self.hparams.use_multiscale:
            fake_output_g = self.model.discriminator(fake_images, return_features=True)
            
            if len(fake_output_g) == 3:
                fake_pred_g, fake_scale_preds_g, fake_features_g = fake_output_g
            elif len(fake_output_g) == 2:
                fake_pred_g, fake_features_g = fake_output_g
                fake_scale_preds_g = []
            else:
                fake_pred_g = fake_output_g
                fake_features_g = []
                fake_scale_preds_g = []
        else:
            fake_output_g = self.model.discriminator(fake_images, return_features=False)
            if isinstance(fake_output_g, tuple) and len(fake_output_g) == 2:
                fake_pred_g, fake_features_g = fake_output_g
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
        
        # ============================================
        # ADVANCED LOSSES
        # ============================================
        
        if hasattr(self, 'advanced_loss_manager') and self.advanced_loss_manager is not None:
            advanced_losses = self.advanced_loss_manager.compute_generator_losses(
                fake_images, real_images,
                z1=z, z2=z2, fake2=fake_images2,
                global_step=self.global_train_step
            )
            
            # Add advanced losses to generator loss
            for loss_name, loss_value in advanced_losses.items():
                g_loss += loss_value
                self.log(f"train/g_loss_{loss_name}", loss_value,
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
        
        # Log generator losses
        self.log("train/g_loss", g_loss,
                on_step=True, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)
        self.log("train/g_loss_adv", g_loss_adv,
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
        self.log("train/real_img_std", real_images.std(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("train/fake_img_std", fake_images.std(),
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=True)
        
        # Update global step counter
        self.global_train_step += 1
    
    def on_validation_epoch_start(self):
        """Clear validation outputs at the start of validation epoch"""
        self.validation_step_outputs.clear()
    
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        """Validation step with fixed metrics"""
        real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_size = real_images.size(0)
        
        # Generate fake images
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        with torch.no_grad():
            fake_images = self.model.generator(z)
            
            # Get discriminator predictions
            if self.hparams.use_multiscale:
                real_output = self.model.discriminator(real_images, return_features=False)
                fake_output = self.model.discriminator(fake_images, return_features=False)
                
                # Handle multi-scale outputs
                if isinstance(real_output, tuple):
                    real_pred = real_output[0]
                    fake_pred = fake_output[0]
                else:
                    real_pred = real_output
                    fake_pred = fake_output
            else:
                real_pred = self.model.discriminator(real_images)
                fake_pred = self.model.discriminator(fake_images)
            
            # Compute validation losses
            d_loss_real = self.adversarial_loss(real_pred, True)
            d_loss_fake = self.adversarial_loss(fake_pred, False)
            d_loss = d_loss_real + d_loss_fake
            
            # Generator loss
            if self.gan_loss == "hinge" or self.gan_loss == "wgan":
                g_loss = -fake_pred.mean()
            else:
                g_loss = self.adversarial_loss(fake_pred, True)
        
        # Log validation metrics
        self.log("val/d_loss", d_loss, 
                on_step=False, on_epoch=True, prog_bar=False, 
                logger=True, sync_dist=True)
        self.log("val/g_loss", g_loss, 
                on_step=False, on_epoch=True, prog_bar=False, 
                logger=True, sync_dist=True)
        self.log("val/d_loss_real", d_loss_real,
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
        self.log("val/d_loss_fake", d_loss_fake,
                on_step=False, on_epoch=True, prog_bar=False,
                logger=True, sync_dist=True)
        
        # Store for epoch-level evaluation
        self.validation_step_outputs.append({
            'real_images': real_images.cpu(),
            'fake_images': fake_images.cpu()
        })
    
    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics and log images"""
        # Log sample images
        if self.current_epoch - self.last_logged_epoch >= self.hparams.log_images_every_n_epochs:
            self._log_sample_images()
            self.last_logged_epoch = self.current_epoch
        
        # Compute metrics if enabled and enough epochs have passed
        if self.current_epoch > 0 and self.current_epoch - self.last_computed_metrics_epoch >= 50:
            self.last_computed_metrics_epoch = self.current_epoch

            # Collect real and fake images for metrics
            try:
                real_images = torch.cat([out['real_images'] for out in self.validation_step_outputs[:10]], dim=0)
                fake_images = torch.cat([out['fake_images'] for out in self.validation_step_outputs[:10]], dim=0)
                
                # Move to device
                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)
                
                # Compute FID
                if self.compute_fid:
                    try:
                        fid_score = self.fid_metric.calculate_fid(
                            real_images, 
                            fake_images,
                            batch_size=min(50, len(real_images))
                        )
                        
                        self.log("val/fid_score", fid_score,
                                on_step=False, on_epoch=True, prog_bar=True,
                                logger=True, sync_dist=True)
                    except Exception as e:
                        logger.warning(f"Failed to compute FID: {e}")
                
                # Compute LPIPS
                if self.compute_lpips:
                    try:
                        lpips_score = self.lpips_metric.calculate_lpips(
                            real_images,
                            fake_images,
                            batch_size=min(32, len(real_images))
                        )
                        
                        self.log("val/lpips_score", lpips_score,
                                on_step=False, on_epoch=True, prog_bar=True,
                                logger=True, sync_dist=True)
                    except Exception as e:
                        logger.warning(f"Failed to compute LPIPS: {e}")
                        
            except Exception as e:
                logger.warning(f"Failed to collect images for metrics: {e}")
        
        # Clear validation outputs
        self.validation_step_outputs.clear()
    
    def _log_sample_images(self):
        """Generate and log sample images with fixed noise"""
        self.model.generator.eval()
        
        with torch.no_grad():
            # Use fixed noise for consistent samples
            noise = self.fixed_noise[:self.hparams.num_sample_images].to(self.device)
            
            # Generate from regular model
            fake_images = self.model.generator(noise)
            
            # Generate from EMA model if available
            if self.ema_generator is not None:
                ema_fake_images = self.ema_generator(noise)
            else:
                ema_fake_images = fake_images
            
            # Denormalize images (from [-1, 1] to [0, 1])
            fake_images = (fake_images + 1) / 2
            ema_fake_images = (ema_fake_images + 1) / 2
            
            # Create grid
            n_images = min(16, self.hparams.num_sample_images)
            fake_grid = torchvision.utils.make_grid(
                fake_images[:n_images], 
                nrow=4, 
                normalize=False,
                padding=2
            )
            
            ema_fake_grid = torchvision.utils.make_grid(
                ema_fake_images[:n_images], 
                nrow=4, 
                normalize=False,
                padding=2
            )
            
            # Log to tensorboard
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image(
                    'generated_images', 
                    fake_grid, 
                    global_step=self.global_train_step
                )
                
                if self.ema_generator is not None:
                    self.logger.experiment.add_image(
                        'ema_generated_images', 
                        ema_fake_grid, 
                        global_step=self.global_train_step
                    )
        
        self.model.generator.train()
    
    def configure_optimizers(self):
        """Configure optimizers with fixed parameters"""
        # Generator optimizer
        opt_g = Adam(
            self.model.generator.parameters(),
            lr=self.hparams.generator_lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
        )
        
        # Discriminator optimizer
        opt_d = Adam(
            self.model.discriminator.parameters(),
            lr=self.hparams.discriminator_lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
        )
        
        return [opt_d, opt_g]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state to checkpoint"""
        checkpoint['global_train_step'] = self.global_train_step
        
        # Save EMA state if available
        if self.ema_generator is not None:
            checkpoint['ema_generator_state_dict'] = self.ema_generator.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state from checkpoint"""
        self.global_train_step = checkpoint.get('global_train_step', 0)
        
        # Load EMA state if available
        if self.ema_generator is not None and 'ema_generator_state_dict' in checkpoint:
            self.ema_generator.load_state_dict(checkpoint['ema_generator_state_dict'])