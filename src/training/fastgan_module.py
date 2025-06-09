"""
PyTorch Lightning Module for FASTGAN Training
Handles training loop, loss computation, and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import torchvision
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path
import copy

from ..models.fastgan import FastGAN, FastGANGenerator
from ..evaluation.metrics import FIDScore, LPIPSScore


class FastGANModule(pl.LightningModule):
    """PyTorch Lightning module for FASTGAN training"""
    
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
                print("Warning: FID computation requested but pytorch-fid not available")
                self.compute_fid = False
                
        if compute_lpips:
            try:
                self.lpips_metric = LPIPSScore()
            except ImportError:
                print("Warning: LPIPS computation requested but lpips not available")
                self.compute_lpips = False
        
        # Training tracking
        self.automatic_optimization = False  # Manual optimization for GANs
        
    def _create_ema_model(self) -> FastGANGenerator:
        """Create EMA version of generator with proper initialization"""
        # Get the generator's initialization parameters
        generator = self.model.generator
        
        # Create a new generator with the same parameters
        ema_generator = FastGANGenerator(
            latent_dim=generator.latent_dim,
            ngf=generator.ngf,
            n_layers=generator.n_layers,
            image_size=generator.image_size,
            channels=generator.channels,
            use_skip_connections=generator.use_skip_connections,
            norm_type=getattr(generator, 'norm_type', 'batch')  # Default to batch if not found
        )
        
        # Copy the state dict
        ema_generator.load_state_dict(generator.state_dict())
        ema_generator.eval()
        
        # Ensure it's on the same device
        if next(generator.parameters()).is_cuda:
            ema_generator = ema_generator.cuda()
        
        return ema_generator
    
    def _update_ema(self):
        """Update EMA generator"""
        if self.ema_generator is None or not self.use_ema:
            return
            
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_generator.parameters(), 
                self.model.generator.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def adversarial_loss(self, y_hat: torch.Tensor, y: bool) -> torch.Tensor:
        """Compute adversarial loss
        
        Args:
            y_hat: Discriminator predictions
            y: True for real, False for fake
        """
        if self.gan_loss == "hinge":
            if y:  # Real
                return F.relu(1.0 - y_hat).mean()
            else:  # Fake
                return F.relu(1.0 + y_hat).mean()
        elif self.gan_loss == "lsgan":
            target = torch.ones_like(y_hat) if y else torch.zeros_like(y_hat)
            return F.mse_loss(y_hat, target)
        elif self.gan_loss == "vanilla":
            target = torch.ones_like(y_hat) if y else torch.zeros_like(y_hat)
            return F.binary_cross_entropy_with_logits(y_hat, target)
        elif self.gan_loss == "wgan":
            return -y_hat.mean() if y else y_hat.mean()
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
        """Training step"""
        real_images = batch['image']
        batch_size = real_images.size(0)
        
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Sample noise
        noise = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        # Generate fake images
        fake_images = self.model.generator(noise)
        
        # Train discriminator
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
            self.log("train/gradient_penalty", gp, prog_bar=False)
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train generator every n_critic steps
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
                    real_features_selected = [real_features[i] for i in self.feature_layers if i < len(real_features)]
                    fake_features_selected = [fake_features_g[i] for i in self.feature_layers if i < len(fake_features_g)]
                else:
                    real_features_selected = real_features
                    fake_features_selected = fake_features_g
                
                if real_features_selected and fake_features_selected:
                    fm_loss = self.feature_matching_loss(fake_features_selected, real_features_selected)
                    g_loss += self.feature_matching_weight * fm_loss
                    self.log("train/feature_matching_loss", fm_loss, prog_bar=False)
            
            self.manual_backward(g_loss)
            opt_g.step()
            
            # Update EMA
            if self.use_ema and self.ema_generator is not None:
                self._update_ema()
            
            # Logging
            self.log("train/g_loss", g_loss, prog_bar=True)
            self.log("train/g_loss_adv", g_loss_adv, prog_bar=False)
        
        # Logging
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.log("train/d_loss_real", d_loss_real, prog_bar=False)
        self.log("train/d_loss_fake", d_loss_fake, prog_bar=False)
        
        # Log learning rates
        self.log("train/lr_g", opt_g.param_groups[0]['lr'], prog_bar=False)
        self.log("train/lr_d", opt_d.param_groups[0]['lr'], prog_bar=False)
    
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
        
        self.log("val/d_loss", val_d_loss, prog_bar=True)
        self.log("val/g_loss", val_g_loss, prog_bar=False)
        
        return {
            "val_d_loss": val_d_loss,
            "val_g_loss": val_g_loss,
            "real_images": real_images,
            "fake_images": fake_images
        }
    
    def on_validation_epoch_end(self):
        """Log images at end of validation epoch"""
        if self.current_epoch % self.hparams.log_images_every_n_epochs == 0:
            self._log_sample_images()
    
    def _log_sample_images(self):
        """Log sample images to tensorboard/wandb"""
        with torch.no_grad():
            # Generate samples with fixed noise
            num_samples = min(16, len(self.fixed_noise))
            if self.use_ema and self.ema_generator is not None:
                fake_images = self.ema_generator(self.fixed_noise[:num_samples])
            else:
                fake_images = self.model.generator(self.fixed_noise[:num_samples])
            
            # Create grid
            grid = torchvision.utils.make_grid(
                fake_images, 
                nrow=4, 
                normalize=True, 
                value_range=(-1, 1)
            )
            
            # Log to tensorboard
            if self.logger:
                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.add_image(
                        "generated_images", 
                        grid, 
                        self.current_epoch
                    )
    
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
        
        return [g_optimizer, d_optimizer]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state"""
        if self.use_ema and self.ema_generator is not None:
            checkpoint['ema_generator_state_dict'] = self.ema_generator.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state"""
        if self.use_ema and 'ema_generator_state_dict' in checkpoint:
            if self.ema_generator is None:
                self.ema_generator = self._create_ema_model()
            self.ema_generator.load_state_dict(checkpoint['ema_generator_state_dict'])