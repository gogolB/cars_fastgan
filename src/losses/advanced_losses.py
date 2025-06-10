"""
Advanced Loss Functions for CARS-FASTGAN
Modular loss functions that integrate with existing FastGANModule

This module provides additional loss functions without breaking existing functionality.
All losses are optional and can be enabled through configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import logging
import warnings

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torchvision.models as models
    from torchvision.models import vgg19, VGG19_Weights
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    logger.warning("torchvision not available for perceptual loss")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.info("LPIPS not available. Install with: pip install lpips")


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for texture preservation"""
    
    def __init__(self, feature_layers: List[str] = None, device: str = 'cuda'):
        super().__init__()
        
        if not VGG_AVAILABLE:
            raise ImportError("torchvision required for perceptual loss")
        
        if feature_layers is None:
            # Good defaults for microscopy
            self.feature_layers = ['relu2_2', 'relu3_3']
        else:
            self.feature_layers = feature_layers
        
        # Load pretrained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Extract layers
        self.blocks = nn.ModuleList()
        layer_name_mapping = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15, 'relu4_3': 24
        }
        
        prev_idx = 0
        for layer_name in self.feature_layers:
            if layer_name in layer_name_mapping:
                idx = layer_name_mapping[layer_name] + 1
                block = nn.Sequential(*list(vgg.children())[prev_idx:idx])
                self.blocks.append(block)
                prev_idx = idx
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Handle grayscale
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if real.shape[1] == 1:
            real = real.repeat(1, 3, 1, 1)
        
        # Normalize
        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std
        
        loss = 0.0
        x_fake, x_real = fake, real
        
        for block in self.blocks:
            x_fake = block(x_fake)
            x_real = block(x_real)
            loss += F.l1_loss(x_fake, x_real.detach())
        
        return loss / len(self.blocks)


class SSIMLoss(nn.Module):
    """SSIM loss for structural preservation"""
    
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Create gaussian window
        def gaussian_window(window_size, sigma=1.5):
            coords = torch.arange(window_size).float()
            coords -= window_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.view(1, 1, window_size) * g.view(1, window_size, 1)
        
        window = gaussian_window(window_size).unsqueeze(0)
        self.register_buffer('window', window)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        
        if channel == self.channel:
            window = self.window
        else:
            window = self.window.repeat(channel, 1, 1, 1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()  # Return as loss


class TotalVariationLoss(nn.Module):
    """TV loss for background smoothness"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss wrapper"""
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        super().__init__()
        if not LPIPS_AVAILABLE:
            self.lpips = None
            warnings.warn("LPIPS not available. Install with: pip install lpips")
        else:
            self.lpips = lpips.LPIPS(net=net).to(device)
            for param in self.lpips.parameters():
                param.requires_grad = False
    
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            return torch.tensor(0.0, device=fake.device)
        
        # Handle grayscale
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if real.shape[1] == 1:
            real = real.repeat(1, 3, 1, 1)
        
        # LPIPS expects [-1, 1]
        if fake.min() >= 0:
            fake = fake * 2 - 1
        if real.min() >= 0:
            real = real * 2 - 1
        
        return self.lpips(fake, real).mean()


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss for better frequency matching"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # FFT
        fake_freq = torch.fft.fft2(fake, dim=(-2, -1))
        real_freq = torch.fft.fft2(real, dim=(-2, -1))
        
        # Magnitude spectrum
        fake_mag = torch.abs(fake_freq)
        real_mag = torch.abs(real_freq)
        
        # Focal weighting
        _, _, h, w = fake.shape
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(fake.device), x.to(fake.device)
        
        # Distance from center in frequency domain
        dist = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2)
        weight = 1 + self.alpha * dist / dist.max()
        
        # Weighted loss
        loss = (weight * (fake_mag - real_mag).abs()).mean()
        return loss


class ModeSeekinLoss(nn.Module):
    """Encourage diversity between different generated samples"""
    
    def forward(self, fake1: torch.Tensor, fake2: torch.Tensor,
                z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Image space distance
        img_dist = F.l1_loss(fake1, fake2, reduction='none').mean(dim=[1, 2, 3])
        
        # Latent space distance
        z_dist = F.l1_loss(z1, z2, reduction='none').mean(dim=1)
        
        # Maximize ratio (return negative for minimization)
        loss = -torch.mean(img_dist / (z_dist + 1e-8))
        return loss


class AdvancedLossManager:
    """Manager for advanced losses in FastGANModule
    
    This class handles initialization and computation of all advanced losses
    while maintaining backward compatibility with the existing module.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.losses = {}
        
        # Initialize enabled losses
        if config.get('use_perceptual_loss', False) and VGG_AVAILABLE:
            try:
                self.losses['perceptual'] = PerceptualLoss(
                    feature_layers=config.get('perceptual_layers'),
                    device=device
                ).to(device)
                logger.info("✓ Perceptual loss initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize perceptual loss: {e}")
        
        if config.get('use_lpips_loss', False):
            try:
                self.losses['lpips'] = LPIPSLoss(device=device).to(device)
                logger.info("✓ LPIPS loss initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS loss: {e}")
        
        if config.get('use_ssim_loss', False):
            self.losses['ssim'] = SSIMLoss().to(device)
            logger.info("✓ SSIM loss initialized")
        
        if config.get('use_tv_loss', False):
            self.losses['tv'] = TotalVariationLoss().to(device)
            logger.info("✓ Total Variation loss initialized")
        
        if config.get('use_focal_freq_loss', False):
            self.losses['focal_freq'] = FocalFrequencyLoss(
                alpha=config.get('focal_freq_alpha', 1.0)
            ).to(device)
            logger.info("✓ Focal Frequency loss initialized")
        
        if config.get('use_mode_seeking', False):
            self.losses['mode_seeking'] = ModeSeekinLoss().to(device)
            logger.info("✓ Mode Seeking loss initialized")
        
        # Store weights
        self.weights = {
            'perceptual': config.get('perceptual_weight', 10.0),
            'lpips': config.get('lpips_weight', 10.0),
            'ssim': config.get('ssim_weight', 1.0),
            'tv': config.get('tv_weight', 1e-4),
            'focal_freq': config.get('focal_freq_weight', 1.0),
            'mode_seeking': config.get('mode_seeking_weight', 0.1)
        }
    
    def compute_generator_losses(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        z1: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
        fake2: Optional[torch.Tensor] = None,
        global_step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute all enabled generator losses"""
        losses = {}
        
        # Perceptual losses
        if 'perceptual' in self.losses:
            losses['perceptual'] = self.losses['perceptual'](fake, real) * self.weights['perceptual']
        
        if 'lpips' in self.losses:
            losses['lpips'] = self.losses['lpips'](fake, real) * self.weights['lpips']
        
        # Structure and smoothness
        if 'ssim' in self.losses:
            losses['ssim'] = self.losses['ssim'](fake, real) * self.weights['ssim']
        
        if 'tv' in self.losses:
            losses['tv'] = self.losses['tv'](fake) * self.weights['tv']
        
        # Frequency
        if 'focal_freq' in self.losses:
            losses['focal_freq'] = self.losses['focal_freq'](fake, real) * self.weights['focal_freq']
        
        # Diversity (only every N steps to save computation)
        if 'mode_seeking' in self.losses and z1 is not None and z2 is not None and fake2 is not None:
            if global_step % self.config.get('mode_seeking_freq', 5) == 0:
                losses['mode_seeking'] = self.losses['mode_seeking'](
                    fake, fake2, z1, z2
                ) * self.weights['mode_seeking']
        
        return losses
    
    def to(self, device):
        """Move all losses to device"""
        for loss in self.losses.values():
            loss.to(device)
        return self