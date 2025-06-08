"""
FASTGAN Model Implementation
Optimized for training on small datasets (100-1000 images)

Based on: "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis"
https://arxiv.org/abs/2101.04775
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SkipConnection(nn.Module):
    """Skip connection module for FASTGAN generator"""
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply skip connection"""
        skip_processed = self.conv(skip)
        return x + self.scale_factor * skip_processed


class GeneratorBlock(nn.Module):
    """Generator block with skip connections"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_skip: bool = True,
        upsample: bool = True,
        norm_type: str = "batch"
    ):
        super().__init__()
        self.use_skip = use_skip
        self.upsample = upsample
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Normalization
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            self.norm1 = nn.InstanceNorm2d(out_channels)
        elif norm_type == "layer":
            self.norm1 = nn.LayerNorm([out_channels])
        else:
            self.norm1 = nn.Identity()
            
        self.activation = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        if norm_type == "batch":
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif norm_type == "layer":
            self.norm2 = nn.LayerNorm([out_channels])
        else:
            self.norm2 = nn.Identity()
        
        # Skip connection
        if self.use_skip:
            self.skip_connection = SkipConnection(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample if needed
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if skip is not None:
                skip = F.interpolate(skip, scale_factor=2, mode='nearest')
        
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Add skip connection
        if self.use_skip and skip is not None:
            out = self.skip_connection(out, skip)
            
        out = self.activation(out)
        
        return out


class FastGANGenerator(nn.Module):
    """FASTGAN Generator with skip connections"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        ngf: int = 64,
        n_layers: int = 4,
        image_size: int = 512,
        channels: int = 1,
        use_skip_connections: bool = True,
        norm_type: str = "batch"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.n_layers = n_layers
        self.image_size = image_size
        self.channels = channels
        self.use_skip_connections = use_skip_connections
        
        # Calculate initial size
        self.init_size = image_size // (2 ** n_layers)
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        in_channels = ngf * 8
        
        for i in range(n_layers):
            out_channels = max(ngf, in_channels // 2)
            
            block = GeneratorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_skip=use_skip_connections,
                upsample=True,
                norm_type=norm_type
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # Final output layer
        self.final_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.fc(z)
        x = x.view(batch_size, self.ngf * 8, self.init_size, self.init_size)
        
        # Store features for skip connections
        features = [x]
        
        # Generator blocks
        for i, block in enumerate(self.blocks):
            if self.use_skip_connections and len(features) > 1:
                # Use feature from corresponding depth for skip connection
                skip_idx = min(len(features) - 1, i)
                skip = features[skip_idx]
            else:
                skip = None
                
            x = block(x, skip)
            features.append(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator block"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        downsample: bool = True,
        use_spectral_norm: bool = True,
        norm_type: str = "batch"
    ):
        super().__init__()
        self.downsample = downsample
        
        # Convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(conv)
        else:
            self.conv = conv
        
        # Normalization
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            
        return x


class FastGANDiscriminator(nn.Module):
    """FASTGAN Discriminator with multi-scale architecture"""
    
    def __init__(
        self,
        ndf: int = 64,
        n_layers: int = 3,
        image_size: int = 512,
        channels: int = 1,
        use_spectral_norm: bool = True,
        norm_type: str = "batch",
        use_multiscale: bool = True,
        num_scales: int = 2
    ):
        super().__init__()
        self.ndf = ndf
        self.n_layers = n_layers
        self.use_multiscale = use_multiscale
        self.num_scales = num_scales
        
        # Build discriminator blocks
        self.blocks = nn.ModuleList()
        in_channels = channels
        
        for i in range(n_layers):
            out_channels = min(ndf * (2 ** i), ndf * 8)
            
            block = DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=True,
                use_spectral_norm=use_spectral_norm,
                norm_type=norm_type
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # Final layers
        final_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        if use_spectral_norm:
            self.final_conv = nn.utils.spectral_norm(final_conv)
        else:
            self.final_conv = final_conv
        
        # Multi-scale discriminators
        if use_multiscale:
            self.scale_discriminators = nn.ModuleList()
            for scale in range(1, num_scales):
                scale_disc = FastGANDiscriminator(
                    ndf=ndf // 2,
                    n_layers=max(2, n_layers - 1),
                    image_size=image_size // (2 ** scale),
                    channels=channels,
                    use_spectral_norm=use_spectral_norm,
                    norm_type=norm_type,
                    use_multiscale=False  # Avoid recursive multi-scale
                )
                self.scale_discriminators.append(scale_disc)
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass"""
        features = []
        original_input = x  # Store original input for multi-scale
        
        # Main discriminator
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        
        main_output = self.final_conv(x)
        
        # Multi-scale outputs
        if self.use_multiscale and hasattr(self, 'scale_discriminators'):
            scale_outputs = []
            
            for i, scale_disc in enumerate(self.scale_discriminators):
                # Downsample the ORIGINAL input for this scale
                scale_factor = 2 ** (i + 1)
                scale_input = F.avg_pool2d(original_input, scale_factor) if scale_factor > 1 else original_input
                scale_output = scale_disc(scale_input, return_features=False)
                scale_outputs.append(scale_output)
            
            if return_features:
                return main_output, scale_outputs, features
            else:
                return main_output, scale_outputs
        
        if return_features:
            return main_output, features
        else:
            return main_output


class FastGAN(nn.Module):
    """Complete FASTGAN model"""
    
    def __init__(
        self,
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
        norm_type: str = "batch"
    ):
        super().__init__()
        
        self.generator = FastGANGenerator(
            latent_dim=latent_dim,
            ngf=ngf,
            n_layers=generator_layers,
            image_size=image_size,
            channels=channels,
            use_skip_connections=use_skip_connections,
            norm_type=norm_type
        )
        
        self.discriminator = FastGANDiscriminator(
            ndf=ndf,
            n_layers=discriminator_layers,
            image_size=image_size,
            channels=channels,
            use_spectral_norm=use_spectral_norm,
            norm_type=norm_type,
            use_multiscale=use_multiscale,
            num_scales=num_scales
        )
        
        self.latent_dim = latent_dim
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random samples"""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor, return_features: bool = False):
        """Discriminate real/fake"""
        return self.discriminator(x, return_features=return_features)