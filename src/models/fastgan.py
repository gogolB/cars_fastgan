"""
FASTGAN Model Implementation
Optimized for training on small datasets (100-1000 images)

Based on: "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis"
https://arxiv.org/abs/2101.04775

Key Features:
- Skip connections in generator for better gradient flow
- Multi-scale discrimination for capturing both global and local features
- Spectral normalization for training stability
- Flexible architecture supporting different model sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
import numpy as np


def weights_init(m):
    """Initialize network weights using normal distribution
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class SkipConnection(nn.Module):
    """Skip connection module for FASTGAN generator
    
    Implements residual connections with learnable scaling factor
    to improve gradient flow and training stability.
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 0.1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            scale_factor: Scaling factor for skip connection (default: 0.1)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply skip connection
        
        Args:
            x: Main path tensor
            skip: Skip connection tensor
            
        Returns:
            Combined tensor with skip connection
        """
        skip_processed = self.conv(skip)
        return x + self.scale_factor * skip_processed


class GeneratorBlock(nn.Module):
    """Generator block with skip connections and upsampling
    
    This block implements:
    - Upsampling (optional)
    - Two convolutional layers with normalization
    - Skip connections for better gradient flow
    - Configurable normalization (batch, instance, layer, or none)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_skip: bool = True,
        upsample: bool = True,
        norm_type: str = "batch",
        activation: str = "relu"
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_skip: Whether to use skip connections
            upsample: Whether to upsample the input
            norm_type: Type of normalization ('batch', 'instance', 'layer', 'none')
            activation: Activation function ('relu', 'leaky_relu')
        """
        super().__init__()
        self.use_skip = use_skip
        self.upsample = upsample
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Normalization layers
        self.norm1 = self._get_norm_layer(norm_type, out_channels)
        self.norm2 = self._get_norm_layer(norm_type, out_channels)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Skip connection
        if self.use_skip:
            self.skip_connection = SkipConnection(in_channels, out_channels)
    
    def _get_norm_layer(self, norm_type: str, num_features: int) -> nn.Module:
        """Get normalization layer based on type
        
        Args:
            norm_type: Type of normalization
            num_features: Number of features to normalize
            
        Returns:
            Normalization layer
        """
        if norm_type == "batch":
            return nn.BatchNorm2d(num_features)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(num_features)
        elif norm_type == "layer":
            return nn.GroupNorm(1, num_features)  # LayerNorm equivalent for CNNs
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the generator block
        
        Args:
            x: Input tensor
            skip: Optional skip connection input
            
        Returns:
            Output tensor
        """
        # Upsample if needed
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if skip is not None and self.use_skip:
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
    """FASTGAN Generator with skip connections
    
    The generator uses:
    - Progressive upsampling from latent code to full resolution
    - Skip connections between layers for better gradient flow
    - Configurable architecture (number of layers, filters, etc.)
    - Flexible normalization options
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        ngf: int = 64,
        n_layers: int = 4,
        image_size: int = 512,
        channels: int = 1,
        use_skip_connections: bool = True,
        skip_connection_scale: float = 0.1,
        norm_type: str = "batch",
        activation: str = "relu",
        output_activation: str = "tanh",
        init_type: str = "normal",
        init_gain: float = 0.02
    ):
        """
        Args:
            latent_dim: Dimension of latent vector
            ngf: Number of generator filters in first layer
            n_layers: Number of upsampling layers
            image_size: Target image size (must be power of 2)
            channels: Number of output channels
            use_skip_connections: Whether to use skip connections
            skip_connection_scale: Scale factor for skip connections
            norm_type: Type of normalization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            init_type: Weight initialization type
            init_gain: Gain for weight initialization
        """
        super().__init__()
        
        # Store configuration
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.n_layers = n_layers
        self.image_size = image_size
        self.channels = channels
        self.use_skip_connections = use_skip_connections
        self.norm_type = norm_type
        
        # Validate image size
        assert image_size & (image_size - 1) == 0, "Image size must be a power of 2"
        assert image_size >= 2 ** n_layers, "Image size must be >= 2^n_layers"
        
        # Calculate initial size
        self.init_size = image_size // (2 ** n_layers)
        
        # Initial projection from latent code
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        in_channels = ngf * 8
        
        for i in range(n_layers):
            # Calculate output channels (decrease as we go up)
            out_channels = max(ngf, in_channels // 2)
            
            block = GeneratorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_skip=use_skip_connections,
                upsample=True,
                norm_type=norm_type,
                activation=activation
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # Final output layer
        self.final_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Output activation
        if output_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif output_activation == "none":
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, image_size, image_size)
        """
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.fc(z)
        x = x.view(batch_size, self.ngf * 8, self.init_size, self.init_size)
        
        # Store features for skip connections
        features = [x] if self.use_skip_connections else []
        
        # Generator blocks with skip connections
        for i, block in enumerate(self.blocks):
            if self.use_skip_connections and len(features) > 1:
                # Use feature from previous layer as skip connection
                skip = features[-1]
            else:
                skip = None
                
            x = block(x, skip)
            
            if self.use_skip_connections:
                features.append(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator block with downsampling
    
    This block implements:
    - Convolution with optional spectral normalization
    - Configurable normalization
    - Downsampling via average pooling
    - LeakyReLU activation
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        downsample: bool = True,
        use_spectral_norm: bool = True,
        norm_type: str = "batch",
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            downsample: Whether to downsample the input
            use_spectral_norm: Whether to use spectral normalization
            norm_type: Type of normalization
            activation: Activation function
            leaky_slope: Slope for LeakyReLU
        """
        super().__init__()
        self.downsample = downsample
        
        # Convolution with optional spectral norm
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
        elif norm_type == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        # Activation
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator block
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            
        return x


class FastGANDiscriminator(nn.Module):
    """FASTGAN Discriminator with multi-scale architecture
    
    The discriminator uses:
    - Progressive downsampling to capture features at multiple scales
    - Optional multi-scale discrimination
    - Spectral normalization for training stability
    - Feature extraction for feature matching loss
    """
    
    def __init__(
        self,
        ndf: int = 64,
        n_layers: int = 3,
        image_size: int = 512,
        channels: int = 1,
        use_spectral_norm: bool = True,
        norm_type: str = "batch",
        use_multiscale: bool = True,
        num_scales: int = 2,
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2
    ):
        """
        Args:
            ndf: Number of discriminator filters in first layer
            n_layers: Number of downsampling layers
            image_size: Input image size
            channels: Number of input channels
            use_spectral_norm: Whether to use spectral normalization
            norm_type: Type of normalization
            use_multiscale: Whether to use multi-scale discrimination
            num_scales: Number of scales for multi-scale discrimination
            activation: Activation function
            leaky_slope: Slope for LeakyReLU
        """
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
                norm_type=norm_type,
                activation=activation,
                leaky_slope=leaky_slope
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
        if use_multiscale and num_scales > 1:
            self.scale_discriminators = nn.ModuleList()
            for scale in range(1, num_scales):
                scale_disc = FastGANDiscriminator(
                    ndf=ndf // 2,  # Smaller capacity for lower scales
                    n_layers=max(2, n_layers - 1),
                    image_size=image_size // (2 ** scale),
                    channels=channels,
                    use_spectral_norm=use_spectral_norm,
                    norm_type=norm_type,
                    use_multiscale=False,  # Avoid recursive multi-scale
                    activation=activation,
                    leaky_slope=leaky_slope
                )
                self.scale_discriminators.append(scale_disc)
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], 
               Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]]:
        """Forward pass through discriminator
        
        Args:
            x: Input images
            return_features: Whether to return intermediate features
            
        Returns:
            If return_features is False:
                - For single-scale: discriminator predictions
                - For multi-scale: (main_predictions, scale_predictions)
            If return_features is True:
                - For single-scale: (predictions, features)
                - For multi-scale: (main_predictions, scale_predictions, features)
        """
        features = []
        original_input = x
        
        # Main discriminator path
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
                if scale_factor > 1:
                    scale_input = F.avg_pool2d(original_input, scale_factor)
                else:
                    scale_input = original_input
                    
                scale_output = scale_disc(scale_input, return_features=False)
                scale_outputs.append(scale_output)
            
            if return_features:
                return main_output, scale_outputs, features
            else:
                return main_output, scale_outputs
        
        # Single-scale output
        if return_features:
            return main_output, features
        else:
            return main_output


class FastGAN(nn.Module):
    """Complete FASTGAN model
    
    This class combines the generator and discriminator into a complete GAN model.
    It provides convenient methods for generating samples and computing discriminator outputs.
    """
    
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
        """
        Args:
            latent_dim: Dimension of latent vector
            ngf: Base number of generator filters
            ndf: Base number of discriminator filters
            generator_layers: Number of generator upsampling layers
            discriminator_layers: Number of discriminator downsampling layers
            image_size: Target image size
            channels: Number of image channels
            use_skip_connections: Whether to use skip connections in generator
            use_spectral_norm: Whether to use spectral norm in discriminator
            use_multiscale: Whether to use multi-scale discrimination
            num_scales: Number of scales for multi-scale discrimination
            norm_type: Type of normalization to use
        """
        super().__init__()
        
        # Create generator
        self.generator = FastGANGenerator(
            latent_dim=latent_dim,
            ngf=ngf,
            n_layers=generator_layers,
            image_size=image_size,
            channels=channels,
            use_skip_connections=use_skip_connections,
            norm_type=norm_type
        )
        
        # Create discriminator
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
        """Generate random samples
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.generator(z)
    
    def discriminate(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Discriminate real/fake images
        
        Args:
            x: Input images
            return_features: Whether to return intermediate features
            
        Returns:
            Discriminator predictions (and optionally features)
        """
        return self.discriminator(x, return_features=return_features)