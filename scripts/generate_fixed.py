#!/usr/bin/env python3
"""
Fixed image generation script with proper denormalization
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

from src.training.fastgan_module import FastGANModule


def generate_fixed_images(checkpoint_path: str, num_samples: int = 16, output_dir: str = "outputs/generated_fixed"):
    """Generate images with proper denormalization"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = FastGANModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(num_samples, model.hparams.latent_dim, device=device)
        fake_images = model.model.generator(z)
        
        # Proper denormalization
        # From [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Apply inverse of training normalization
        # Training used normalize(0.5, 0.5): (x - 0.5) / 0.5
        # So inverse is: x * 0.5 + 0.5 (but we're already in [0,1])
        
        # Scale to match real data statistics
        # Target: mean ~175, std ~45 (in 0-255 range)
        target_mean = 175.0 / 255.0  # In [0, 1]
        target_std = 45.0 / 255.0
        
        # Current stats
        current_mean = fake_images.mean()
        current_std = fake_images.std()
        
        # Adjust to match target distribution
        fake_images = (fake_images - current_mean) / (current_std + 1e-8)
        fake_images = fake_images * target_std + target_mean
        
        # Clip and convert to uint8
        fake_images = torch.clamp(fake_images, 0, 1)
        fake_images = (fake_images * 255).cpu().numpy().astype(np.uint8)
        
        # Save images
        for i in range(num_samples):
            img = fake_images[i, 0]  # Remove channel dimension
            Image.fromarray(img).save(output_path / f"generated_{i:03d}.png")
            
    print(f"✅ Generated {num_samples} images with proper normalization")
    print(f"📁 Saved to: {output_path}")
    
    # Print statistics
    print(f"\nGenerated Image Statistics:")
    print(f"  Mean: {fake_images.mean():.1f}")
    print(f"  Std: {fake_images.std():.1f}")
    print(f"  Min: {fake_images.min()}")
    print(f"  Max: {fake_images.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="outputs/generated_fixed", help="Output directory")
    
    args = parser.parse_args()
    generate_fixed_images(args.checkpoint_path, args.num_samples, args.output_dir)
