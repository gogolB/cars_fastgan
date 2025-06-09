#!/usr/bin/env python3
"""
Generate properly denormalized samples from CARS-FASTGAN checkpoint
Handles the [-1, 1] to [0, 255] conversion correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.training.fastgan_module import FastGANModule


def generate_samples(checkpoint_path: str, 
                    num_samples: int = 16,
                    output_dir: str = "outputs/generated_samples",
                    save_grid: bool = True,
                    save_individual: bool = True):
    """Generate samples with proper denormalization"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = FastGANModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"   - Latent dim: {model.hparams.latent_dim}")
    print(f"   - Image size: {model.hparams.image_size}")
    
    # Generate images
    print(f"\n🎨 Generating {num_samples} samples...")
    
    all_samples = []
    batch_size = min(16, num_samples)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            current_batch = min(batch_size, num_samples - i)
            
            # Generate latent vectors
            z = torch.randn(current_batch, model.hparams.latent_dim, device=device)
            
            # Generate images (output is in [-1, 1] due to tanh)
            fake_images = model.model.generator(z)
            
            # Proper denormalization: [-1, 1] -> [0, 255]
            fake_images = (fake_images + 1) / 2  # to [0, 1]
            fake_images = (fake_images * 255).cpu().numpy().astype(np.uint8)
            
            all_samples.append(fake_images)
    
    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    
    print(f"\n📊 Generated Image Statistics:")
    print(f"   - Shape: {all_samples.shape}")
    print(f"   - Mean: {all_samples.mean():.1f}")
    print(f"   - Std: {all_samples.std():.1f}")
    print(f"   - Min: {all_samples.min()}")
    print(f"   - Max: {all_samples.max()}")
    
    # Save individual images
    if save_individual:
        individual_dir = output_path / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving individual images...")
        for i in range(num_samples):
            img = all_samples[i, 0]  # Remove channel dimension
            img_path = individual_dir / f"sample_{i:04d}.png"
            Image.fromarray(img).save(img_path)
        
        print(f"   ✅ Saved to: {individual_dir}")
    
    # Create and save grid
    if save_grid:
        print(f"\n🖼️  Creating sample grid...")
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
        fig.suptitle(f'Generated CARS Microscopy Images\n'
                     f'Checkpoint: {Path(checkpoint_path).name}', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten() if grid_size > 1 else [axes]
        
        for i in range(grid_size * grid_size):
            ax = axes[i]
            if i < num_samples:
                img = all_samples[i, 0]
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax.set_title(f'Sample {i+1}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save grid
        grid_path = output_path / "sample_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Grid saved to: {grid_path}")
    
    # Also save a comparison with enhanced contrast
    print(f"\n🔍 Creating enhanced contrast comparison...")
    
    fig, axes = plt.subplots(2, min(8, num_samples), figsize=(min(8, num_samples) * 3, 6))
    fig.suptitle('Original vs Enhanced Contrast', fontsize=16)
    
    for i in range(min(8, num_samples)):
        # Original
        img = all_samples[i, 0]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Enhanced contrast
        from PIL import ImageEnhance
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = np.array(enhancer.enhance(3.0))
        
        axes[1, i].imshow(enhanced, cmap='gray')
        axes[1, i].set_title(f'Enhanced {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    enhanced_path = output_path / "enhanced_comparison.png"
    plt.savefig(enhanced_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Enhanced comparison saved to: {enhanced_path}")
    
    print(f"\n✅ Generation complete!")
    print(f"📁 All outputs saved to: {output_path}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description='Generate properly denormalized samples from CARS-FASTGAN checkpoint'
    )
    parser.add_argument('checkpoint_path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='outputs/generated_samples',
                       help='Output directory for samples')
    parser.add_argument('--no_grid', action='store_true',
                       help='Skip creating sample grid')
    parser.add_argument('--no_individual', action='store_true',
                       help='Skip saving individual images')
    
    args = parser.parse_args()
    
    generate_samples(
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        save_grid=not args.no_grid,
        save_individual=not args.no_individual
    )


if __name__ == "__main__":
    main()