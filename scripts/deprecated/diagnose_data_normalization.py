# scripts/diagnose_data_normalization.py
"""
Diagnose data normalization issues in CARS-FASTGAN
This script will help us understand the data flow and identify where normalization breaks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm

from src.data.dataset import CARSDataModule
from src.training.fastgan_module import FastGANModule


class DataNormalizationDiagnostic:
    def __init__(self, data_path: str, checkpoint_path: str = None):
        self.data_path = Path(data_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.output_dir = Path("outputs/normalization_diagnostic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_raw_data(self):
        """Analyze raw data statistics before any preprocessing"""
        print("\n=== Analyzing Raw Data ===")
        
        # Find raw images
        image_paths = []
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            image_paths.extend(list(self.data_path.rglob(ext)))
        
        if not image_paths:
            print(f"No images found in {self.data_path}")
            return None
            
        print(f"Found {len(image_paths)} images")
        
        # Sample some images
        sample_size = min(50, len(image_paths))
        sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        stats = {
            'raw_values': [],
            'bit_depth': [],
            'shapes': []
        }
        
        for img_path in tqdm(sample_paths, desc="Analyzing raw images"):
            img = np.array(Image.open(img_path))
            stats['raw_values'].append({
                'min': float(np.min(img)),
                'max': float(np.max(img)),
                'mean': float(np.mean(img)),
                'std': float(np.std(img)),
                'dtype': str(img.dtype)
            })
            stats['bit_depth'].append(img.dtype)
            stats['shapes'].append(img.shape)
            
        # Aggregate statistics
        all_mins = [s['min'] for s in stats['raw_values']]
        all_maxs = [s['max'] for s in stats['raw_values']]
        all_means = [s['mean'] for s in stats['raw_values']]
        all_stds = [s['std'] for s in stats['raw_values']]
        
        print(f"\nRaw Data Statistics:")
        print(f"  Value range: [{np.min(all_mins):.1f}, {np.max(all_maxs):.1f}]")
        print(f"  Mean ± Std: {np.mean(all_means):.1f} ± {np.mean(all_stds):.1f}")
        print(f"  Data types: {set(str(d) for d in stats['bit_depth'])}")
        print(f"  Shapes: {set(stats['shapes'])}")
        
        return stats
    
    def analyze_dataloader(self):
        """Analyze data after DataLoader preprocessing"""
        print("\n=== Analyzing DataLoader Output ===")
        
        # Create datamodule
        datamodule = CARSDataModule(
            data_path=str(self.data_path),
            batch_size=8,
            num_workers=0,
            image_size=512,
            use_8bit=True,  # Based on your config
            augment_train=True,
            augment_val=False
        )
        
        datamodule.setup()
        
        # Analyze train and val loaders
        for split, loader in [("train", datamodule.train_dataloader()), 
                              ("val", datamodule.val_dataloader())]:
            print(f"\n{split.upper()} Loader:")
            
            # Get a few batches
            batch_stats = []
            for i, batch in enumerate(loader):
                if i >= 5:  # Analyze 5 batches
                    break
                    
                images = batch['image']
                batch_stats.append({
                    'min': float(images.min()),
                    'max': float(images.max()),
                    'mean': float(images.mean()),
                    'std': float(images.std()),
                    'shape': images.shape
                })
            
            # Print statistics
            print(f"  Shape: {batch_stats[0]['shape']}")
            print(f"  Value range: [{np.mean([s['min'] for s in batch_stats]):.3f}, "
                  f"{np.mean([s['max'] for s in batch_stats]):.3f}]")
            print(f"  Mean ± Std: {np.mean([s['mean'] for s in batch_stats]):.3f} ± "
                  f"{np.mean([s['std'] for s in batch_stats]):.3f}")
        
        return datamodule
    
    def analyze_model_output(self):
        """Analyze model output if checkpoint provided"""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            print("\n=== No checkpoint provided, skipping model analysis ===")
            return None
            
        print("\n=== Analyzing Model Output ===")
        
        # Load model
        model = FastGANModule.load_from_checkpoint(str(self.checkpoint_path))
        model.eval()
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        
        # Generate samples
        with torch.no_grad():
            z = torch.randn(16, model.hparams.latent_dim, device=device)
            fake_images = model.model.generator(z)
            
            print(f"\nGenerator Output (raw):")
            print(f"  Shape: {fake_images.shape}")
            print(f"  Value range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
            print(f"  Mean ± Std: {fake_images.mean():.3f} ± {fake_images.std():.3f}")
            
            # Test different denormalization strategies
            print(f"\nDenormalization Tests:")
            
            # Strategy 1: Simple rescale from [-1, 1] to [0, 255]
            denorm1 = (fake_images + 1) * 127.5
            print(f"  [-1,1] → [0,255]: mean={denorm1.mean():.1f}, std={denorm1.std():.1f}")
            
            # Strategy 2: Assuming already in [0, 1]
            denorm2 = fake_images * 255
            print(f"  [0,1] → [0,255]: mean={denorm2.mean():.1f}, std={denorm2.std():.1f}")
            
            # Strategy 3: Custom denormalization matching training stats
            # This assumes normalize with mean=0.5, std=0.5
            denorm3 = fake_images * 0.5 + 0.5  # Back to [0, 1]
            denorm3 = denorm3 * 255  # To [0, 255]
            print(f"  Custom denorm: mean={denorm3.mean():.1f}, std={denorm3.std():.1f}")
            
        return fake_images
    
    def create_diagnostic_plots(self, raw_stats, datamodule, fake_images=None):
        """Create diagnostic visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Normalization Diagnostic', fontsize=16)
        
        # Plot 1: Raw data distribution
        ax = axes[0, 0]
        raw_means = [s['mean'] for s in raw_stats['raw_values']]
        ax.hist(raw_means, bins=20, alpha=0.7, color='blue')
        ax.set_title('Raw Data Mean Values')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        
        # Plot 2: DataLoader output distribution
        ax = axes[0, 1]
        loader = datamodule.val_dataloader()
        batch = next(iter(loader))
        images = batch['image']
        ax.hist(images.numpy().flatten(), bins=50, alpha=0.7, color='green')
        ax.set_title(f'DataLoader Output\n[{images.min():.2f}, {images.max():.2f}]')
        ax.set_xlabel('Normalized Value')
        
        # Plot 3: Model output distribution (if available)
        ax = axes[0, 2]
        if fake_images is not None:
            ax.hist(fake_images.cpu().numpy().flatten(), bins=50, alpha=0.7, color='red')
            ax.set_title(f'Model Output (Raw)\n[{fake_images.min():.2f}, {fake_images.max():.2f}]')
        else:
            ax.text(0.5, 0.5, 'No model output', ha='center', va='center', 
                   transform=ax.transAxes)
        ax.set_xlabel('Value')
        
        # Plot 4-6: Visual comparisons
        # Raw image
        ax = axes[1, 0]
        sample_path = list(self.data_path.rglob('*.tif'))[0] if list(self.data_path.rglob('*.tif')) else list(self.data_path.rglob('*.png'))[0]
        raw_img = np.array(Image.open(sample_path))
        if raw_img.dtype == np.uint16:
            raw_img = (raw_img / 65535.0 * 255).astype(np.uint8)
        ax.imshow(raw_img, cmap='gray')
        ax.set_title(f'Raw Image\nMean: {raw_img.mean():.1f}')
        ax.axis('off')
        
        # DataLoader image
        ax = axes[1, 1]
        dataloader_img = images[0, 0].numpy()
        # Denormalize for display
        dataloader_img = (dataloader_img * 0.5 + 0.5)  # Assuming normalize(0.5, 0.5)
        dataloader_img = np.clip(dataloader_img, 0, 1)
        ax.imshow(dataloader_img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'DataLoader Image\nRange: [{images[0,0].min():.2f}, {images[0,0].max():.2f}]')
        ax.axis('off')
        
        # Model output (if available)
        ax = axes[1, 2]
        if fake_images is not None:
            model_img = fake_images[0, 0].cpu().numpy()
            # Try the most likely denormalization
            model_img_display = (model_img + 1) / 2  # Assuming tanh output
            model_img_display = np.clip(model_img_display, 0, 1)
            ax.imshow(model_img_display, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Generated Image\nRaw mean: {model_img.mean():.2f}')
        else:
            ax.text(0.5, 0.5, 'No model output', ha='center', va='center', 
                   transform=ax.transAxes)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'normalization_diagnostic.png', dpi=150)
        plt.close()
        
        print(f"\nDiagnostic plot saved to: {self.output_dir / 'normalization_diagnostic.png'}")
    
    def run_diagnostic(self):
        """Run complete diagnostic"""
        print("🔍 Running Data Normalization Diagnostic")
        print("=" * 50)
        
        # Analyze raw data
        raw_stats = self.analyze_raw_data()
        if raw_stats is None:
            return
        
        # Analyze dataloader
        datamodule = self.analyze_dataloader()
        
        # Analyze model output
        fake_images = self.analyze_model_output()
        
        # Create plots
        self.create_diagnostic_plots(raw_stats, datamodule, fake_images)
        
        print("\n" + "=" * 50)
        print("✅ Diagnostic complete!")
        print(f"📊 Check {self.output_dir} for visualization")
        
        # Recommendations
        print("\n💡 Key Findings:")
        if fake_images is not None and fake_images.mean() < -0.9:
            print("  ⚠️  Model outputs are near -1 (tanh activation)")
            print("  → Need to fix denormalization in evaluation/generation")
        
        raw_mean = np.mean([s['mean'] for s in raw_stats['raw_values']])
        if raw_mean > 100:
            print(f"  ℹ️  Raw data has mean ~{raw_mean:.0f} (typical for 8-bit images)")
            print("  → DataLoader normalization to [-1, 1] is expected")


def main():
    parser = argparse.ArgumentParser(description='Diagnose data normalization issues')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    
    args = parser.parse_args()
    
    diagnostic = DataNormalizationDiagnostic(args.data_path, args.checkpoint_path)
    diagnostic.run_diagnostic()


if __name__ == "__main__":
    main()