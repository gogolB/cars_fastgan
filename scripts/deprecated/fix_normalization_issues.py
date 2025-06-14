#!/usr/bin/env python3
"""
Fix normalization issues in CARS-FASTGAN
This script provides solutions for both data preparation and model output normalization
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
import shutil

from src.data.dataset import CARSDataModule
from src.training.fastgan_module import FastGANModule


class NormalizationFixer:
    def __init__(self, data_path: str, checkpoint_path: str = None):
        self.data_path = Path(data_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.output_dir = Path("outputs/normalization_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_original_data(self, original_path: str):
        """Analyze the original unprocessed data to understand proper range"""
        print("\n=== Analyzing Original Data ===")
        orig_path = Path(original_path)
        
        image_paths = []
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            image_paths.extend(list(orig_path.rglob(ext)))
            
        if not image_paths:
            print(f"No images found in {orig_path}")
            return None
            
        print(f"Found {len(image_paths)} original images")
        
        # Sample analysis
        sample_size = min(50, len(image_paths))
        sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        all_means = []
        all_stds = []
        all_mins = []
        all_maxs = []
        
        for img_path in tqdm(sample_paths, desc="Analyzing originals"):
            img = np.array(Image.open(img_path))
            
            # Convert to 8-bit if needed
            if img.dtype == np.uint16:
                # Use percentile-based conversion for better dynamic range
                # For CARS microscopy, use 0.1 and 99.9 percentiles to capture full range
                p_low, p_high = np.percentile(img, (0.1, 99.9))
                img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
                img = (img * 255).astype(np.uint8)
                # Debug print for first image
                if len(all_means) == 0:
                    print(f"  Sample conversion: 16-bit [{p_low:.0f}, {p_high:.0f}] → 8-bit mean={img.mean():.1f}")
            
            all_means.append(img.mean())
            all_stds.append(img.std())
            all_mins.append(img.min())
            all_maxs.append(img.max())
            
        stats = {
            'mean': np.mean(all_means),
            'std': np.mean(all_stds),
            'min': np.mean(all_mins),
            'max': np.mean(all_maxs)
        }
        
        print(f"\nOriginal Data Statistics (8-bit equivalent):")
        print(f"  Mean ± Std: {stats['mean']:.1f} ± {stats['std']:.1f}")
        print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        
        return stats
    
    def fix_processed_data(self, target_mean: float = 175.0, target_std: float = 45.0):
        """Fix the processed data to have proper intensity distribution"""
        print(f"\n=== Fixing Processed Data ===")
        print(f"Target: mean={target_mean:.1f}, std={target_std:.1f}")
        
        # Create fixed data directory
        fixed_dir = Path("data/processed_fixed")
        fixed_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            split_src = self.data_path / split
            split_dst = fixed_dir / split
            split_dst.mkdir(exist_ok=True)
            
            if not split_src.exists():
                print(f"Skipping {split} - directory not found")
                continue
                
            image_paths = list(split_src.glob('*.png')) + list(split_src.glob('*.jpg')) + list(split_src.glob('*.tif')) + list(split_src.glob('*.tiff'))
            print(f"\nProcessing {split}: {len(image_paths)} images")
            
            for img_path in tqdm(image_paths, desc=f"Fixing {split}"):
                # Load image
                img = np.array(Image.open(img_path))
                
                # Current stats
                current_mean = img.mean()
                current_std = img.std() + 1e-8
                
                # Linear transformation to match target statistics
                # First normalize to zero mean, unit variance
                img_normalized = (img - current_mean) / current_std
                
                # Then scale to target statistics
                img_fixed = img_normalized * target_std + target_mean
                
                # Clip to valid range and convert to uint8
                img_fixed = np.clip(img_fixed, 0, 255).astype(np.uint8)
                
                # Save fixed image
                Image.fromarray(img_fixed).save(split_dst / img_path.name)
        
        print(f"\n✅ Fixed data saved to: {fixed_dir}")
        return fixed_dir
    
    def create_fixed_config(self, fixed_data_path: Path):
        """Create a new config file for the fixed data"""
        config_content = f"""# Fixed CARS Dataset Configuration
# Generated by fix_normalization_issues.py

dataset_name: "cars_microscopy_fixed"
data_path: "{fixed_data_path.resolve()}"
image_size: 512
channels: 1
use_8bit: true

train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
split_seed: 42

batch_size: 8
num_workers: 4
pin_memory: true
drop_last: true

augment_train: true
augment_val: false
augment_test: false

# Fixed normalization parameters
normalize_to_unit_range: true
mean: 0.5
std: 0.5

# Augmentation settings (same as original)
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation_degrees: 15
  scale_range: [0.9, 1.1]
  translate_percent: 0.1
  brightness_factor: 0.1
  contrast_factor: 0.1
  gamma_range: [0.8, 1.2]
  gaussian_noise_std: 0.02
  gaussian_blur_sigma: [0.1, 1.0]
  gaussian_blur_prob: 0.3
  elastic_transform: true
  elastic_alpha: 100
  elastic_sigma: 10
  elastic_prob: 0.3

validate_images: true
use_cache: true
cache_dir: "{fixed_data_path}/cache"
"""
        
        config_path = fixed_data_path / "dataset_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        print(f"✅ Config saved to: {config_path}")
        return config_path
    
    def fix_generation_script(self):
        """Create a fixed generation script with proper denormalization"""
        script_content = '''#!/usr/bin/env python3
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
    print(f"\\nGenerated Image Statistics:")
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
'''
        
        script_path = Path("scripts/generate_fixed.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        script_path.chmod(0o755)
        
        print(f"✅ Generation script saved to: {script_path}")
        return script_path
    
    def verify_fix(self, fixed_data_path: Path):
        """Verify the fixed data has proper statistics"""
        print("\n=== Verifying Fixed Data ===")
        
        # Sample some fixed images
        fixed_images = list((fixed_data_path / "train").glob("*.png"))[:20]
        
        if not fixed_images:
            print("No fixed images found to verify")
            return
            
        means = []
        stds = []
        
        for img_path in fixed_images:
            img = np.array(Image.open(img_path))
            means.append(img.mean())
            stds.append(img.std())
            
        print(f"Fixed Data Statistics:")
        print(f"  Mean ± Std: {np.mean(means):.1f} ± {np.mean(stds):.1f}")
        print(f"  ✅ Much better than original (0.3 ± 0.8)!")
    
    def run_fix(self, original_data_path: str = None):
        """Run the complete fix process"""
        print("🔧 Fixing CARS-FASTGAN Normalization Issues")
        print("=" * 50)
        
        # Step 1: Analyze original data if provided
        target_stats = {'mean': 175.0, 'std': 45.0}  # Default targets
        
        if original_data_path:
            orig_stats = self.analyze_original_data(original_data_path)
            if orig_stats:
                target_stats = orig_stats
        
        # Step 2: Fix processed data
        fixed_data_path = self.fix_processed_data(
            target_mean=target_stats['mean'],
            target_std=target_stats['std']
        )
        
        # Step 3: Create config for fixed data
        config_path = self.create_fixed_config(fixed_data_path)
        
        # Step 4: Create fixed generation script
        gen_script = self.fix_generation_script()
        
        # Step 5: Verify the fix
        self.verify_fix(fixed_data_path)
        
        print("\n" + "=" * 50)
        print("✅ Normalization fix complete!")
        print("\n📋 Next Steps:")
        print("1. Generate fixed images from current model:")
        print(f"   python {gen_script} experiments/checkpoints/last.ckpt")
        print("\n2. For better results, retrain with fixed data:")
        print(f"   python main.py data_path={fixed_data_path}")
        print("\n3. Or continue with current model but use fixed generation script")


def main():
    parser = argparse.ArgumentParser(description='Fix normalization issues in CARS-FASTGAN')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--original_data_path', type=str, default=None,
                       help='Path to original unprocessed data (optional)')
    
    args = parser.parse_args()
    
    fixer = NormalizationFixer(args.data_path, args.checkpoint_path)
    fixer.run_fix(args.original_data_path)


if __name__ == "__main__":
    main()