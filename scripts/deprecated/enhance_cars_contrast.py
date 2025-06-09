#!/usr/bin/env python3
"""
Enhanced contrast adjustment for CARS microscopy images
Uses adaptive histogram equalization and target distribution matching
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import json


class CARSContrastEnhancer:
    def __init__(self, input_path: str, output_path: str = "data/processed_enhanced"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Preserve directory structure
        for split in ['train', 'val', 'test']:
            if (self.input_path / split).exists():
                (self.output_path / split).mkdir(exist_ok=True)
                
        self.stats = {
            'before': {'means': [], 'stds': []},
            'after': {'means': [], 'stds': []},
            'processed': 0
        }
        
    def enhance_cars_image(self, img: np.ndarray, target_mean: float = 140.0, target_std: float = 45.0) -> np.ndarray:
        """
        Enhance CARS image contrast using multiple techniques
        
        Args:
            img: Input image (8-bit or 16-bit)
            target_mean: Target mean intensity (0-255 scale)
            target_std: Target standard deviation
        """
        # Convert to float for processing
        if img.dtype == np.uint16:
            # First do percentile scaling for 16-bit
            p0_1, p99_9 = np.percentile(img, (0.1, 99.9))
            img_float = np.clip((img - p0_1) / (p99_9 - p0_1), 0, 1)
        elif img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32)
            
        # Method 1: Adaptive Histogram Equalization (CLAHE)
        # This preserves local structures while enhancing contrast
        img_uint8_temp = (img_float * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8_temp)
        img_clahe_float = img_clahe.astype(np.float32) / 255.0
        
        # Method 2: Gamma correction to brighten dark regions
        # Estimate optimal gamma based on current mean
        current_mean = img_clahe_float.mean() * 255
        if current_mean < target_mean:
            # Need to brighten - use gamma < 1
            gamma = 0.4 + (current_mean / target_mean) * 0.6  # Range: 0.4 to 1.0
        else:
            # Need to darken - use gamma > 1
            gamma = 1.0 + (current_mean / target_mean - 1.0) * 0.5
        gamma = np.clip(gamma, 0.3, 2.0)
        
        img_gamma = np.power(img_clahe_float, gamma)
        
        # Method 3: Linear stretching to match target distribution
        # Current stats
        current_mean = img_gamma.mean()
        current_std = img_gamma.std() + 1e-8
        
        # Normalize to zero mean, unit variance
        img_normalized = (img_gamma - current_mean) / current_std
        
        # Scale to target distribution (in 0-1 range)
        target_mean_01 = target_mean / 255.0
        target_std_01 = target_std / 255.0
        img_stretched = img_normalized * target_std_01 + target_mean_01
        
        # Clip to valid range
        img_final = np.clip(img_stretched, 0, 1)
        
        # Convert back to uint8
        img_enhanced = (img_final * 255).astype(np.uint8)
        
        return img_enhanced, gamma
    
    def process_dataset(self):
        """Process all images in the dataset"""
        print("🔄 Enhancing CARS image contrast...")
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = self.input_path / split
            if not split_dir.exists():
                continue
                
            print(f"\nProcessing {split} split...")
            
            # Find all images
            image_files = list(split_dir.glob('*.png')) + list(split_dir.glob('*.tif')) + list(split_dir.glob('*.tiff'))
            
            for img_path in tqdm(image_files, desc=f"Enhancing {split}"):
                # Load image
                img = np.array(Image.open(img_path))
                
                # Record original stats
                orig_mean = img.mean()
                orig_std = img.std()
                self.stats['before']['means'].append(orig_mean)
                self.stats['before']['stds'].append(orig_std)
                
                # Enhance contrast
                img_enhanced, gamma = self.enhance_cars_image(img)
                
                # Record enhanced stats
                enh_mean = img_enhanced.mean()
                enh_std = img_enhanced.std()
                self.stats['after']['means'].append(enh_mean)
                self.stats['after']['stds'].append(enh_std)
                
                # Save enhanced image
                output_path = self.output_path / split / f"{img_path.stem}.png"
                Image.fromarray(img_enhanced).save(output_path)
                
                self.stats['processed'] += 1
                
                # Debug info for first few images
                if self.stats['processed'] <= 3:
                    print(f"  {img_path.name}: mean {orig_mean:.1f}→{enh_mean:.1f}, "
                          f"std {orig_std:.1f}→{enh_std:.1f}, gamma={gamma:.2f}")
        
        # Print summary statistics
        print(f"\n✅ Enhancement complete!")
        print(f"  Processed: {self.stats['processed']} images")
        print(f"  Original: mean={np.mean(self.stats['before']['means']):.1f} ± {np.mean(self.stats['before']['stds']):.1f}")
        print(f"  Enhanced: mean={np.mean(self.stats['after']['means']):.1f} ± {np.mean(self.stats['after']['stds']):.1f}")
        
        # Save statistics
        stats_path = self.output_path / "enhancement_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'processed_count': self.stats['processed'],
                'original_mean': float(np.mean(self.stats['before']['means'])),
                'original_std': float(np.mean(self.stats['before']['stds'])),
                'enhanced_mean': float(np.mean(self.stats['after']['means'])),
                'enhanced_std': float(np.mean(self.stats['after']['stds']))
            }, f, indent=2)
    
    def create_comparison_plot(self, num_samples: int = 6):
        """Create before/after comparison visualization"""
        print("\n📊 Creating comparison visualization...")
        
        # Get sample images
        train_dir = self.input_path / 'train'
        if not train_dir.exists():
            print("No train directory found for visualization")
            return
            
        sample_files = list(train_dir.glob('*.png'))[:num_samples]
        if not sample_files:
            sample_files = list(train_dir.glob('*.tif'))[:num_samples]
            
        if not sample_files:
            print("No images found for visualization")
            return
            
        # Create comparison plot
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i, img_path in enumerate(sample_files):
            # Original
            img_orig = np.array(Image.open(img_path))
            axes[i, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=255 if img_orig.dtype == np.uint8 else img_orig.max())
            axes[i, 0].set_title(f'Original\nmean={img_orig.mean():.1f}')
            axes[i, 0].axis('off')
            
            # Enhanced
            enhanced_path = self.output_path / 'train' / f"{img_path.stem}.png"
            if enhanced_path.exists():
                img_enh = np.array(Image.open(enhanced_path))
                axes[i, 1].imshow(img_enh, cmap='gray', vmin=0, vmax=255)
                axes[i, 1].set_title(f'Enhanced\nmean={img_enh.mean():.1f}')
                axes[i, 1].axis('off')
                
                # Difference map
                diff = img_enh.astype(np.float32) - img_orig.astype(np.float32)
                axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-100, vmax=100)
                axes[i, 2].set_title('Difference')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'enhancement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison saved to: {self.output_path / 'enhancement_comparison.png'}")
    
    def create_dataset_config(self):
        """Create config for enhanced dataset"""
        config_content = f"""# CARS Dataset Configuration - Contrast Enhanced
# Generated by enhance_cars_contrast.py

dataset_name: "cars_microscopy_enhanced"
data_path: "{self.output_path.resolve()}"
image_size: 512
channels: 1
use_8bit: true

train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
split_seed: 42

batch_size: 16
num_workers: 0  # Mac compatibility
pin_memory: true
drop_last: true

augment_train: true
augment_val: false
augment_test: false

# Standard normalization for enhanced data
normalize_to_unit_range: true
mean: 0.5
std: 0.5

# Augmentation settings
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation_degrees: 15
  scale_range: [0.9, 1.1]
  translate_percent: 0.1
  brightness_factor: 0.05  # Reduced since we already enhanced
  contrast_factor: 0.05    # Reduced since we already enhanced
  gamma_range: [0.9, 1.1]  # Narrower range
  gaussian_noise_std: 0.02
  gaussian_blur_sigma: [0.1, 1.0]
  gaussian_blur_prob: 0.3
  elastic_transform: true
  elastic_alpha: 100
  elastic_sigma: 10
  elastic_prob: 0.3

validate_images: true
use_cache: true
cache_dir: "{self.output_path}/cache"
"""
        
        config_path = self.output_path / "dataset_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        print(f"\n📝 Config saved to: {config_path}")
        return config_path


def main():
    parser = argparse.ArgumentParser(description='Enhance contrast of CARS microscopy images')
    parser.add_argument('--input_path', type=str, default='data/processed_correct',
                       help='Path to processed data with low contrast')
    parser.add_argument('--output_path', type=str, default='data/processed_enhanced',
                       help='Output directory for enhanced data')
    parser.add_argument('--target_mean', type=float, default=140.0,
                       help='Target mean intensity (0-255)')
    parser.add_argument('--target_std', type=float, default=45.0,
                       help='Target standard deviation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create comparison visualization')
    
    args = parser.parse_args()
    
    print("🚀 CARS Contrast Enhancement")
    print("=" * 60)
    
    enhancer = CARSContrastEnhancer(args.input_path, args.output_path)
    enhancer.process_dataset()
    
    if args.visualize:
        enhancer.create_comparison_plot()
        
    enhancer.create_dataset_config()
    
    print("\n" + "=" * 60)
    print("✅ Contrast enhancement complete!")
    print(f"📁 Output directory: {args.output_path}")
    print("\n🎯 Next step:")
    print(f"python scripts/launch_improved_training.py --fixed_data_path {args.output_path}")


if __name__ == "__main__":
    main()