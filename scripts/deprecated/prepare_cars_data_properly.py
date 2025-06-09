#!/usr/bin/env python3
"""
Properly prepare CARS microscopy data with correct 16-bit to 8-bit conversion
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import json


class CARSDataPreparation:
    def __init__(self, input_path: str, output_path: str = "data/processed_correct"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split).mkdir(exist_ok=True)
            
        self.stats = {
            'processed_count': 0,
            'failed_count': 0,
            'intensity_stats': []
        }
        
    def convert_16bit_to_8bit(self, img_16bit: np.ndarray) -> np.ndarray:
        """Convert 16-bit CARS image to 8-bit with proper scaling"""
        # Use percentile-based scaling for CARS microscopy
        # This preserves the actual data distribution
        p_low, p_high = np.percentile(img_16bit, (0.1, 99.9))
        
        # Scale to 0-1 range
        img_normalized = np.clip((img_16bit - p_low) / (p_high - p_low), 0, 1)
        
        # Convert to 8-bit
        img_8bit = (img_normalized * 255).astype(np.uint8)
        
        return img_8bit, p_low, p_high
    
    def process_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """Process all TIF files and create train/val/test splits"""
        print("🔍 Finding TIF files...")
        
        # Find all TIF files
        tif_files = list(self.input_path.rglob('*.tif')) + list(self.input_path.rglob('*.tiff'))
        print(f"Found {len(tif_files)} TIF files")
        
        if not tif_files:
            raise ValueError("No TIF files found!")
            
        # Create train/val/test splits
        # First split: train+val vs test
        train_val_files, test_files = train_test_split(
            tif_files, test_size=test_ratio, random_state=seed
        )
        
        # Second split: train vs val
        train_files, val_files = train_test_split(
            train_val_files, 
            test_size=val_ratio/(train_ratio + val_ratio), 
            random_state=seed
        )
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        print(f"  Test: {len(test_files)}")
        
        # Process each split
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        all_8bit_means = []
        
        for split_name, file_list in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            for tif_path in tqdm(file_list, desc=f"Converting {split_name}"):
                try:
                    # Load 16-bit image
                    img_16bit = np.array(Image.open(tif_path))
                    
                    if img_16bit.dtype != np.uint16:
                        print(f"Warning: {tif_path.name} is not 16-bit, copying as-is")
                        # Just copy if already 8-bit
                        output_path = self.output_path / split_name / f"{tif_path.stem}.png"
                        shutil.copy2(tif_path, output_path)
                        continue
                    
                    # Convert to 8-bit with proper scaling
                    img_8bit, p_low, p_high = self.convert_16bit_to_8bit(img_16bit)
                    
                    # Save as PNG
                    output_path = self.output_path / split_name / f"{tif_path.stem}.png"
                    Image.fromarray(img_8bit).save(output_path)
                    
                    # Track statistics
                    self.stats['processed_count'] += 1
                    all_8bit_means.append(img_8bit.mean())
                    
                    # Debug info for first few images
                    if self.stats['processed_count'] <= 3:
                        print(f"  {tif_path.name}: 16-bit [{p_low:.0f}, {p_high:.0f}] → 8-bit mean={img_8bit.mean():.1f}")
                    
                except Exception as e:
                    print(f"Error processing {tif_path}: {e}")
                    self.stats['failed_count'] += 1
        
        # Calculate final statistics
        if all_8bit_means:
            self.stats['final_mean'] = np.mean(all_8bit_means)
            self.stats['final_std'] = np.std(all_8bit_means)
            
            print(f"\n✅ Processing complete!")
            print(f"  Processed: {self.stats['processed_count']} images")
            print(f"  Failed: {self.stats['failed_count']} images")
            print(f"  8-bit statistics: mean={self.stats['final_mean']:.1f} ± {self.stats['final_std']:.1f}")
            print(f"  (This should be around 128-180 for typical microscopy)")
        
        # Save statistics
        stats_path = self.output_path / "preparation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        return self.output_path
    
    def create_dataset_config(self):
        """Create Hydra config for the prepared dataset"""
        config_content = f"""# CARS Dataset Configuration - Properly Converted
# Generated by prepare_cars_data_properly.py

dataset_name: "cars_microscopy_correct"
data_path: "{self.output_path.resolve()}"
image_size: 512
channels: 1
use_8bit: true

train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
split_seed: 42

batch_size: 16  # Optimal for standard model
num_workers: 0  # For Mac compatibility
pin_memory: true
drop_last: true

augment_train: true
augment_val: false
augment_test: false

# Normalization for properly scaled data
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
cache_dir: "{self.output_path}/cache"
"""
        
        config_path = self.output_path / "dataset_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        print(f"\n📝 Config saved to: {config_path}")
        return config_path


def main():
    parser = argparse.ArgumentParser(description='Properly prepare CARS microscopy data')
    parser.add_argument('--input_path', type=str, default='data/raw',
                       help='Path to raw TIF files')
    parser.add_argument('--output_path', type=str, default='data/processed_correct',
                       help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splits')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    print("🚀 CARS Data Preparation - Proper 16-bit Conversion")
    print("=" * 60)
    
    preparator = CARSDataPreparation(args.input_path, args.output_path)
    output_path = preparator.process_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    preparator.create_dataset_config()
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print(f"📁 Output directory: {output_path}")
    print("\n🎯 Next step:")
    print(f"python scripts/launch_improved_training.py --fixed_data_path {output_path}")


if __name__ == "__main__":
    main()