#!/usr/bin/env python3
"""
Analyze raw TIF files to understand the original data range
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_raw_tif_files(raw_data_path: str):
    """Analyze raw TIF files before any processing"""
    raw_path = Path(raw_data_path)
    
    # Find all TIF files
    tif_files = list(raw_path.rglob('*.tif')) + list(raw_path.rglob('*.tiff'))
    
    if not tif_files:
        print(f"No TIF files found in {raw_path}")
        return
        
    print(f"Found {len(tif_files)} TIF files")
    
    # Analyze a sample
    sample_size = min(50, len(tif_files))
    sample_files = np.random.choice(tif_files, sample_size, replace=False)
    
    stats = {
        'uint8': {'count': 0, 'means': [], 'ranges': []},
        'uint16': {'count': 0, 'means': [], 'ranges': []},
        'other': {'count': 0, 'means': [], 'ranges': []}
    }
    
    for tif_path in tqdm(sample_files, desc="Analyzing TIF files"):
        img = np.array(Image.open(tif_path))
        
        if img.dtype == np.uint8:
            key = 'uint8'
            stats[key]['count'] += 1
            stats[key]['means'].append(img.mean())
            stats[key]['ranges'].append((img.min(), img.max()))
        elif img.dtype == np.uint16:
            key = 'uint16'
            stats[key]['count'] += 1
            stats[key]['means'].append(img.mean())
            stats[key]['ranges'].append((img.min(), img.max()))
            
            # Also check 8-bit equivalent
            img_8bit = (img / 65535.0 * 255).astype(np.uint8)
            print(f"  16-bit sample: mean={img.mean():.1f}, 8-bit equivalent: mean={img_8bit.mean():.1f}")
        else:
            key = 'other'
            stats[key]['count'] += 1
            stats[key]['means'].append(float(img.mean()))
            stats[key]['ranges'].append((float(img.min()), float(img.max())))
    
    # Print summary
    print("\n=== Raw TIF Analysis Results ===")
    
    for dtype, data in stats.items():
        if data['count'] > 0:
            print(f"\n{dtype} images: {data['count']}/{sample_size}")
            if data['means']:
                print(f"  Mean intensity: {np.mean(data['means']):.1f} ± {np.std(data['means']):.1f}")
                all_mins = [r[0] for r in data['ranges']]
                all_maxs = [r[1] for r in data['ranges']]
                print(f"  Range: [{np.min(all_mins):.1f}, {np.max(all_maxs):.1f}]")
                
                if dtype == 'uint16':
                    # Show 8-bit equivalent
                    mean_16 = np.mean(data['means'])
                    mean_8bit = mean_16 / 65535.0 * 255
                    print(f"  8-bit equivalent mean: {mean_8bit:.1f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Load and show a sample image
    sample_img = np.array(Image.open(sample_files[0]))
    
    # Original
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title(f'Sample Raw TIF\nShape: {sample_img.shape}, dtype: {sample_img.dtype}')
    ax1.axis('off')
    
    # Histogram
    ax2.hist(sample_img.flatten(), bins=50, alpha=0.7)
    ax2.set_title(f'Pixel Distribution\nMean: {sample_img.mean():.1f}, Range: [{sample_img.min()}, {sample_img.max()}]')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('raw_tif_analysis.png', dpi=150)
    print(f"\nVisualization saved to: raw_tif_analysis.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze raw TIF files')
    parser.add_argument('--raw_data_path', type=str, default='data/raw',
                       help='Path to raw data directory')
    
    args = parser.parse_args()
    analyze_raw_tif_files(args.raw_data_path)


if __name__ == "__main__":
    main()