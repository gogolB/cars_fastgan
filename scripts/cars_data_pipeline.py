#!/usr/bin/env python3
"""
CARS Data Pipeline - Comprehensive data preparation, enhancement, and analysis
Combines functionality from multiple data-related scripts into a unified pipeline

This script replaces:
- prepare_cars_data.py
- prepare_cars_data_properly.py
- fix_normalization_issues.py
- enhance_cars_contrast.py
- analyze_raw_tif_files.py
- src/data/analyze.py
- diagnose_data_normalization.py

Usage:
    # Full pipeline with analysis
    python cars_data_pipeline.py --input_path /path/to/raw/data --output_path data/processed --full_pipeline
    
    # Just prepare data
    python cars_data_pipeline.py --input_path /path/to/raw/data --output_path data/processed --task prepare
    
    # Analyze existing data
    python cars_data_pipeline.py --input_path data/processed --task analyze
    
    # Enhance contrast with multiple methods
    python cars_data_pipeline.py --input_path data/processed --task enhance --enhancement_methods all
    
    # Fix existing processed data
    python cars_data_pipeline.py --input_path data/processed --task fix
"""

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import stats
from skimage import exposure, filters, measure
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')


class CARSDataPipeline:
    """Comprehensive CARS data pipeline for preparation, enhancement, and analysis"""
    
    def __init__(self, input_path: str, output_path: str = "data/processed"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'train': self.output_path / 'train',
            'val': self.output_path / 'val',
            'test': self.output_path / 'test',
            'analysis': self.output_path / 'analysis',
            'visualizations': self.output_path / 'visualizations',
            'enhanced': self.output_path / 'enhanced',
            'reports': self.output_path / 'reports'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        self.stats = {
            'processed_count': 0,
            'failed_count': 0,
            'enhancement_results': {},
            'analysis_results': {},
            'normalization_diagnostics': {}
        }
        
        print(f"🔧 CARS Data Pipeline Initialized")
        print(f"📁 Input: {self.input_path}")
        print(f"📁 Output: {self.output_path}")
        print("=" * 60)
    
    def find_images(self, path: Optional[Path] = None) -> List[Path]:
        """Find all image files in the specified path"""
        search_path = path or self.input_path
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(search_path.rglob(f'*{ext}')))
            image_paths.extend(list(search_path.rglob(f'*{ext.upper()}')))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        return image_paths
    
    # ==================== Data Preparation Methods ====================
    
    def analyze_bit_depth_impact(self, sample_size: int = 50) -> Dict:
        """Analyze the impact of 16-bit to 8-bit conversion"""
        print("\n📊 Analyzing bit depth conversion impact...")
        
        image_paths = self.find_images()
        if not image_paths:
            print("❌ No images found!")
            return {}
        
        sample_paths = np.random.choice(image_paths, min(sample_size, len(image_paths)), replace=False)
        
        metrics = {
            'ssim_scores': [],
            'mse_scores': [],
            'psnr_scores': [],
            'dynamic_range_16bit': [],
            'dynamic_range_8bit': [],
            'info_loss_percent': []
        }
        
        for img_path in tqdm(sample_paths, desc="Analyzing bit depth"):
            try:
                img_16bit = np.array(Image.open(img_path))
                
                if img_16bit.dtype != np.uint16:
                    continue
                
                # Convert using different methods
                # Method 1: Simple scaling
                img_8bit_simple = (img_16bit / 65535.0 * 255).astype(np.uint8)
                
                # Method 2: Percentile scaling (recommended for CARS)
                p_low, p_high = np.percentile(img_16bit, (0.1, 99.9))
                img_8bit_percentile = np.clip((img_16bit - p_low) / (p_high - p_low), 0, 1)
                img_8bit_percentile = (img_8bit_percentile * 255).astype(np.uint8)
                
                # Calculate metrics on percentile method
                img_16bit_norm = img_16bit.astype(np.float32) / 65535.0
                img_8bit_norm = img_8bit_percentile.astype(np.float32) / 255.0
                
                ssim_score = ssim(img_16bit_norm, img_8bit_norm, data_range=1.0)
                mse = np.mean((img_16bit_norm - img_8bit_norm) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                
                metrics['ssim_scores'].append(ssim_score)
                metrics['mse_scores'].append(mse)
                metrics['psnr_scores'].append(psnr)
                metrics['dynamic_range_16bit'].append(np.max(img_16bit) - np.min(img_16bit))
                metrics['dynamic_range_8bit'].append(np.max(img_8bit_percentile) - np.min(img_8bit_percentile))
                
                unique_16bit = len(np.unique(img_16bit))
                unique_8bit = len(np.unique(img_8bit_percentile))
                info_loss = (1 - unique_8bit / unique_16bit) * 100
                metrics['info_loss_percent'].append(info_loss)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Generate summary
        summary = {
            'ssim_mean': np.mean(metrics['ssim_scores']),
            'psnr_mean': np.mean(metrics['psnr_scores']),
            'info_loss_mean': np.mean(metrics['info_loss_percent']),
            'recommendation': 'use_8bit' if np.mean(metrics['ssim_scores']) > 0.95 else 'keep_16bit'
        }
        
        self.stats['analysis_results']['bit_depth'] = summary
        
        # Create visualization
        self._create_bit_depth_visualization(metrics)
        
        return summary
    
    def prepare_dataset(self, 
                       use_8bit: bool = True,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       seed: int = 42,
                       percentile_low: float = 0.1,
                       percentile_high: float = 99.9,
                       use_log_transform: bool = False) -> Dict:
        """Prepare dataset with proper conversion and splitting
        
        Args:
            use_8bit: Convert to 8-bit format
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for splits
            percentile_low: Lower percentile for normalization (default 0.1)
            percentile_high: Upper percentile for normalization (default 99.9)
            use_log_transform: Apply log transformation before normalization (good for sparse data)
        """
        print("\n🔄 Preparing dataset...")
        print(f"   Using percentiles: ({percentile_low}, {percentile_high})")
        if use_log_transform:
            print("   Applying log transformation for sparse data")
        
        image_paths = self.find_images()
        if not image_paths:
            raise ValueError("No images found!")
        
        print(f"Found {len(image_paths)} images")
        
        # Create train/val/test splits
        train_val_paths, test_paths = train_test_split(
            image_paths, test_size=test_ratio, random_state=seed
        )
        train_paths, val_paths = train_test_split(
            train_val_paths, test_size=val_ratio/(train_ratio + val_ratio), random_state=seed
        )
        
        splits = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
        
        print(f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Process each split
        all_stats = []
        
        for split_name, paths in splits.items():
            print(f"\nProcessing {split_name} split...")
            
            for img_path in tqdm(paths, desc=f"Processing {split_name}"):
                try:
                    img = np.array(Image.open(img_path))
                    
                    # Convert bit depth if needed
                    if img.dtype == np.uint16 and use_8bit:
                        # Apply log transformation if requested (before percentile normalization)
                        if use_log_transform:
                            # Convert to float and apply log(1 + x) to handle zeros
                            img_float = img.astype(np.float32)
                            img = np.log1p(img_float)
                            
                            # Now apply percentile normalization on log-transformed data
                            p_low, p_high = np.percentile(img, (percentile_low, percentile_high))
                            
                            if self.stats['processed_count'] == 0:
                                print(f"  First image (log-transformed) percentiles: p{percentile_low}={p_low:.2f}, p{percentile_high}={p_high:.2f}")
                        else:
                            # Standard percentile normalization
                            p_low, p_high = np.percentile(img, (percentile_low, percentile_high))
                            
                            if self.stats['processed_count'] == 0:
                                print(f"  First image percentiles: p{percentile_low}={p_low:.0f}, p{percentile_high}={p_high:.0f}")
                        
                        # Normalize to [0, 1] range
                        if p_high > p_low:
                            img_normalized = np.clip((img - p_low) / (p_high - p_low), 0, 1)
                        else:
                            img_normalized = np.zeros_like(img, dtype=np.float32)
                        
                        # Convert to uint8 with proper scaling
                        img = (img_normalized * 255).astype(np.uint8)
                        
                    elif img.dtype == np.uint16 and not use_8bit:
                        # Keep as 16-bit but apply percentile normalization
                        if use_log_transform:
                            img_float = img.astype(np.float32)
                            img = np.log1p(img_float)
                        
                        p_low, p_high = np.percentile(img, (percentile_low, percentile_high))
                        if p_high > p_low:
                            img = np.clip((img - p_low) / (p_high - p_low) * 65535, 0, 65535).astype(np.uint16)
                        else:
                            img = np.zeros_like(img, dtype=np.uint16)
                    
                    elif img.dtype == np.uint8 and use_log_transform:
                        # Apply log transform to 8-bit images if requested
                        img_float = img.astype(np.float32)
                        img_log = np.log1p(img_float)
                        
                        # Normalize log values to 0-255 range
                        p_low, p_high = np.percentile(img_log, (percentile_low, percentile_high))
                        if p_high > p_low:
                            img_normalized = np.clip((img_log - p_low) / (p_high - p_low), 0, 1)
                            img = (img_normalized * 255).astype(np.uint8)
                        
                    elif img.dtype != np.uint8 and use_8bit:
                        # Handle other dtypes
                        img = img.astype(np.uint8)
                    
                    # Ensure grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    
                    # Save processed image - ensure proper format
                    output_path = self.dirs[split_name] / f"{img_path.stem}.png"
                    
                    # Verify image is in correct format before saving
                    if use_8bit and img.dtype != np.uint8:
                        print(f"  Warning: Converting {img.dtype} to uint8 before saving")
                        img = img.astype(np.uint8)
                    
                    # Save with PIL to ensure proper PNG encoding
                    Image.fromarray(img).save(output_path, 'PNG')
                    
                    all_stats.append({
                        'split': split_name,
                        'mean': img.mean(),
                        'std': img.std(),
                        'min': img.min(),
                        'max': img.max()
                    })
                    
                    self.stats['processed_count'] += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    self.stats['failed_count'] += 1
        
        # Calculate dataset statistics
        stats_df = pd.DataFrame(all_stats)
        dataset_stats = {
            'global_mean': stats_df['mean'].mean(),
            'global_std': stats_df['std'].mean(),
            'processed': self.stats['processed_count'],
            'failed': self.stats['failed_count']
        }
        
        print(f"\n✅ Dataset prepared - Mean: {dataset_stats['global_mean']:.1f}, Std: {dataset_stats['global_std']:.1f}")
        
        return dataset_stats
    
    def fix_existing_uint16_images(self, percentile_low: float = 0.1, percentile_high: float = 99.9, use_log_transform: bool = False) -> bool:
        """Fix any existing uint16 images in processed directories
        
        Args:
            percentile_low: Lower percentile for normalization
            percentile_high: Upper percentile for normalization
            use_log_transform: Apply log transformation before normalization
        """
        print(f"\n🔧 Checking for uint16 images to fix (using percentiles: {percentile_low}, {percentile_high})...")
        if use_log_transform:
            print("   Will apply log transformation")
        
        fixed_count = 0
        checked_count = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dirs[split]
            if not split_dir.exists():
                continue
                
            image_files = list(split_dir.glob('*.png')) + list(split_dir.glob('*.tif'))
            
            for img_path in tqdm(image_files, desc=f"Checking {split}"):
                checked_count += 1
                img = np.array(Image.open(img_path))
                
                if img.dtype == np.uint16:
                    # Apply log transformation if requested
                    if use_log_transform:
                        img_float = img.astype(np.float32)
                        img = np.log1p(img_float)
                    
                    # Apply proper percentile normalization
                    p_low, p_high = np.percentile(img, (percentile_low, percentile_high))
                    
                    if p_high > p_low:
                        img_normalized = np.clip((img - p_low) / (p_high - p_low), 0, 1)
                        img_uint8 = (img_normalized * 255).astype(np.uint8)
                    else:
                        img_uint8 = np.zeros_like(img, dtype=np.uint8)
                    
                    # Save fixed image
                    Image.fromarray(img_uint8).save(img_path, 'PNG')
                    fixed_count += 1
                    
                    if fixed_count == 1:
                        print(f"  Fixed first image: {img_path.name}")
                        print(f"    Original: dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
                        print(f"    Fixed: dtype={img_uint8.dtype}, range=[{img_uint8.min()}, {img_uint8.max()}]")
        
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} uint16 images out of {checked_count} total")
            return True
        else:
            print(f"✅ All {checked_count} images are already in correct format")
            return False
    
    # ==================== Enhancement Methods ====================
    
    def enhance_dataset(self, methods: Union[str, List[str]] = 'all') -> Dict:
        """Apply multiple enhancement techniques to the dataset"""
        print("\n🎨 Enhancing dataset...")
        
        if methods == 'all':
            methods = ['clahe', 'gamma', 'histogram_equalization', 'adaptive_gamma', 'unsharp_mask']
        elif isinstance(methods, str):
            methods = [methods]
        
        results = {}
        
        for method in methods:
            print(f"\nApplying {method} enhancement...")
            method_dir = self.dirs['enhanced'] / method
            method_dir.mkdir(exist_ok=True)
            
            # Process each split
            for split in ['train', 'val', 'test']:
                split_src = self.dirs[split]
                split_dst = method_dir / split
                split_dst.mkdir(exist_ok=True)
                
                image_paths = list(split_src.glob('*.png'))
                
                for img_path in tqdm(image_paths, desc=f"Enhancing {split} with {method}"):
                    img = np.array(Image.open(img_path))
                    
                    # Apply enhancement
                    if method == 'clahe':
                        enhanced = self._apply_clahe(img)
                    elif method == 'gamma':
                        enhanced = self._apply_gamma_correction(img)
                    elif method == 'histogram_equalization':
                        enhanced = self._apply_histogram_equalization(img)
                    elif method == 'adaptive_gamma':
                        enhanced = self._apply_adaptive_gamma(img)
                    elif method == 'unsharp_mask':
                        enhanced = self._apply_unsharp_mask(img)
                    else:
                        enhanced = img
                    
                    # Save enhanced image
                    Image.fromarray(enhanced).save(split_dst / img_path.name)
            
            # Evaluate enhancement
            results[method] = self._evaluate_enhancement(method_dir)
        
        self.stats['enhancement_results'] = results
        
        # Create comparison visualization
        self._create_enhancement_comparison(methods)
        
        return results
    
    def _apply_clahe(self, img: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Ensure uint8 type
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(img)
    
    def _apply_gamma_correction(self, img: np.ndarray, gamma: float = None) -> np.ndarray:
        """Apply gamma correction with automatic gamma estimation"""
        if gamma is None:
            # Estimate gamma based on image statistics
            mean_intensity = img.mean() / 255.0
            gamma = np.log(0.5) / np.log(mean_intensity) if mean_intensity > 0 else 1.0
            gamma = np.clip(gamma, 0.5, 2.0)
        
        img_float = img.astype(np.float32) / 255.0
        corrected = np.power(img_float, gamma)
        return (corrected * 255).astype(np.uint8)
    
    def _apply_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """Apply global histogram equalization"""
        # Ensure uint8 type
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.equalizeHist(img)
    
    def _apply_adaptive_gamma(self, img: np.ndarray, target_mean: float = 140.0) -> np.ndarray:
        """Apply adaptive gamma correction to reach target mean
        
        Special handling for sparse microscopy data with bright spots
        """
        # For sparse microscopy data, use log transformation first
        img_float = img.astype(np.float32)
        
        # Add small epsilon to avoid log(0)
        img_log = np.log1p(img_float)
        
        # Normalize log-transformed image
        img_log_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min() + 1e-8)
        
        # Apply gamma correction on log-transformed data
        current_mean = img_log_norm.mean() * 255
        
        if current_mean > 0:
            # Estimate gamma to reach target
            gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
            gamma = np.clip(gamma, 0.3, 3.0)
        else:
            gamma = 1.0
        
        # Apply gamma
        img_gamma = np.power(img_log_norm, gamma)
        
        # Convert back to uint8
        return (img_gamma * 255).astype(np.uint8)
    
    def _apply_unsharp_mask(self, img: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
        """Apply unsharp masking for edge enhancement"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), radius)
        
        # Calculate the mask
        mask = img.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply the mask
        sharpened = img.astype(np.float32) + amount * mask
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # ==================== Analysis Methods ====================
    
    def analyze_dataset(self, comprehensive: bool = True) -> Dict:
        """Perform comprehensive dataset analysis"""
        print("\n📊 Analyzing dataset...")
        
        analyses = {}
        
        # 1. Basic statistics
        analyses['statistics'] = self._analyze_statistics()
        
        # 2. Distribution analysis
        analyses['distributions'] = self._analyze_distributions()
        
        # 3. Texture analysis
        if comprehensive:
            analyses['texture'] = self._analyze_texture_properties()
        
        # 4. Frequency analysis
        if comprehensive:
            analyses['frequency'] = self._analyze_frequency_content()
        
        # 5. Quality metrics
        analyses['quality'] = self._analyze_image_quality()
        
        # 6. Normalization diagnostics
        analyses['normalization'] = self._diagnose_normalization()
        
        self.stats['analysis_results'] = analyses
        
        # Generate report
        self._generate_analysis_report(analyses)
        
        return analyses
    
    def _analyze_statistics(self) -> Dict:
        """Analyze basic statistics across splits"""
        print("  - Computing basic statistics...")
        
        split_stats = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dirs[split]
            if not split_dir.exists():
                continue
            
            image_paths = list(split_dir.glob('*.png'))
            if not image_paths:
                continue
            
            # Sample statistics
            means, stds, mins, maxs = [], [], [], []
            
            for img_path in image_paths[:50]:  # Sample first 50
                img = np.array(Image.open(img_path))
                means.append(img.mean())
                stds.append(img.std())
                mins.append(img.min())
                maxs.append(img.max())
            
            split_stats[split] = {
                'count': len(image_paths),
                'mean_intensity': f"{np.mean(means):.2f} ± {np.std(means):.2f}",
                'intensity_range': f"[{int(np.mean(mins))}, {int(np.mean(maxs))}]"
            }
        
        return split_stats
    
    def _analyze_distributions(self) -> Dict:
        """Analyze pixel intensity distributions"""
        print("  - Analyzing intensity distributions...")
        
        # Sample images from train set
        train_dir = self.dirs['train']
        if not train_dir.exists():
            return {}
        
        sample_images = list(train_dir.glob('*.png'))[:50]
        all_pixels = []
        
        for img_path in sample_images:
            img = np.array(Image.open(img_path))
            all_pixels.extend(img.flatten())
        
        all_pixels = np.array(all_pixels)
        
        # Compute distribution statistics
        distribution_stats = {
            'skewness': stats.skew(all_pixels),
            'kurtosis': stats.kurtosis(all_pixels),
            'normality_test_pvalue': stats.normaltest(all_pixels)[1]
        }
        
        # Create histogram
        self._create_distribution_plot(all_pixels)
        
        return distribution_stats
    
    def _analyze_texture_properties(self) -> Dict:
        """Analyze texture properties relevant for microscopy"""
        print("  - Analyzing texture properties...")
        
        # This is a placeholder - implement actual texture analysis
        # Could include: Haralick features, LBP, etc.
        
        return {
            'texture_analysis': 'Not implemented in this version'
        }
    
    def _analyze_frequency_content(self) -> Dict:
        """Analyze frequency domain properties"""
        print("  - Analyzing frequency content...")
        
        # Sample analysis on a few images
        train_dir = self.dirs['train']
        if not train_dir.exists():
            return {}
        
        sample_images = list(train_dir.glob('*.png'))[:10]
        freq_metrics = []
        
        for img_path in sample_images:
            img = np.array(Image.open(img_path))
            
            # FFT
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Analyze frequency distribution
            h, w = img.shape
            center_h, center_w = h // 2, w // 2
            
            # Compute energy in different frequency bands
            total_energy = np.sum(magnitude_spectrum ** 2)
            
            # Low frequency (center region)
            low_freq_mask = np.zeros_like(magnitude_spectrum, dtype=bool)
            cv2.circle(low_freq_mask, (center_w, center_h), min(h, w) // 8, True, -1)
            low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask]) / total_energy
            
            # High frequency (outer region)
            high_freq_mask = ~low_freq_mask
            high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]) / total_energy
            
            freq_metrics.append({
                'low_freq_ratio': low_freq_energy,
                'high_freq_ratio': high_freq_energy
            })
        
        freq_df = pd.DataFrame(freq_metrics)
        
        return {
            'low_freq_dominance': freq_df['low_freq_ratio'].mean(),
            'high_freq_content': freq_df['high_freq_ratio'].mean()
        }
    
    def _analyze_image_quality(self) -> Dict:
        """Analyze image quality metrics"""
        print("  - Analyzing image quality...")
        
        quality_metrics = {
            'sharpness_scores': [],
            'noise_estimates': [],
            'dynamic_range_scores': []
        }
        
        sample_images = list(self.dirs['train'].glob('*.png'))[:50]
        
        for img_path in sample_images:
            img = np.array(Image.open(img_path))
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness = laplacian.var()
            quality_metrics['sharpness_scores'].append(sharpness)
            
            # Noise estimation (using median filter)
            denoised = cv2.medianBlur(img, 5)
            noise = np.std(img.astype(float) - denoised.astype(float))
            quality_metrics['noise_estimates'].append(noise)
            
            # Dynamic range
            dr = (img.max() - img.min()) / 255.0
            quality_metrics['dynamic_range_scores'].append(dr)
        
        return {
            'mean_sharpness': np.mean(quality_metrics['sharpness_scores']),
            'mean_noise': np.mean(quality_metrics['noise_estimates']),
            'mean_dynamic_range': np.mean(quality_metrics['dynamic_range_scores'])
        }
    
    def _diagnose_normalization(self) -> Dict:
        """Diagnose normalization issues in the pipeline"""
        print("  - Diagnosing normalization...")
        
        # Check a sample image through the pipeline
        sample_path = list(self.find_images())[0]
        
        # Load raw
        raw_img = np.array(Image.open(sample_path))
        raw_stats = {'mean': raw_img.mean(), 'std': raw_img.std(), 'dtype': str(raw_img.dtype)}
        
        # Check processed version if exists
        processed_stats = {}
        for split in ['train', 'val', 'test']:
            processed_path = self.dirs[split] / f"{sample_path.stem}.png"
            if processed_path.exists():
                proc_img = np.array(Image.open(processed_path))
                processed_stats = {
                    'mean': proc_img.mean(),
                    'std': proc_img.std(),
                    'dtype': str(proc_img.dtype)
                }
                break
        
        # Normalization diagnosis
        diagnosis = {
            'raw_stats': raw_stats,
            'processed_stats': processed_stats,
            'normalization_ok': processed_stats.get('dtype') == 'uint8' and 
                               50 < processed_stats.get('mean', 0) < 200 if processed_stats else False
        }
        
        return diagnosis
    
    # ==================== Visualization Methods ====================
    
    def _create_bit_depth_visualization(self, metrics: Dict):
        """Create bit depth analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('16-bit to 8-bit Conversion Analysis', fontsize=16)
        
        # SSIM distribution
        axes[0, 0].hist(metrics['ssim_scores'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(np.mean(metrics['ssim_scores']), color='red', linestyle='--',
                           label=f'Mean: {np.mean(metrics["ssim_scores"]):.3f}')
        axes[0, 0].set_xlabel('SSIM Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('SSIM Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Information loss
        axes[0, 1].hist(metrics['info_loss_percent'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(metrics['info_loss_percent']), color='red', linestyle='--',
                           label=f'Mean: {np.mean(metrics["info_loss_percent"]):.1f}%')
        axes[0, 1].set_xlabel('Information Loss (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Information Loss Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PSNR distribution
        axes[1, 0].hist(metrics['psnr_scores'], bins=20, alpha=0.7, color='green')
        axes[1, 0].axvline(np.mean(metrics['psnr_scores']), color='red', linestyle='--',
                           label=f'Mean: {np.mean(metrics["psnr_scores"]):.1f} dB')
        axes[1, 0].set_xlabel('PSNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('PSNR Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Dynamic range comparison
        axes[1, 1].scatter(metrics['dynamic_range_16bit'], metrics['dynamic_range_8bit'], alpha=0.5)
        axes[1, 1].plot([0, max(metrics['dynamic_range_16bit'])], 
                        [0, max(metrics['dynamic_range_16bit']) * 255/65535], 
                        'r--', label='Expected scaling')
        axes[1, 1].set_xlabel('16-bit Dynamic Range')
        axes[1, 1].set_ylabel('8-bit Dynamic Range')
        axes[1, 1].set_title('Dynamic Range Preservation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'bit_depth_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plot(self, pixel_values: np.ndarray):
        """Create pixel intensity distribution plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(pixel_values, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Density')
        ax1.set_title('Pixel Intensity Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Log-scale histogram (useful for sparse data)
        ax2.hist(pixel_values, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
        ax2.set_yscale('log')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Log Density')
        ax2.set_title('Pixel Intensity Distribution (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'intensity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_enhancement_comparison(self, methods: List[str]):
        """Create visual comparison of enhancement methods"""
        # Sample a few images
        sample_images = list(self.dirs['train'].glob('*.png'))[:3]
        
        if not sample_images:
            return
        
        n_methods = len(methods) + 1  # +1 for original
        n_samples = len(sample_images)
        
        fig, axes = plt.subplots(n_samples, n_methods, figsize=(4 * n_methods, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(sample_images):
            # Original
            orig_img = np.array(Image.open(img_path))
            axes[i, 0].imshow(orig_img, cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Enhanced versions
            for j, method in enumerate(methods):
                enhanced_path = self.dirs['enhanced'] / method / 'train' / img_path.name
                if enhanced_path.exists():
                    enh_img = np.array(Image.open(enhanced_path))
                    axes[i, j+1].imshow(enh_img, cmap='gray')
                    axes[i, j+1].set_title(method.replace('_', ' ').title())
                    axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'enhancement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _evaluate_enhancement(self, enhanced_dir: Path) -> Dict:
        """Evaluate enhancement quality"""
        # Simple evaluation based on contrast improvement
        original_contrast = []
        enhanced_contrast = []
        
        for img_path in list(self.dirs['train'].glob('*.png'))[:20]:
            orig_img = np.array(Image.open(img_path))
            enh_path = enhanced_dir / 'train' / img_path.name
            
            if enh_path.exists():
                enh_img = np.array(Image.open(enh_path))
                
                # Measure contrast as std dev
                original_contrast.append(orig_img.std())
                enhanced_contrast.append(enh_img.std())
        
        if original_contrast:
            contrast_improvement = np.mean(enhanced_contrast) / np.mean(original_contrast)
            mean_shift = np.mean(enhanced_contrast) - np.mean(original_contrast)
            
            return {
                'contrast_improvement': contrast_improvement,
                'mean_shift': mean_shift,
                'std_change': np.mean(enhanced_contrast) - np.mean(original_contrast)
            }
        
        return {}
    
    # ==================== Report Generation ====================
    
    def _generate_analysis_report(self, analyses: Dict):
        """Generate comprehensive analysis report"""
        report_path = self.dirs['reports'] / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# CARS Data Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n\n")
            if 'statistics' in analyses:
                for split, stats in analyses['statistics'].items():
                    f.write(f"### {split.title()} Split\n")
                    for key, value in stats.items():
                        f.write(f"- {key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
            
            # Distribution analysis
            if 'distributions' in analyses:
                f.write("## Distribution Analysis\n\n")
                dist = analyses['distributions']
                f.write(f"- Skewness: {dist.get('skewness', 'N/A'):.3f}\n")
                f.write(f"- Kurtosis: {dist.get('kurtosis', 'N/A'):.3f}\n")
                f.write(f"- Normality test p-value: {dist.get('normality_test_pvalue', 'N/A'):.4f}\n\n")
            
            # Image quality
            if 'quality' in analyses:
                f.write("## Image Quality Metrics\n\n")
                quality = analyses['quality']
                f.write(f"- Mean sharpness: {quality.get('mean_sharpness', 'N/A'):.2f}\n")
                f.write(f"- Mean noise level: {quality.get('mean_noise', 'N/A'):.2f}\n")
                f.write(f"- Mean dynamic range: {quality.get('mean_dynamic_range', 'N/A'):.2%}\n\n")
            
            # Normalization diagnostics
            if 'normalization' in analyses:
                normalization = analyses['normalization']
                f.write("## Normalization Diagnostics\n\n")
                f.write(f"- Raw data: {normalization['raw_stats']}\n")
                f.write(f"- Processed data: {normalization['processed_stats']}\n")
                f.write(f"- Normalization OK: {'✅' if normalization['normalization_ok'] else '❌'}\n\n")
                
                # Microscopy-specific warnings
                if normalization['raw_stats'].get('dtype') == 'uint16':
                    f.write("⚠️ **Warning**: Raw data is 16-bit. For non-linear optical microscopy:\n")
                    f.write("  - Use percentile normalization (0.1, 99.9) to preserve sparse bright features\n")
                    f.write("  - Consider log transformation for high dynamic range\n")
                    f.write("  - Use 'minmax' normalization instead of 'standard' in dataset config\n\n")
            
            # Enhancement results
            if 'enhancement_results' in self.stats and self.stats['enhancement_results']:
                f.write("## Enhancement Results\n\n")
                for method, results in self.stats['enhancement_results'].items():
                    f.write(f"### {method.replace('_', ' ').title()}\n")
                    f.write(f"- Mean shift: {results.get('mean_shift', 'N/A'):.2f}\n")
                    f.write(f"- Std change: {results.get('std_change', 'N/A'):.2f}\n")
                    f.write(f"- Contrast improvement: {results.get('contrast_improvement', 'N/A'):.2f}x\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Bit depth recommendation
            if 'bit_depth' in self.stats['analysis_results']:
                bit_depth = self.stats['analysis_results']['bit_depth']
                f.write(f"- **Bit depth**: {'Keep 16-bit' if bit_depth['recommendation'] == 'keep_16bit' else 'Use 8-bit'}\n")
                f.write(f"  - SSIM: {bit_depth['ssim_mean']:.3f}\n")
                f.write(f"  - Information loss: {bit_depth['info_loss_mean']:.1f}%\n\n")
            
            # Enhancement recommendation
            if self.stats.get('enhancement_results'):
                best_method = max(self.stats['enhancement_results'].items(), 
                                key=lambda x: x[1].get('contrast_improvement', 0))
                f.write(f"- **Best enhancement method**: {best_method[0]}\n")
                f.write(f"  - Contrast improvement: {best_method[1].get('contrast_improvement', 1):.2f}x\n\n")
            
            # Data quality assessment
            if 'quality' in analyses:
                quality = analyses['quality']
                if quality.get('mean_sharpness', 0) < 100:
                    f.write("- ⚠️ Low sharpness detected - consider unsharp masking\n")
                if quality.get('mean_noise', 0) > 10:
                    f.write("- ⚠️ High noise levels - consider denoising\n")
                if quality.get('mean_dynamic_range', 0) < 0.5:
                    f.write("- ⚠️ Low dynamic range - enhancement recommended\n")
        
        print(f"\n✅ Analysis report saved: {report_path}")
    
    def create_config_file(self):
        """Create Hydra configuration file for the processed dataset"""
        config = {
            'dataset_name': 'cars_microscopy_processed',
            'data_path': str(self.output_path.resolve()),
            'image_size': 512,
            'channels': 1,
            'use_8bit': True,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'split_seed': 42,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': True,
            'augment_train': True,
            'augment_val': False,
            'augment_test': False,
            'normalize_method': 'minmax',  # Changed from 'standard' for microscopy
            'normalize_to_unit_range': True,
            'mean': 0.5,
            'std': 0.5,
            'validate_images': True,
            'use_cache': False,
            'cache_dir': str(self.output_path / 'cache'),
            'augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotation_degrees': 15,
                'scale_range': [0.9, 1.1],
                'translate_percent': 0.1,
                'brightness_factor': 0.05,  # Reduced for microscopy
                'contrast_factor': 0.05,    # Reduced for microscopy
                'gamma_range': [0.9, 1.1],  # Narrower range for microscopy
                'gaussian_noise_std': 0.01,  # Reduced for microscopy
                'gaussian_blur_sigma': [0.1, 0.5],  # Reduced for microscopy
                'gaussian_blur_prob': 0.2,  # Reduced probability
                'elastic_transform': True,
                'elastic_alpha': 100,
                'elastic_sigma': 10,
                'elastic_prob': 0.3
            }
        }
        
        config_path = self.output_path / 'dataset_config.yaml'
        
        # Write YAML manually to avoid dependencies
        yaml_content = "# CARS Dataset Configuration\n"
        yaml_content += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        def dict_to_yaml(d, indent=0):
            yaml_str = ""
            for key, value in d.items():
                spaces = "  " * indent
                if isinstance(value, dict):
                    yaml_str += f"{spaces}{key}:\n"
                    yaml_str += dict_to_yaml(value, indent + 1)
                elif isinstance(value, list):
                    yaml_str += f"{spaces}{key}: {value}\n"
                elif isinstance(value, str):
                    yaml_str += f"{spaces}{key}: \"{value}\"\n"
                else:
                    yaml_str += f"{spaces}{key}: {value}\n"
            return yaml_str
        
        yaml_content += dict_to_yaml(config)
        
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Configuration saved: {config_path}")
        
        return config_path
    
    def run_full_pipeline(self, 
                         use_8bit: bool = True,
                         enhancement_methods: Union[str, List[str]] = 'all',
                         comprehensive_analysis: bool = True,
                         percentile_low: float = 0.1,
                         percentile_high: float = 99.9,
                         use_log_transform: bool = False):
        """Run the complete data pipeline
        
        Args:
            use_8bit: Convert to 8-bit format
            enhancement_methods: Enhancement methods to apply
            comprehensive_analysis: Run comprehensive analysis
            percentile_low: Lower percentile for normalization
            percentile_high: Upper percentile for normalization
            use_log_transform: Apply log transformation before normalization
        """
        print("🚀 Running full CARS data pipeline...\n")
        
        # Step 0: Check and fix any existing uint16 images
        self.fix_existing_uint16_images(percentile_low, percentile_high, use_log_transform)
        
        # Step 1: Bit depth analysis
        bit_depth_analysis = self.analyze_bit_depth_impact()
        
        # Use recommendation if not specified
        if use_8bit is None:
            use_8bit = bit_depth_analysis['recommendation'] == 'use_8bit'
        
        # Step 2: Prepare dataset
        dataset_stats = self.prepare_dataset(
            use_8bit=use_8bit,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            use_log_transform=use_log_transform
        )
        
        # Step 3: Enhance dataset (using microscopy-appropriate methods)
        if enhancement_methods == 'all':
            # For microscopy, prioritize CLAHE and adaptive gamma
            enhancement_methods = ['clahe', 'adaptive_gamma', 'histogram_equalization']
        enhancement_results = self.enhance_dataset(methods=enhancement_methods)
        
        # Step 4: Comprehensive analysis
        analysis_results = self.analyze_dataset(comprehensive=comprehensive_analysis)
        
        # Step 5: Create configuration
        config_path = self.create_config_file()
        
        # Step 6: Verify normalization
        diagnosis = self._diagnose_normalization()
        if diagnosis['normalization_ok']:
            print("✅ Normalization verification passed")
        else:
            print("⚠️  Normalization issues detected:")
            print(f"   Raw: {diagnosis['raw_stats']}")
            print(f"   Processed: {diagnosis['processed_stats']}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ CARS Data Pipeline Complete!")
        print(f"📁 Output directory: {self.output_path}")
        print(f"📊 Processed images: {self.stats['processed_count']}")
        print(f"❌ Failed images: {self.stats['failed_count']}")
        print(f"📋 Configuration: {config_path}")
        print(f"📈 Reports: {self.dirs['reports']}")
        print("=" * 60)
        
        return {
            'output_path': self.output_path,
            'stats': self.stats,
            'config_path': config_path
        }


def main():
    """Main function for the CARS data pipeline"""
    parser = argparse.ArgumentParser(
        description='CARS Data Pipeline - Comprehensive data preparation, enhancement, and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input data directory')
    parser.add_argument('--output_path', type=str, default='data/processed',
                       help='Output directory for processed data')
    
    # Task selection
    parser.add_argument('--task', type=str, 
                       choices=['prepare', 'enhance', 'analyze', 'full', 'fix'],
                       default='full',
                       help='Task to perform (default: full)')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Run full pipeline (equivalent to --task full)')
    
    # Data preparation options
    parser.add_argument('--use_8bit', type=bool, default=None,
                       help='Convert to 8-bit (None=auto based on analysis)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splits')
    
    # Percentile normalization options
    parser.add_argument('--percentile_low', type=float, default=0.1,
                       help='Lower percentile for normalization (default: 0.1, try 1.0 or 5.0 for sparse data)')
    parser.add_argument('--percentile_high', type=float, default=99.9,
                       help='Upper percentile for normalization (default: 99.9, try 99.0 or 95.0 for sparse data)')
    parser.add_argument('--use_log_transform', action='store_true',
                       help='Apply log transformation before normalization (recommended for sparse microscopy data)')
    
    # Enhancement options
    parser.add_argument('--enhancement_methods', nargs='+', 
                       default='all',
                       help='Enhancement methods to apply (default: all)')
    
    # Analysis options
    parser.add_argument('--comprehensive_analysis', action='store_true',
                       help='Perform comprehensive analysis (slower but more detailed)')
    
    # Bit depth analysis
    parser.add_argument('--analyze_bit_depth', action='store_true',
                       help='Analyze bit depth conversion impact')
    parser.add_argument('--bit_depth_samples', type=int, default=50,
                       help='Number of samples for bit depth analysis')
    
    args = parser.parse_args()
    
    # Handle full_pipeline flag
    if args.full_pipeline:
        args.task = 'full'
    
    # Create pipeline
    pipeline = CARSDataPipeline(args.input_path, args.output_path)
    
    try:
        if args.task == 'full':
            # Run complete pipeline
            pipeline.run_full_pipeline(
                use_8bit=args.use_8bit,
                enhancement_methods=args.enhancement_methods,
                comprehensive_analysis=args.comprehensive_analysis,
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high,
                use_log_transform=args.use_log_transform
            )
            
        elif args.task == 'prepare':
            # Just prepare data
            if args.analyze_bit_depth or args.use_8bit is None:
                bit_depth_analysis = pipeline.analyze_bit_depth_impact(args.bit_depth_samples)
                if args.use_8bit is None:
                    args.use_8bit = bit_depth_analysis['recommendation'] == 'use_8bit'
            
            pipeline.prepare_dataset(
                use_8bit=args.use_8bit,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high,
                use_log_transform=args.use_log_transform
            )
            pipeline.create_config_file()
            
        elif args.task == 'enhance':
            # Enhance existing data
            if not any(pipeline.dirs[split].exists() for split in ['train', 'val', 'test']):
                print("❌ No processed data found. Run with --task prepare first.")
                return
            
            pipeline.enhance_dataset(methods=args.enhancement_methods)
            
        elif args.task == 'analyze':
            # Analyze existing data
            if not any(pipeline.dirs[split].exists() for split in ['train', 'val', 'test']):
                print("❌ No processed data found. Run with --task prepare first.")
                return
            
            pipeline.analyze_dataset(comprehensive=args.comprehensive_analysis)
            
        elif args.task == 'fix':
            # Fix existing processed data
            if not any(pipeline.dirs[split].exists() for split in ['train', 'val', 'test']):
                print("❌ No processed data found to fix.")
                return
            
            print("🔧 Fixing existing processed data...")
            fixed = pipeline.fix_existing_uint16_images(
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high,
                use_log_transform=args.use_log_transform
            )
            
            if fixed:
                # Run analysis on fixed data
                pipeline.analyze_dataset(comprehensive=False)
                
                # Update config
                pipeline.create_config_file()
                
                print("\n✅ Data fixed! Please retrain your model with the corrected data.")
            else:
                print("\n✅ No fixes needed - data is already in correct format.")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()