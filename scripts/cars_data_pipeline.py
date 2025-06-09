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
                       seed: int = 42) -> Dict:
        """Prepare dataset with proper conversion and splitting"""
        print("\n🔄 Preparing dataset...")
        
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
                        # Use percentile-based conversion
                        p_low, p_high = np.percentile(img, (0.1, 99.9))
                        img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
                        img = (img * 255).astype(np.uint8)
                    elif img.dtype != np.uint8 and use_8bit:
                        img = img.astype(np.uint8)
                    
                    # Ensure grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    
                    # Save processed image
                    output_path = self.dirs[split_name] / f"{img_path.stem}.png"
                    Image.fromarray(img).save(output_path)
                    
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
        """Apply adaptive gamma correction to reach target mean"""
        current_mean = img.mean()
        
        # Iterative gamma adjustment
        gamma = 1.0
        best_gamma = 1.0
        best_diff = abs(current_mean - target_mean)
        
        for g in np.linspace(0.3, 3.0, 50):
            test_img = self._apply_gamma_correction(img, g)
            diff = abs(test_img.mean() - target_mean)
            
            if diff < best_diff:
                best_diff = diff
                best_gamma = g
        
        return self._apply_gamma_correction(img, best_gamma)
    
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
        
        self.stats['analysis_results'].update(analyses)
        
        # Generate comprehensive report
        self._generate_analysis_report(analyses)
        
        return analyses
    
    def _analyze_statistics(self) -> Dict:
        """Analyze basic image statistics"""
        print("  - Computing basic statistics...")
        
        stats_by_split = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dirs[split]
            if not split_dir.exists():
                continue
            
            images = list(split_dir.glob('*.png'))
            if not images:
                continue
            
            split_stats = []
            
            for img_path in tqdm(images[:100], desc=f"Analyzing {split}", leave=False):
                img = np.array(Image.open(img_path))
                
                split_stats.append({
                    'mean': img.mean(),
                    'std': img.std(),
                    'min': img.min(),
                    'max': img.max(),
                    'median': np.median(img),
                    'q25': np.percentile(img, 25),
                    'q75': np.percentile(img, 75)
                })
            
            stats_df = pd.DataFrame(split_stats)
            stats_by_split[split] = {
                'count': len(images),
                'mean': stats_df['mean'].mean(),
                'std': stats_df['std'].mean(),
                'range': [stats_df['min'].min(), stats_df['max'].max()]
            }
        
        return stats_by_split
    
    def _analyze_distributions(self) -> Dict:
        """Analyze pixel intensity distributions"""
        print("  - Analyzing distributions...")
        
        # Sample images from each split
        all_pixels = []
        split_pixels = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dirs[split]
            if not split_dir.exists():
                continue
            
            images = list(split_dir.glob('*.png'))[:20]
            split_data = []
            
            for img_path in images:
                img = np.array(Image.open(img_path))
                split_data.extend(img.flatten())
            
            if split_data:
                split_pixels[split] = split_data
                all_pixels.extend(split_data)
        
        # Compute distribution metrics
        if all_pixels:
            distribution_stats = {
                'skewness': stats.skew(all_pixels),
                'kurtosis': stats.kurtosis(all_pixels),
                'normality_test': stats.normaltest(all_pixels[:10000])[1]  # p-value
            }
        else:
            distribution_stats = {}
        
        # Create distribution plots
        self._create_distribution_plots(split_pixels)
        
        return distribution_stats
    
    def _analyze_texture_properties(self) -> Dict:
        """Analyze texture properties using various methods"""
        print("  - Analyzing texture properties...")
        
        texture_metrics = []
        sample_images = list(self.dirs['train'].glob('*.png'))[:50]
        
        for img_path in tqdm(sample_images, desc="Texture analysis", leave=False):
            img = np.array(Image.open(img_path))
            
            # Compute various texture features
            metrics = {
                'contrast': measure.shannon_entropy(img),
                'energy': np.sum(img ** 2) / img.size,
                'homogeneity': 1 / (1 + np.var(img)),
                'edge_density': np.sum(filters.sobel(img)) / img.size
            }
            
            texture_metrics.append(metrics)
        
        # Aggregate metrics
        texture_df = pd.DataFrame(texture_metrics)
        
        return {
            'contrast_mean': texture_df['contrast'].mean(),
            'energy_mean': texture_df['energy'].mean(),
            'homogeneity_mean': texture_df['homogeneity'].mean(),
            'edge_density_mean': texture_df['edge_density'].mean()
        }
    
    def _analyze_frequency_content(self) -> Dict:
        """Analyze frequency domain properties"""
        print("  - Analyzing frequency content...")
        
        freq_metrics = []
        sample_images = list(self.dirs['train'].glob('*.png'))[:30]
        
        for img_path in sample_images:
            img = np.array(Image.open(img_path))
            
            # Compute FFT
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Analyze frequency distribution
            h, w = img.shape
            center_h, center_w = h // 2, w // 2
            
            # Define frequency bands
            total_energy = np.sum(magnitude_spectrum)
            
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
            'normalization_ok': abs(processed_stats.get('mean', 0) - 127.5) < 50 if processed_stats else False
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
        axes[1, 0].set_xlabel('PSNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Dynamic range comparison
        dr_data = pd.DataFrame({
            '16-bit': metrics['dynamic_range_16bit'],
            '8-bit': metrics['dynamic_range_8bit']
        })
        dr_data.plot(kind='box', ax=axes[1, 1])
        axes[1, 1].set_ylabel('Dynamic Range')
        axes[1, 1].set_title('Dynamic Range Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'bit_depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_enhancement_comparison(self, methods: List[str]):
        """Create enhancement method comparison visualization"""
        # Get sample images
        sample_images = list(self.dirs['train'].glob('*.png'))[:3]
        
        if not sample_images:
            return
        
        fig, axes = plt.subplots(len(sample_images), len(methods) + 1, 
                                figsize=(4 * (len(methods) + 1), 4 * len(sample_images)))
        
        if len(sample_images) == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(sample_images):
            # Original
            img = np.array(Image.open(img_path))
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Original\nMean: {img.mean():.1f}')
            axes[i, 0].axis('off')
            
            # Enhanced versions
            for j, method in enumerate(methods):
                enhanced_path = self.dirs['enhanced'] / method / 'train' / img_path.name
                if enhanced_path.exists():
                    enh_img = np.array(Image.open(enhanced_path))
                    axes[i, j + 1].imshow(enh_img, cmap='gray')
                    axes[i, j + 1].set_title(f'{method}\nMean: {enh_img.mean():.1f}')
                    axes[i, j + 1].axis('off')
        
        plt.suptitle('Enhancement Methods Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'enhancement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plots(self, split_pixels: Dict):
        """Create pixel distribution plots"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
        
        # Combined histogram
        for split, pixels in split_pixels.items():
            if pixels:
                axes[0].hist(pixels[::100], bins=50, alpha=0.6, label=split, 
                           color=colors.get(split, 'gray'), density=True)
        
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Pixel Intensity Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        if split_pixels:
            box_data = [pixels[::1000] for pixels in split_pixels.values() if pixels]
            box_labels = [split for split, pixels in split_pixels.items() if pixels]
            axes[1].boxplot(box_data, labels=box_labels)
            axes[1].set_ylabel('Pixel Value')
            axes[1].set_title('Distribution Comparison')
            axes[1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        for split, pixels in split_pixels.items():
            if pixels:
                sorted_pixels = np.sort(pixels[::100])
                cdf = np.arange(1, len(sorted_pixels) + 1) / len(sorted_pixels)
                axes[2].plot(sorted_pixels, cdf, label=split, color=colors.get(split, 'gray'))
        
        axes[2].set_xlabel('Pixel Value')
        axes[2].set_ylabel('Cumulative Probability')
        axes[2].set_title('Cumulative Distribution Functions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['visualizations'] / 'pixel_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_enhancement(self, method_dir: Path) -> Dict:
        """Evaluate enhancement method effectiveness"""
        original_stats = []
        enhanced_stats = []
        
        # Compare original vs enhanced
        for img_path in list(self.dirs['train'].glob('*.png'))[:50]:
            enhanced_path = method_dir / 'train' / img_path.name
            
            if enhanced_path.exists():
                orig = np.array(Image.open(img_path))
                enh = np.array(Image.open(enhanced_path))
                
                original_stats.append({
                    'mean': orig.mean(),
                    'std': orig.std(),
                    'contrast': orig.std() / (orig.mean() + 1e-8)
                })
                
                enhanced_stats.append({
                    'mean': enh.mean(),
                    'std': enh.std(),
                    'contrast': enh.std() / (enh.mean() + 1e-8)
                })
        
        if not original_stats:
            return {}
        
        orig_df = pd.DataFrame(original_stats)
        enh_df = pd.DataFrame(enhanced_stats)
        
        return {
            'mean_shift': enh_df['mean'].mean() - orig_df['mean'].mean(),
            'std_change': enh_df['std'].mean() - orig_df['std'].mean(),
            'contrast_improvement': enh_df['contrast'].mean() / orig_df['contrast'].mean()
        }
    
    def _generate_analysis_report(self, analyses: Dict):
        """Generate comprehensive analysis report"""
        report_path = self.dirs['reports'] / f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        with open(report_path, 'w') as f:
            f.write("# CARS Data Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n\n")
            if 'statistics' in analyses:
                for split, stats in analyses['statistics'].items():
                    f.write(f"### {split.capitalize()} Split\n")
                    f.write(f"- Images: {stats['count']}\n")
                    f.write(f"- Mean intensity: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
                    f.write(f"- Range: [{stats['range'][0]}, {stats['range'][1]}]\n\n")
            
            # Distribution analysis
            if 'distributions' in analyses:
                f.write("## Distribution Analysis\n\n")
                dist = analyses['distributions']
                f.write(f"- Skewness: {dist.get('skewness', 'N/A'):.3f}\n")
                f.write(f"- Kurtosis: {dist.get('kurtosis', 'N/A'):.3f}\n")
                f.write(f"- Normality test p-value: {dist.get('normality_test', 'N/A'):.4f}\n\n")
            
            # Texture properties
            if 'texture' in analyses:
                f.write("## Texture Properties\n\n")
                texture = analyses['texture']
                for key, value in texture.items():
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
            
            # Frequency content
            if 'frequency' in analyses:
                f.write("## Frequency Analysis\n\n")
                freq = analyses['frequency']
                f.write(f"- Low frequency dominance: {freq.get('low_freq_dominance', 0):.2%}\n")
                f.write(f"- High frequency content: {freq.get('high_freq_content', 0):.2%}\n\n")
            
            # Quality metrics
            if 'quality' in analyses:
                f.write("## Image Quality Metrics\n\n")
                quality = analyses['quality']
                f.write(f"- Mean sharpness: {quality.get('mean_sharpness', 0):.2f}\n")
                f.write(f"- Mean noise level: {quality.get('mean_noise', 0):.2f}\n")
                f.write(f"- Mean dynamic range: {quality.get('mean_dynamic_range', 0):.2%}\n\n")
            
            # Normalization diagnostics
            if 'normalization' in analyses:
                f.write("## Normalization Diagnostics\n\n")
                norm = analyses['normalization']
                f.write(f"- Raw data: {norm.get('raw_stats', {})}\n")
                f.write(f"- Processed data: {norm.get('processed_stats', {})}\n")
                f.write(f"- Normalization OK: {'✅' if norm.get('normalization_ok', False) else '❌'}\n\n")
            
            # Enhancement results
            if self.stats.get('enhancement_results'):
                f.write("## Enhancement Results\n\n")
                for method, results in self.stats['enhancement_results'].items():
                    f.write(f"### {method.replace('_', ' ').title()}\n")
                    f.write(f"- Mean shift: {results.get('mean_shift', 0):.2f}\n")
                    f.write(f"- Std change: {results.get('std_change', 0):.2f}\n")
                    f.write(f"- Contrast improvement: {results.get('contrast_improvement', 1):.2f}x\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Bit depth recommendation
            if 'bit_depth' in self.stats.get('analysis_results', {}):
                bd = self.stats['analysis_results']['bit_depth']
                f.write(f"- **Bit depth**: {'Use 8-bit' if bd['recommendation'] == 'use_8bit' else 'Keep 16-bit'}\n")
                f.write(f"  - SSIM: {bd['ssim_mean']:.3f}\n")
                f.write(f"  - Information loss: {bd['info_loss_mean']:.1f}%\n\n")
            
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
            'normalize_method': 'standard',
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
                'brightness_factor': 0.1,
                'contrast_factor': 0.1,
                'gamma_range': [0.8, 1.2],
                'gaussian_noise_std': 0.02,
                'gaussian_blur_sigma': [0.1, 1.0],
                'gaussian_blur_prob': 0.3,
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
                         comprehensive_analysis: bool = True):
        """Run the complete data pipeline"""
        print("🚀 Running full CARS data pipeline...\n")
        
        # Step 1: Bit depth analysis
        bit_depth_analysis = self.analyze_bit_depth_impact()
        
        # Use recommendation if not specified
        if use_8bit is None:
            use_8bit = bit_depth_analysis['recommendation'] == 'use_8bit'
        
        # Step 2: Prepare dataset
        dataset_stats = self.prepare_dataset(use_8bit=use_8bit)
        
        # Step 3: Enhance dataset
        enhancement_results = self.enhance_dataset(methods=enhancement_methods)
        
        # Step 4: Comprehensive analysis
        analysis_results = self.analyze_dataset(comprehensive=comprehensive_analysis)
        
        # Step 5: Create configuration
        config_path = self.create_config_file()
        
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
                       choices=['prepare', 'enhance', 'analyze', 'full'],
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
                comprehensive_analysis=args.comprehensive_analysis
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
                seed=args.seed
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
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()