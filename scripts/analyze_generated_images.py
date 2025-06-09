"""
Generated Image Analysis & Debugging Script
Analyzes noise patterns, contrast issues, and quality problems in generated CARS images
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageEnhance
import cv2
from skimage import filters, feature, measure, exposure
from scipy import ndimage, signal
import pandas as pd
import argparse
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ImageAnalyzer:
    """Comprehensive image analysis for debugging generated CARS images"""
    
    def __init__(self, generated_dir: str, real_dir: str, output_dir: str = "outputs/image_analysis"):
        self.generated_dir = Path(generated_dir)
        self.real_dir = Path(real_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 CARS Image Analysis & Debugging")
        print(f"📁 Generated: {self.generated_dir}")
        print(f"📁 Real: {self.real_dir}")
        print(f"💾 Output: {self.output_dir}")
        print("=" * 60)
        
        # Load images
        self.generated_images = self._load_images(self.generated_dir, "generated")
        self.real_images = self._load_images(self.real_dir, "real")
        
        # Results storage
        self.analysis_results = {}
    
    def _load_images(self, image_dir: Path, image_type: str) -> List[np.ndarray]:
        """Load images from directory"""
        image_paths = []
        
        # Look for images in subdirectories too (for cancerous/healthy folders)
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_paths.extend(list(image_dir.glob(ext)))
            image_paths.extend(list(image_dir.rglob(ext)))  # Recursive search
        
        if not image_paths:
            print(f"⚠️  No images found in {image_dir}")
            return []
        
        images = []
        for img_path in image_paths[:20]:  # Limit to 20 images for analysis
            try:
                img = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"✅ Loaded {len(images)} {image_type} images")
        return images
    
    def analyze_dynamic_range(self) -> Dict:
        """Analyze dynamic range and intensity distributions"""
        print("\n📊 Analyzing dynamic range...")
        
        def get_stats(images, name):
            if not images:
                return {}
            
            all_pixels = np.concatenate([img.flatten() for img in images])
            return {
                f'{name}_min': float(np.min(all_pixels)),
                f'{name}_max': float(np.max(all_pixels)),
                f'{name}_mean': float(np.mean(all_pixels)),
                f'{name}_std': float(np.std(all_pixels)),
                f'{name}_median': float(np.median(all_pixels)),
                f'{name}_q25': float(np.percentile(all_pixels, 25)),
                f'{name}_q75': float(np.percentile(all_pixels, 75)),
                f'{name}_dynamic_range': float(np.max(all_pixels) - np.min(all_pixels))
            }
        
        real_stats = get_stats(self.real_images, 'real')
        gen_stats = get_stats(self.generated_images, 'generated')
        
        # Combine stats
        stats = {**real_stats, **gen_stats}
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dynamic Range Analysis', fontsize=16)
        
        if self.real_images and self.generated_images:
            # Histograms
            real_pixels = np.concatenate([img.flatten() for img in self.real_images])
            gen_pixels = np.concatenate([img.flatten() for img in self.generated_images])
            
            axes[0, 0].hist(real_pixels, bins=100, alpha=0.7, label='Real', density=True)
            axes[0, 0].hist(gen_pixels, bins=100, alpha=0.7, label='Generated', density=True)
            axes[0, 0].set_title('Pixel Intensity Histograms')
            axes[0, 0].set_xlabel('Pixel Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            
            # Cumulative histograms
            axes[0, 1].hist(real_pixels, bins=100, alpha=0.7, label='Real', density=True, cumulative=True)
            axes[0, 1].hist(gen_pixels, bins=100, alpha=0.7, label='Generated', density=True, cumulative=True)
            axes[0, 1].set_title('Cumulative Histograms')
            axes[0, 1].set_xlabel('Pixel Value')
            axes[0, 1].set_ylabel('Cumulative Density')
            axes[0, 1].legend()
            
            # Box plots
            data_for_box = [real_pixels[::1000], gen_pixels[::1000]]  # Sample for speed
            axes[0, 2].boxplot(data_for_box, labels=['Real', 'Generated'])
            axes[0, 2].set_title('Intensity Distribution')
            axes[0, 2].set_ylabel('Pixel Value')
            
            # Log histograms
            axes[1, 0].hist(real_pixels, bins=100, alpha=0.7, label='Real', density=True)
            axes[1, 0].hist(gen_pixels, bins=100, alpha=0.7, label='Generated', density=True)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_title('Log Scale Histograms')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Log Density')
            axes[1, 0].legend()
            
            # Quantile-quantile plot
            real_sorted = np.sort(real_pixels[::1000])
            gen_sorted = np.sort(gen_pixels[::1000])
            min_len = min(len(real_sorted), len(gen_sorted))
            axes[1, 1].scatter(real_sorted[:min_len], gen_sorted[:min_len], alpha=0.5)
            axes[1, 1].plot([0, 255], [0, 255], 'r--', label='Perfect Match')
            axes[1, 1].set_title('Q-Q Plot: Real vs Generated')
            axes[1, 1].set_xlabel('Real Quantiles')
            axes[1, 1].set_ylabel('Generated Quantiles')
            axes[1, 1].legend()
            
            # Dynamic range comparison
            ranges = [real_stats['real_dynamic_range'], gen_stats['generated_dynamic_range']]
            axes[1, 2].bar(['Real', 'Generated'], ranges, alpha=0.7)
            axes[1, 2].set_title('Dynamic Range Comparison')
            axes[1, 2].set_ylabel('Dynamic Range')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dynamic_range_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['dynamic_range'] = stats
        print(f"✅ Dynamic range analysis complete")
        
        return stats
    
    def detect_noise_patterns(self) -> Dict:
        """Detect and analyze noise patterns"""
        print("\n🔍 Detecting noise patterns...")
        
        def analyze_noise(images, name):
            if not images:
                return {}
            
            noise_metrics = {
                'high_freq_energy': [],
                'edge_density': [],
                'texture_uniformity': [],
                'checkerboard_score': [],
                'salt_pepper_score': [],
                'fft_checkerboard_score': [],  # New: FFT-based checkerboard detection
                'structural_complexity': [],   # New: Measure structural features
                'follicle_like_structures': [] # New: Detect circular/follicular patterns
            }
            
            for img in images[:10]:  # Analyze first 10 images
                # High frequency energy
                laplacian = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
                hf_energy = np.var(laplacian)
                noise_metrics['high_freq_energy'].append(hf_energy)
                
                # Edge density
                edges = feature.canny(img, sigma=1)
                edge_density = np.sum(edges) / edges.size
                noise_metrics['edge_density'].append(edge_density)
                
                # Texture uniformity (lower = more uniform)
                glcm = feature.graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)
                uniformity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                noise_metrics['texture_uniformity'].append(uniformity)
                
                # Checkerboard pattern detection
                kernel = np.array([[-1, 1, -1], [1, -4, 1], [-1, 1, -1]], dtype=np.float32)
                checkerboard_response = cv2.filter2D(img.astype(np.float32), -1, kernel)
                checkerboard_score = np.mean(np.abs(checkerboard_response))
                noise_metrics['checkerboard_score'].append(checkerboard_score)
                
                # Salt and pepper detection
                median_filtered = ndimage.median_filter(img, size=3)
                diff = np.abs(img.astype(np.float32) - median_filtered.astype(np.float32))
                salt_pepper_score = np.mean(diff)
                noise_metrics['salt_pepper_score'].append(salt_pepper_score)
                
                # NEW: FFT-based checkerboard detection
                f_transform = np.fft.fft2(img)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                
                # Look for checkerboard patterns in FFT (high frequencies at corners)
                h, w = img.shape
                corners = [
                    magnitude_spectrum[0:h//4, 0:w//4],      # Top-left
                    magnitude_spectrum[0:h//4, -w//4:],      # Top-right
                    magnitude_spectrum[-h//4:, 0:w//4],      # Bottom-left
                    magnitude_spectrum[-h//4:, -w//4:]       # Bottom-right
                ]
                corner_energy = sum(np.sum(corner) for corner in corners)
                center_energy = np.sum(magnitude_spectrum[h//4:-h//4, w//4:-w//4])
                fft_checkerboard = corner_energy / (center_energy + 1e-8)
                noise_metrics['fft_checkerboard_score'].append(fft_checkerboard)
                
                # NEW: Structural complexity (measure organized patterns)
                # Use local binary patterns to detect organized structures
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
                structural_complexity = -np.sum(lbp_hist * np.log(lbp_hist + 1e-8))  # Entropy
                noise_metrics['structural_complexity'].append(structural_complexity)
                
                # NEW: Follicle-like structure detection
                # Use Hough circles to detect circular structures
                img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                circles = cv2.HoughCircles(
                    img_8bit, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=50, param2=30, minRadius=5, maxRadius=50
                )
                follicle_count = len(circles[0]) if circles is not None else 0
                noise_metrics['follicle_like_structures'].append(follicle_count)
            
            # Calculate statistics
            stats = {}
            for metric, values in noise_metrics.items():
                if values:
                    stats[f'{name}_{metric}_mean'] = float(np.mean(values))
                    stats[f'{name}_{metric}_std'] = float(np.std(values))
            
            return stats
        
        real_noise = analyze_noise(self.real_images, 'real')
        gen_noise = analyze_noise(self.generated_images, 'generated')
        
        noise_stats = {**real_noise, **gen_noise}
        
        # Create noise analysis plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))  # Expanded for new metrics
        fig.suptitle('Comprehensive Noise & Structure Analysis', fontsize=16)
        
        if self.real_images and self.generated_images:
            metrics = [
                'high_freq_energy', 'edge_density', 'texture_uniformity', 
                'checkerboard_score', 'salt_pepper_score', 'fft_checkerboard_score',
                'structural_complexity', 'follicle_like_structures'
            ]
            
            for i, metric in enumerate(metrics):
                row = i // 3
                col = i % 3
                
                real_key = f'real_{metric}_mean'
                gen_key = f'generated_{metric}_mean'
                
                if real_key in noise_stats and gen_key in noise_stats:
                    values = [noise_stats[real_key], noise_stats[gen_key]]
                    labels = ['Real', 'Generated']
                    bars = axes[row, col].bar(labels, values, alpha=0.7)
                    
                    # Color code problematic metrics
                    if metric == 'fft_checkerboard_score' and noise_stats[gen_key] > noise_stats[real_key] * 2:
                        bars[1].set_color('red')  # Red for high checkerboard
                    elif metric == 'follicle_like_structures' and noise_stats[gen_key] < noise_stats[real_key] * 0.5:
                        bars[1].set_color('orange')  # Orange for missing structures
                    
                    axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                    axes[row, col].set_ylabel('Score')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                          f'{value:.2f}', ha='center', va='bottom')
        else:
            # If no comparison possible, just show message
            metrics = []
            for i in range(9):
                row = i // 3
                col = i % 3
                axes[row, col].text(0.5, 0.5, 'No comparison\navailable', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'Analysis {i+1}')
        
        # Remove empty subplot if needed
        if len(metrics) < 9:
            axes[2, 2].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['noise_patterns'] = noise_stats
        print(f"✅ Noise pattern analysis complete")
        
        return noise_stats
    
    def create_enhanced_visualizations(self) -> None:
        """Create enhanced contrast visualizations"""
        print("\n🎨 Creating enhanced visualizations...")
        
        def enhance_image(img, factor=3.0):
            """Enhance contrast of an image"""
            # Convert to PIL for enhancement
            pil_img = Image.fromarray(img)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(factor)
            return np.array(enhanced)
        
        def apply_clahe(img):
            """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(img)
        
        if not self.real_images or not self.generated_images:
            print("⚠️  Insufficient images for visualization")
            return
        
        # Select representative images
        real_sample = self.real_images[0] if self.real_images else np.zeros((512, 512))
        gen_sample = self.generated_images[0] if self.generated_images else np.zeros((512, 512))
        
        # Create multi-contrast visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('Multi-Contrast Analysis: Real vs Generated', fontsize=16)
        
        contrast_factors = [1.0, 2.0, 3.0, 5.0]
        
        for i, factor in enumerate(contrast_factors):
            # Real images
            real_enhanced = enhance_image(real_sample, factor)
            axes[i, 0].imshow(real_enhanced, cmap='gray')
            axes[i, 0].set_title(f'Real (Contrast {factor}x)')
            axes[i, 0].axis('off')
            
            # Generated images  
            gen_enhanced = enhance_image(gen_sample, factor)
            axes[i, 1].imshow(gen_enhanced, cmap='gray')
            axes[i, 1].set_title(f'Generated (Contrast {factor}x)')
            axes[i, 1].axis('off')
            
            # CLAHE on real
            real_clahe = apply_clahe(real_sample)
            axes[i, 2].imshow(real_clahe, cmap='gray')
            axes[i, 2].set_title(f'Real CLAHE')
            axes[i, 2].axis('off')
            
            # CLAHE on generated
            gen_clahe = apply_clahe(gen_sample)
            axes[i, 3].imshow(gen_clahe, cmap='gray')
            axes[i, 3].set_title(f'Generated CLAHE')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_contrast_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create high-contrast grid of multiple samples
        if len(self.generated_images) >= 16:
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig.suptitle('Generated Samples (5x Contrast Enhancement)', fontsize=16)
            
            for i in range(16):
                row, col = i // 4, i % 4
                enhanced = enhance_image(self.generated_images[i], 5.0)
                axes[row, col].imshow(enhanced, cmap='gray')
                axes[row, col].set_title(f'Sample {i+1}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'generated_samples_enhanced.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Enhanced visualizations created")
    
    def analyze_frequency_content(self) -> Dict:
        """Analyze frequency content and artifacts"""
        print("\n📊 Analyzing frequency content...")
        
        def get_frequency_stats(images, name):
            if not images:
                return {}
            
            freq_stats = {
                'low_freq_energy': [],
                'mid_freq_energy': [],
                'high_freq_energy': [],
                'freq_concentration': []
            }
            
            for img in images[:5]:  # Analyze first 5 images
                # FFT analysis
                f_transform = np.fft.fft2(img)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                
                # Define frequency bands
                h, w = img.shape
                center_h, center_w = h // 2, w // 2
                
                # Create masks for different frequency bands
                y, x = np.ogrid[:h, :w]
                
                # Low frequencies (center)
                low_mask = ((x - center_w)**2 + (y - center_h)**2) < (min(h, w) * 0.1)**2
                low_energy = np.sum(magnitude_spectrum[low_mask])
                
                # Mid frequencies
                mid_mask = (((x - center_w)**2 + (y - center_h)**2) >= (min(h, w) * 0.1)**2) & \
                          (((x - center_w)**2 + (y - center_h)**2) < (min(h, w) * 0.3)**2)
                mid_energy = np.sum(magnitude_spectrum[mid_mask])
                
                # High frequencies (edges)
                high_mask = ((x - center_w)**2 + (y - center_h)**2) >= (min(h, w) * 0.3)**2
                high_energy = np.sum(magnitude_spectrum[high_mask])
                
                total_energy = low_energy + mid_energy + high_energy
                
                freq_stats['low_freq_energy'].append(low_energy / total_energy)
                freq_stats['mid_freq_energy'].append(mid_energy / total_energy)
                freq_stats['high_freq_energy'].append(high_energy / total_energy)
                
                # Frequency concentration (how concentrated is the energy?)
                freq_concentration = np.sum(magnitude_spectrum**2) / (np.sum(magnitude_spectrum)**2)
                freq_stats['freq_concentration'].append(freq_concentration)
            
            # Calculate statistics
            stats = {}
            for metric, values in freq_stats.items():
                if values:
                    stats[f'{name}_{metric}_mean'] = float(np.mean(values))
                    stats[f'{name}_{metric}_std'] = float(np.std(values))
            
            return stats
        
        real_freq = get_frequency_stats(self.real_images, 'real')
        gen_freq = get_frequency_stats(self.generated_images, 'generated')
        
        freq_stats = {**real_freq, **gen_freq}
        
        # Create frequency analysis visualization
        if self.real_images and self.generated_images:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Frequency Domain Analysis', fontsize=16)
            
            for i, (images, name) in enumerate([(self.real_images, 'Real'), (self.generated_images, 'Generated')]):
                img = images[0]
                
                # Original image
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title(f'{name} Image')
                axes[i, 0].axis('off')
                
                # FFT magnitude
                f_transform = np.fft.fft2(img)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                
                axes[i, 1].imshow(magnitude_spectrum, cmap='hot')
                axes[i, 1].set_title(f'{name} FFT Magnitude')
                axes[i, 1].axis('off')
                
                # High-pass filtered
                h, w = img.shape
                center_h, center_w = h // 2, w // 2
                y, x = np.ogrid[:h, :w]
                high_pass_mask = ((x - center_w)**2 + (y - center_h)**2) >= (min(h, w) * 0.1)**2
                
                f_high = f_shift.copy()
                f_high[~high_pass_mask] = 0
                img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(f_high)))
                
                axes[i, 2].imshow(img_high, cmap='gray')
                axes[i, 2].set_title(f'{name} High Freq Only')
                axes[i, 2].axis('off')
                
                # Frequency band energy plot
                bands = ['Low', 'Mid', 'High']
                energies = [
                    freq_stats.get(f'{name.lower()}_low_freq_energy_mean', 0),
                    freq_stats.get(f'{name.lower()}_mid_freq_energy_mean', 0),
                    freq_stats.get(f'{name.lower()}_high_freq_energy_mean', 0)
                ]
                
                axes[i, 3].bar(bands, energies, alpha=0.7)
                axes[i, 3].set_title(f'{name} Frequency Energy')
                axes[i, 3].set_ylabel('Relative Energy')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'frequency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.analysis_results['frequency_content'] = freq_stats
        print(f"✅ Frequency analysis complete")
        
        return freq_stats
    
    def generate_diagnostic_report(self) -> Path:
        """Generate comprehensive diagnostic report"""
        print("\n📋 Generating diagnostic report...")
        
        report_path = self.output_dir / 'diagnostic_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# CARS-FASTGAN Generated Image Diagnostic Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Real Images**: {len(self.real_images)}\n")
            f.write(f"**Generated Images**: {len(self.generated_images)}\n\n")
            
            # Dynamic Range Analysis
            if 'dynamic_range' in self.analysis_results:
                dr = self.analysis_results['dynamic_range']
                f.write("## Dynamic Range Analysis\n\n")
                f.write("| Metric | Real | Generated | Issue |\n")
                f.write("|--------|------|-----------|-------|\n")
                
                real_range = dr.get('real_dynamic_range', 0)
                gen_range = dr.get('generated_dynamic_range', 0)
                f.write(f"| Dynamic Range | {real_range:.1f} | {gen_range:.1f} | {'⚠️ Low' if gen_range < real_range * 0.7 else '✅ OK'} |\n")
                
                real_mean = dr.get('real_mean', 0)
                gen_mean = dr.get('generated_mean', 0)
                f.write(f"| Mean Intensity | {real_mean:.1f} | {gen_mean:.1f} | {'⚠️ Mismatch' if abs(real_mean - gen_mean) > 20 else '✅ OK'} |\n")
                
                real_std = dr.get('real_std', 0)
                gen_std = dr.get('generated_std', 0)
                f.write(f"| Std Deviation | {real_std:.1f} | {gen_std:.1f} | {'⚠️ Low Contrast' if gen_std < real_std * 0.7 else '✅ OK'} |\n")
                f.write("\n")
            
            # Noise Analysis
            if 'noise_patterns' in self.analysis_results:
                noise = self.analysis_results['noise_patterns']
                f.write("## Noise Pattern Analysis\n\n")
                
                # FFT Checkerboard Analysis (NEW)
                real_fft_cb = noise.get('real_fft_checkerboard_score_mean', 0)
                gen_fft_cb = noise.get('generated_fft_checkerboard_score_mean', 0)
                f.write(f"- **FFT Checkerboard Score**: Real {real_fft_cb:.2f}, Generated {gen_fft_cb:.2f}\n")
                if gen_fft_cb > real_fft_cb * 2:
                    f.write("  - 🚨 **CRITICAL: Strong FFT checkerboard artifacts detected**\n")
                    f.write("  - This indicates upsampling/deconvolution artifacts\n")
                else:
                    f.write("  - ✅ Minimal FFT checkerboard artifacts\n")
                
                # Structural Complexity (NEW)
                real_struct = noise.get('real_structural_complexity_mean', 0)
                gen_struct = noise.get('generated_structural_complexity_mean', 0)
                f.write(f"- **Structural Complexity**: Real {real_struct:.2f}, Generated {gen_struct:.2f}\n")
                if gen_struct < real_struct * 0.7:
                    f.write("  - 🚨 **CRITICAL: Generated images lack structural complexity**\n")
                    f.write("  - Model is not learning organized tissue patterns\n")
                else:
                    f.write("  - ✅ Adequate structural complexity\n")
                
                # Follicular Structures (NEW)
                real_foll = noise.get('real_follicle_like_structures_mean', 0)
                gen_foll = noise.get('generated_follicle_like_structures_mean', 0)
                f.write(f"- **Follicle-like Structures**: Real {real_foll:.1f}, Generated {gen_foll:.1f}\n")
                if gen_foll < real_foll * 0.3:
                    f.write("  - 🚨 **CRITICAL: Missing follicular structures**\n")
                    f.write("  - Generated images lack characteristic thyroid follicles\n")
                else:
                    f.write("  - ✅ Adequate follicular structure detection\n")
                
                # Original metrics
                real_hf = noise.get('real_high_freq_energy_mean', 0)
                gen_hf = noise.get('generated_high_freq_energy_mean', 0)
                f.write(f"- **High Frequency Energy**: Real {real_hf:.2f}, Generated {gen_hf:.2f}\n")
                if gen_hf > real_hf * 1.5:
                    f.write("  - ⚠️ **Excessive high-frequency content suggests noise artifacts**\n")
                else:
                    f.write("  - ✅ High-frequency content within normal range\n")
                
                real_cb = noise.get('real_checkerboard_score_mean', 0)
                gen_cb = noise.get('generated_checkerboard_score_mean', 0)
                f.write(f"- **Spatial Checkerboard Score**: Real {real_cb:.2f}, Generated {gen_cb:.2f}\n")
                if gen_cb > real_cb * 2:
                    f.write("  - ⚠️ **Strong spatial checkerboard artifacts detected**\n")
                else:
                    f.write("  - ✅ Minimal spatial checkerboard artifacts\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Diagnostic Recommendations\n\n")
            
            # Check for common issues
            issues_found = []
            recommendations = []
            
            # Check dynamic range issues
            if 'dynamic_range' in self.analysis_results:
                dr = self.analysis_results['dynamic_range']
                if dr.get('generated_dynamic_range', 0) < dr.get('real_dynamic_range', 0) * 0.7:
                    issues_found.append("Low dynamic range")
                    recommendations.append("**Increase model capacity** - Use 'standard' instead of 'micro' model")
                    recommendations.append("**Adjust normalization** - Check input/output scaling")
                
                # Check for mean intensity mismatch (NEW)
                real_mean = dr.get('real_mean', 0)
                gen_mean = dr.get('generated_mean', 0)
                if abs(real_mean - gen_mean) > 50:
                    issues_found.append("Severe intensity mismatch")
                    recommendations.append("**CRITICAL: Fix output normalization** - Generated images too dark/bright")
                    recommendations.append("**Check tanh output scaling** - Ensure proper [-1,1] to [0,255] conversion")
                
                # Check for low contrast
                real_std = dr.get('real_std', 0)
                gen_std = dr.get('generated_std', 0)
                if gen_std < real_std * 0.7:
                    issues_found.append("Low contrast/compressed dynamic range")
                    recommendations.append("**Improve discriminator** - Current discriminator too weak")
                    recommendations.append("**Increase feature matching weight** - Force generator to match distributions")
            
            # Check structural issues (NEW)
            if 'noise_patterns' in self.analysis_results:
                noise = self.analysis_results['noise_patterns']
                
                # FFT checkerboard artifacts
                real_fft_cb = noise.get('real_fft_checkerboard_score_mean', 0)
                gen_fft_cb = noise.get('generated_fft_checkerboard_score_mean', 0)
                if gen_fft_cb > real_fft_cb * 2:
                    issues_found.append("FFT checkerboard artifacts")
                    recommendations.append("**CRITICAL: Fix upsampling layers** - Replace nearest neighbor with learned upsampling")
                    recommendations.append("**Add anti-aliasing filters** - Use blur before upsampling")
                    recommendations.append("**Use SubPixel convolution** - Replace transpose convolution")
                
                # Missing structural complexity
                real_struct = noise.get('real_structural_complexity_mean', 0)
                gen_struct = noise.get('generated_structural_complexity_mean', 0)
                if gen_struct < real_struct * 0.7:
                    issues_found.append("Missing structural complexity")
                    recommendations.append("**CRITICAL: Increase model capacity** - Micro model insufficient for tissue structure")
                    recommendations.append("**Use Progressive Growing** - Start small, gradually increase resolution")
                    recommendations.append("**Add self-attention layers** - Help model learn spatial relationships")
                
                # Missing follicular structures
                real_foll = noise.get('real_follicle_like_structures_mean', 0)
                gen_foll = noise.get('generated_follicle_like_structures_mean', 0)
                if real_foll > 0 and gen_foll < real_foll * 0.3:
                    issues_found.append("Missing follicular structures")
                    recommendations.append("**CRITICAL: Model not learning thyroid morphology**")
                    recommendations.append("**Increase discriminator receptive field** - Use larger kernels")
                    recommendations.append("**Add perceptual loss** - Help preserve structural features")
                    recommendations.append("**Consider StyleGAN2** - Better for structured medical images")
                
                # Original checks
                if noise.get('generated_checkerboard_score_mean', 0) > noise.get('real_checkerboard_score_mean', 0) * 2:
                    issues_found.append("Spatial checkerboard artifacts")
                    recommendations.append("**Fix spatial upsampling** - Use stride=2 conv transpose with proper padding")
                
                if noise.get('generated_high_freq_energy_mean', 0) > noise.get('real_high_freq_energy_mean', 0) * 1.5:
                    issues_found.append("Excessive high-frequency noise")
                    recommendations.append("**Lower learning rates** - Reduce generator LR to 0.0001")
                    recommendations.append("**Increase discriminator capacity** - Use larger discriminator")
                    recommendations.append("**Add spectral normalization** to generator")
            
            if not issues_found:
                f.write("✅ **No major issues detected**\n\n")
            else:
                f.write("### Issues Detected:\n")
                for issue in issues_found:
                    f.write(f"- ⚠️ {issue}\n")
                f.write("\n### Recommended Fixes:\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            f.write("## Next Training Run Suggestions\n\n")
            f.write("```bash\n")
            f.write("# Improved configuration\n")
            f.write("python scripts/launch_training.py \\\n")
            f.write("    --model_config standard \\\n")  # Upgrade from micro
            f.write("    --batch_size 16 \\\n")  # Smaller batch for stability
            f.write("    --max_epochs 800 \\\n")
            f.write("    --experiment_name cars_fastgan_improved\n")
            f.write("```\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `dynamic_range_analysis.png` - Intensity distribution comparison\n")
            f.write("- `noise_analysis.png` - Noise pattern metrics\n")
            f.write("- `enhanced_contrast_comparison.png` - High-contrast visualizations\n")
            f.write("- `generated_samples_enhanced.png` - Enhanced generated samples\n")
            f.write("- `frequency_analysis.png` - Frequency domain analysis\n")
        
        print(f"✅ Diagnostic report saved: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run complete diagnostic analysis"""
        print("🚀 Starting comprehensive image analysis...")
        
        # Run all analyses
        self.analyze_dynamic_range()
        self.detect_noise_patterns()
        self.create_enhanced_visualizations()
        self.analyze_frequency_content()
        
        # Generate report
        report_path = self.generate_diagnostic_report()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📋 Read report: {report_path}")
        
        return self.analysis_results


def main():
    parser = argparse.ArgumentParser(description='Analyze generated CARS images for debugging')
    parser.add_argument('--generated_dir', type=str, default='outputs/evaluation/samples',
                       help='Directory containing generated images')
    parser.add_argument('--real_dir', type=str, default='data/raw',
                       help='Directory containing real images')
    parser.add_argument('--output_dir', type=str, default='outputs/image_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = ImageAnalyzer(args.generated_dir, args.real_dir, args.output_dir)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()