#!/usr/bin/env python3
"""
CARS Microscopy Data Analysis Tool
Analyzes 16-bit to 8-bit conversion impact for FASTGAN training

Usage:
    python src/data/analyze.py --data_path /path/to/your/cars/images --output_dir ./outputs/analysis
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CARSDataAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.data_path.rglob(ext)))
        
        print(f"Found {len(self.image_paths)} images in {data_path}")
        
        # Store analysis results
        self.results = {}
        
    def analyze_bit_depth_impact(self, sample_size=100):
        """Analyze the impact of converting from 16-bit to 8-bit"""
        print(f"\n=== Analyzing 16-bit to 8-bit conversion impact ===")
        
        # Sample images for analysis
        sample_paths = np.random.choice(self.image_paths, 
                                      min(sample_size, len(self.image_paths)), 
                                      replace=False)
        
        metrics = {
            'ssim_scores': [],
            'mse_scores': [],
            'psnr_scores': [],
            'dynamic_range_16bit': [],
            'dynamic_range_8bit': [],
            'histogram_correlation': [],
            'unique_values_16bit': [],
            'unique_values_8bit': [],
            'info_loss_percent': []
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, img_path in enumerate(sample_paths[:6]):  # Show first 6 for visualization
            # Load 16-bit image
            img_16bit = np.array(Image.open(img_path))
            
            # Ensure it's actually 16-bit
            if img_16bit.dtype != np.uint16:
                print(f"Warning: {img_path.name} is not 16-bit, skipping...")
                continue
                
            # Convert to 8-bit using simple scaling
            img_8bit = (img_16bit / 65535.0 * 255).astype(np.uint8)
            
            # For comparison, convert both to 0-1 range to avoid confusion
            img_16bit_normalized = img_16bit.astype(np.float32) / 65535.0
            img_8bit_normalized = img_8bit.astype(np.float32) / 255.0
            
            # Calculate metrics in normalized space
            ssim_score = ssim(img_16bit_normalized, img_8bit_normalized, data_range=1.0)
            
            mse = np.mean((img_16bit_normalized - img_8bit_normalized) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # Dynamic range analysis (in original scales for interpretability)
            dr_16bit = np.max(img_16bit) - np.min(img_16bit)
            dr_8bit = np.max(img_8bit) - np.min(img_8bit)  # Keep in 8-bit scale
            
            # Histogram correlation (compare in normalized space)
            hist_16bit, _ = np.histogram(img_16bit_normalized, bins=256, range=(0, 1))
            hist_8bit, _ = np.histogram(img_8bit_normalized, bins=256, range=(0, 1))
            hist_corr = np.corrcoef(hist_16bit, hist_8bit)[0, 1]
            
            # Unique values
            unique_16bit = len(np.unique(img_16bit))
            unique_8bit = len(np.unique(img_8bit))
            info_loss = (1 - unique_8bit / unique_16bit) * 100
            
            # Store metrics
            metrics['ssim_scores'].append(ssim_score)
            metrics['mse_scores'].append(mse)
            metrics['psnr_scores'].append(psnr)
            metrics['dynamic_range_16bit'].append(dr_16bit)
            metrics['dynamic_range_8bit'].append(dr_8bit)
            metrics['histogram_correlation'].append(hist_corr)
            metrics['unique_values_16bit'].append(unique_16bit)
            metrics['unique_values_8bit'].append(unique_8bit)
            metrics['info_loss_percent'].append(info_loss)
            
            if i < 6:  # Visualize first 6
                # Plot comparison (both normalized to 0-1 for display)
                axes[i].imshow(np.hstack([
                    img_16bit_normalized,  # Already normalized 0-1
                    img_8bit_normalized   # Already normalized 0-1
                ]), cmap='gray')
                axes[i].set_title(f'Image {i+1}\n16-bit | 8-bit\nSSIM: {ssim_score:.3f}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bit_depth_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        summary_stats = pd.DataFrame({
            'Metric': ['SSIM', 'PSNR (dB)', 'Histogram Correlation', 'Information Loss (%)'],
            'Mean': [
                np.mean(metrics['ssim_scores']),
                np.mean(metrics['psnr_scores']),
                np.mean(metrics['histogram_correlation']),
                np.mean(metrics['info_loss_percent'])
            ],
            'Std': [
                np.std(metrics['ssim_scores']),
                np.std(metrics['psnr_scores']),
                np.std(metrics['histogram_correlation']),
                np.std(metrics['info_loss_percent'])
            ],
            'Min': [
                np.min(metrics['ssim_scores']),
                np.min(metrics['psnr_scores']),
                np.min(metrics['histogram_correlation']),
                np.min(metrics['info_loss_percent'])
            ],
            'Max': [
                np.max(metrics['ssim_scores']),
                np.max(metrics['psnr_scores']),
                np.max(metrics['histogram_correlation']),
                np.max(metrics['info_loss_percent'])
            ]
        })
        
        print("\n=== Bit Depth Conversion Analysis Results ===")
        print(summary_stats.round(4))
        
        # Save results
        summary_stats.to_csv(self.output_dir / 'bit_depth_analysis.csv', index=False)
        
        # Create detailed plots
        self._create_analysis_plots(metrics)
        
        # Store results
        self.results['bit_depth_analysis'] = {
            'summary_stats': summary_stats,
            'detailed_metrics': metrics,
            'recommendation': self._get_bit_depth_recommendation(metrics)
        }
        
        return self.results['bit_depth_analysis']
    
    def _create_analysis_plots(self, metrics):
        """Create detailed analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # SSIM distribution
        axes[0,0].hist(metrics['ssim_scores'], bins=20, alpha=0.7, color='blue')
        axes[0,0].axvline(np.mean(metrics['ssim_scores']), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(metrics["ssim_scores"]):.3f}')
        axes[0,0].set_xlabel('SSIM Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('SSIM Score Distribution\n(Higher is better, >0.9 is excellent)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Information loss
        axes[0,1].hist(metrics['info_loss_percent'], bins=20, alpha=0.7, color='orange')
        axes[0,1].axvline(np.mean(metrics['info_loss_percent']), color='red', linestyle='--',
                         label=f'Mean: {np.mean(metrics["info_loss_percent"]):.1f}%')
        axes[0,1].set_xlabel('Information Loss (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Information Loss Distribution\n(Lower is better)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Dynamic range comparison
        dr_comparison = pd.DataFrame({
            '16-bit Dynamic Range': metrics['dynamic_range_16bit'],
            '8-bit Dynamic Range': [x * (65535/255) for x in metrics['dynamic_range_8bit']]  # Scale to compare
        })
        dr_comparison.plot(kind='box', ax=axes[1,0])
        axes[1,0].set_ylabel('Dynamic Range (16-bit scale)')
        axes[1,0].set_title('Dynamic Range Comparison')
        axes[1,0].grid(True, alpha=0.3)
        
        # PSNR vs SSIM scatter
        axes[1,1].scatter(metrics['ssim_scores'], metrics['psnr_scores'], alpha=0.6)
        axes[1,1].set_xlabel('SSIM Score')
        axes[1,1].set_ylabel('PSNR (dB)')
        axes[1,1].set_title('PSNR vs SSIM Relationship')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_bit_depth_recommendation(self, metrics):
        """Generate recommendation based on analysis"""
        mean_ssim = np.mean(metrics['ssim_scores'])
        mean_info_loss = np.mean(metrics['info_loss_percent'])
        mean_hist_corr = np.mean(metrics['histogram_correlation'])
        
        recommendation = {
            'use_8bit': False,
            'confidence': 'low',
            'reasons': []
        }
        
        if mean_ssim > 0.95:
            recommendation['reasons'].append(f'Excellent SSIM preservation ({mean_ssim:.3f})')
            recommendation['use_8bit'] = True
            
        if mean_info_loss < 15:
            recommendation['reasons'].append(f'Low information loss ({mean_info_loss:.1f}%)')
            recommendation['use_8bit'] = True
            
        if mean_hist_corr > 0.9:
            recommendation['reasons'].append(f'Strong histogram correlation ({mean_hist_corr:.3f})')
            recommendation['use_8bit'] = True
            
        # Determine confidence
        positive_indicators = sum([
            mean_ssim > 0.95,
            mean_info_loss < 15,
            mean_hist_corr > 0.9
        ])
        
        if positive_indicators >= 2:
            recommendation['confidence'] = 'high'
        elif positive_indicators == 1:
            recommendation['confidence'] = 'medium'
        else:
            recommendation['confidence'] = 'low'
            recommendation['use_8bit'] = False
            recommendation['reasons'] = [
                f'Poor SSIM preservation ({mean_ssim:.3f} < 0.95)',
                f'High information loss ({mean_info_loss:.1f}% > 15%)',
                f'Weak histogram correlation ({mean_hist_corr:.3f} < 0.9)'
            ]
        
        return recommendation
    
    def analyze_dataset_characteristics(self):
        """Analyze overall dataset characteristics"""
        print(f"\n=== Analyzing dataset characteristics ===")
        
        characteristics = {
            'total_images': len(self.image_paths),
            'image_sizes': [],
            'pixel_value_stats': {
                'means': [],
                'stds': [],
                'mins': [],
                'maxs': [],
                'medians': []
            },
            'file_sizes_mb': [],
            'healthy_images': 0,
            'cancerous_images': 0
        }
        
        # Analyze a sample of images
        sample_size = min(400, len(self.image_paths))
        sample_paths = np.random.choice(self.image_paths, sample_size, replace=False)
        
        for img_path in sample_paths:
            try:
                # Load image
                img = np.array(Image.open(img_path))
                
                # Basic characteristics
                characteristics['image_sizes'].append(img.shape)
                characteristics['file_sizes_mb'].append(img_path.stat().st_size / (1024*1024))
                
                # Pixel statistics
                characteristics['pixel_value_stats']['means'].append(np.mean(img))
                characteristics['pixel_value_stats']['stds'].append(np.std(img))
                characteristics['pixel_value_stats']['mins'].append(np.min(img))
                characteristics['pixel_value_stats']['maxs'].append(np.max(img))
                characteristics['pixel_value_stats']['medians'].append(np.median(img))
                
                # Try to infer class from filename/path (adjust based on your naming)
                path_str = str(img_path).lower()
                if 'healthy' in path_str or 'normal' in path_str:
                    characteristics['healthy_images'] += 1
                elif 'cancer' in path_str or 'malignant' in path_str:
                    characteristics['cancerous_images'] += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Print summary
        print(f"Dataset Analysis Results:")
        print(f"- Total images: {characteristics['total_images']}")
        print(f"- Sample analyzed: {len(sample_paths)}")
        print(f"- Image dimensions: {set(characteristics['image_sizes'])}")
        print(f"- Average file size: {np.mean(characteristics['file_sizes_mb']):.2f} MB")
        print(f"- Pixel value range: {np.mean(characteristics['pixel_value_stats']['mins']):.0f} - {np.mean(characteristics['pixel_value_stats']['maxs']):.0f}")
        print(f"- Mean pixel intensity: {np.mean(characteristics['pixel_value_stats']['means']):.0f} ± {np.mean(characteristics['pixel_value_stats']['stds']):.0f}")
        
        self.results['dataset_characteristics'] = characteristics
        return characteristics
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print(f"\n=== Generating Analysis Report ===")
        
        report_path = self.output_dir / 'cars_data_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# CARS Microscopy Data Analysis Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n")
            chars = self.results.get('dataset_characteristics', {})
            f.write(f"- **Total Images**: {chars.get('total_images', 'Unknown')}\n")
            f.write(f"- **Sample Analyzed**: {len(chars.get('image_sizes', []))}\n")
            f.write(f"- **Image Dimensions**: {set(chars.get('image_sizes', []))}\n")
            f.write(f"- **Average File Size**: {np.mean(chars.get('file_sizes_mb', [0])):.2f} MB\n\n")
            
            # Bit depth analysis
            if 'bit_depth_analysis' in self.results:
                f.write("## 16-bit to 8-bit Conversion Analysis\n\n")
                bit_analysis = self.results['bit_depth_analysis']
                recommendation = bit_analysis['recommendation']
                
                f.write("### Recommendation\n")
                f.write(f"**Use 8-bit conversion**: {'✅ YES' if recommendation['use_8bit'] else '❌ NO'}\n")
                f.write(f"**Confidence**: {recommendation['confidence'].upper()}\n\n")
                
                f.write("### Reasoning\n")
                for reason in recommendation['reasons']:
                    f.write(f"- {reason}\n")
                f.write("\n")
                
                f.write("### Detailed Metrics\n")
                f.write(bit_analysis['summary_stats'].to_markdown(index=False))
                f.write("\n\n")
            
            f.write("## Next Steps\n")
            f.write("1. Review the generated plots in the outputs folder\n")
            f.write("2. Based on the recommendation, proceed with appropriate bit depth\n")
            f.write("3. Set up FASTGAN training with optimized preprocessing\n")
            f.write("4. Consider data augmentation strategies for the limited dataset\n")
        
        print(f"Report saved to: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Analyze CARS microscopy data for FASTGAN training')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to directory containing CARS images')
    parser.add_argument('--output_dir', type=str, default='./outputs/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--sample_size', type=int, default=50,
                       help='Number of images to sample for bit depth analysis')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CARSDataAnalyzer(args.data_path, args.output_dir)
    
    # Run analyses
    print("Starting CARS data analysis...")
    
    # Dataset characteristics
    analyzer.analyze_dataset_characteristics()
    
    # Bit depth analysis
    analyzer.analyze_bit_depth_impact(args.sample_size)
    
    # Generate report
    analyzer.generate_report()
    
    print(f"\nAnalysis complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()