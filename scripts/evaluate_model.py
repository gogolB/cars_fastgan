"""
Comprehensive Model Evaluation for CARS-FASTGAN
Post-training evaluation with medical imaging metrics, sample generation, and analysis
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.fastgan_module import FastGANModule
from src.evaluation.metrics import ComprehensiveEvaluator
from src.data.dataset import CARSDataModule


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, checkpoint_path: str, data_path: str, output_dir: str = "outputs/evaluation"):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"🔬 CARS-FASTGAN Model Evaluation")
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        print(f"📊 Data: {self.data_path}")
        print(f"💾 Output: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
        print("=" * 60)
        
        # Storage for results
        self.evaluation_results = {}
        self.model = None
        self.datamodule = None
    
    def load_model(self) -> bool:
        """Load trained model from checkpoint"""
        print("🔄 Loading trained model...")
        
        try:
            self.model = FastGANModule.load_from_checkpoint(
                self.checkpoint_path,
                map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)
            
            print(f"✅ Model loaded successfully")
            print(f"   - Latent dim: {self.model.hparams.latent_dim}")
            print(f"   - Generator filters: {self.model.hparams.ngf}")
            print(f"   - Discriminator filters: {self.model.hparams.ndf}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def setup_data(self) -> bool:
        """Setup data module"""
        print("\n📊 Setting up data...")
        
        try:
            self.datamodule = CARSDataModule(
                data_path=str(self.data_path),
                batch_size=16,  # Larger batch for evaluation
                num_workers=4,
                image_size=512,
                use_8bit=True,
                augment_train=False,  # No augmentation for evaluation
                augment_val=False,
                augment_test=False
            )
            
            self.datamodule.setup()
            
            print(f"✅ Data setup complete")
            print(f"   - Train samples: {len(self.datamodule.train_dataset)}")
            print(f"   - Val samples: {len(self.datamodule.val_dataset)}")
            print(f"   - Test samples: {len(self.datamodule.test_dataset)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup data: {e}")
            return False
    
    def generate_sample_grid(self, num_samples: int = 64, grid_size: Tuple[int, int] = (8, 8)) -> Path:
        """Generate a grid of sample images"""
        print(f"\n🎨 Generating {num_samples} sample images...")
        
        with torch.no_grad():
            # Generate samples
            samples = []
            batch_size = 16
            
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                fake_images = self.model.model.generate(current_batch_size, self.device)
                samples.append(fake_images.cpu())
            
            samples = torch.cat(samples, dim=0)[:num_samples]
        
        # Create grid visualization
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 16))
        fig.suptitle('Generated CARS Microscopy Images', fontsize=20)
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                img = samples[i, 0].numpy()  # Remove channel dimension
                img = (img + 1) / 2  # Denormalize to [0, 1]
                img = np.clip(img, 0, 1)
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Sample {i+1}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save grid
        grid_path = self.output_dir / "samples" / "generated_grid.png"
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual samples
        for i in range(min(16, num_samples)):  # Save first 16 as individual files
            img = samples[i, 0].numpy()
            img = (img + 1) / 2
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            sample_path = self.output_dir / "samples" / f"sample_{i+1:03d}.png"
            Image.fromarray(img).save(sample_path)
        
        print(f"✅ Sample grid saved: {grid_path}")
        print(f"✅ Individual samples saved: {self.output_dir}/samples/")
        
        return grid_path
    
    def create_real_vs_fake_comparison(self, num_comparisons: int = 16) -> Path:
        """Create side-by-side comparison of real vs generated images"""
        print(f"\n🔍 Creating real vs fake comparisons...")
        
        # Get real images
        val_loader = self.datamodule.val_dataloader()
        real_batch = next(iter(val_loader))
        real_images = real_batch['image'][:num_comparisons]
        
        # Generate fake images
        with torch.no_grad():
            fake_images = self.model.model.generate(num_comparisons, self.device)
            fake_images = fake_images.cpu()
        
        # Create comparison grid
        rows = 4
        cols = num_comparisons // rows
        fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 16))
        fig.suptitle('Real vs Generated CARS Images', fontsize=20)
        
        for i in range(num_comparisons):
            row = (i // cols) * 2
            col = i % cols
            
            # Real image (top)
            real_img = real_images[i, 0].numpy()
            real_img = (real_img + 1) / 2
            real_img = np.clip(real_img, 0, 1)
            
            axes[row, col].imshow(real_img, cmap='gray')
            axes[row, col].set_title(f'Real {i+1}', fontsize=10)
            axes[row, col].axis('off')
            
            # Fake image (bottom)
            fake_img = fake_images[i, 0].numpy()
            fake_img = (fake_img + 1) / 2
            fake_img = np.clip(fake_img, 0, 1)
            
            axes[row + 1, col].imshow(fake_img, cmap='gray')
            axes[row + 1, col].set_title(f'Generated {i+1}', fontsize=10)
            axes[row + 1, col].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = self.output_dir / "comparisons" / "real_vs_fake.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison saved: {comparison_path}")
        return comparison_path
    
    def run_comprehensive_metrics(self, num_real: int = 500, num_fake: int = 1000) -> Dict:
        """Run comprehensive evaluation metrics"""
        print(f"\n📊 Running comprehensive metrics evaluation...")
        print(f"   - Real images: {num_real}")
        print(f"   - Generated images: {num_fake}")
        
        # Collect real images
        print("📥 Collecting real images...")
        real_images = []
        val_loader = self.datamodule.val_dataloader()
        
        with tqdm(total=num_real, desc="Real images") as pbar:
            for batch in val_loader:
                batch_images = batch['image']
                real_images.append(batch_images)
                pbar.update(len(batch_images))
                
                if len(torch.cat(real_images, dim=0)) >= num_real:
                    break
        
        real_images = torch.cat(real_images, dim=0)[:num_real]
        
        # Generate fake images
        print("🎨 Generating fake images...")
        fake_images = []
        batch_size = 16
        
        with torch.no_grad():
            with tqdm(total=num_fake, desc="Fake images") as pbar:
                for i in range(0, num_fake, batch_size):
                    current_batch_size = min(batch_size, num_fake - i)
                    fake_batch = self.model.model.generate(current_batch_size, self.device)
                    fake_images.append(fake_batch.cpu())
                    pbar.update(current_batch_size)
        
        fake_images = torch.cat(fake_images, dim=0)[:num_fake]
        
        # Run comprehensive evaluation
        print("🔬 Computing evaluation metrics...")
        evaluator = ComprehensiveEvaluator(device=str(self.device))
        results = evaluator.evaluate(real_images, fake_images)
        
        # Save results
        results_path = self.output_dir / "metrics" / "comprehensive_metrics.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (torch.Tensor, np.ndarray, np.float32, np.float64)) else v 
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.evaluation_results['metrics'] = serializable_results
        
        print(f"✅ Metrics saved: {results_path}")
        
        # Print key results
        if 'fid_score' in results:
            print(f"🎯 FID Score: {results['fid_score']:.3f}")
        if 'lpips_score' in results:
            print(f"📏 LPIPS Score: {results['lpips_score']:.3f}")
        if 'is_mean' in results:
            print(f"📈 IS Score: {results['is_mean']:.3f} ± {results.get('is_std', 0):.3f}")
        
        return serializable_results
    
    def analyze_latent_space(self, num_samples: int = 100) -> Path:
        """Analyze latent space properties"""
        print(f"\n🔬 Analyzing latent space...")
        
        # Generate samples with different latent codes
        latent_codes = []
        generated_images = []
        
        with torch.no_grad():
            for i in range(num_samples):
                z = torch.randn(1, self.model.hparams.latent_dim, device=self.device)
                fake_img = self.model.model.generator(z)
                
                latent_codes.append(z.cpu().numpy().flatten())
                generated_images.append(fake_img.cpu().numpy().flatten())
        
        latent_codes = np.array(latent_codes)
        generated_images = np.array(generated_images)
        
        # Analyze latent space structure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Latent Space Analysis', fontsize=16)
        
        # Latent code distribution
        axes[0, 0].hist(latent_codes.flatten(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Latent Code Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Generated image pixel distribution
        axes[0, 1].hist(generated_images.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Generated Pixel Distribution')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # Latent space variance
        latent_vars = np.var(latent_codes, axis=0)
        axes[0, 2].plot(latent_vars)
        axes[0, 2].set_title('Latent Dimension Variance')
        axes[0, 2].set_xlabel('Dimension')
        axes[0, 2].set_ylabel('Variance')
        
        # Correlation between latent dimensions
        latent_corr = np.corrcoef(latent_codes.T)
        im = axes[1, 0].imshow(latent_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Latent Dimension Correlations')
        plt.colorbar(im, ax=axes[1, 0])
        
        # PCA of latent space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_codes)
        
        axes[1, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6)
        axes[1, 1].set_title(f'Latent Space PCA\n(Variance explained: {pca.explained_variance_ratio_.sum():.3f})')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        
        # Latent interpolation example
        # Generate interpolation between two random points
        z1 = torch.randn(1, self.model.hparams.latent_dim, device=self.device)
        z2 = torch.randn(1, self.model.hparams.latent_dim, device=self.device)
        
        interp_steps = 8
        interp_images = []
        
        with torch.no_grad():
            for i in range(interp_steps):
                alpha = i / (interp_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = self.model.model.generator(z_interp)
                img = (img.cpu().numpy()[0, 0] + 1) / 2
                interp_images.append(img)
        
        # Display interpolation in subplot
        interp_grid = np.concatenate(interp_images, axis=1)
        axes[1, 2].imshow(interp_grid, cmap='gray')
        axes[1, 2].set_title('Latent Space Interpolation')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save analysis
        analysis_path = self.output_dir / "analysis" / "latent_space_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Latent space analysis saved: {analysis_path}")
        return analysis_path
    
    def create_diversity_analysis(self, num_samples: int = 100) -> Path:
        """Analyze diversity of generated samples"""
        print(f"\n🎨 Analyzing sample diversity...")
        
        # Generate samples
        with torch.no_grad():
            fake_images = self.model.model.generate(num_samples, self.device)
            fake_images = fake_images.cpu()
        
        # Convert to numpy for analysis
        images_np = fake_images.numpy()
        
        # Calculate pairwise distances
        from sklearn.metrics import pairwise_distances
        
        # Flatten images for distance calculation
        images_flat = images_np.reshape(num_samples, -1)
        distances = pairwise_distances(images_flat, metric='euclidean')
        
        # Remove diagonal (self-distances)
        mask = np.eye(distances.shape[0], dtype=bool)
        distances_no_diag = distances[~mask]
        
        # Create diversity analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sample Diversity Analysis', fontsize=16)
        
        # Distance distribution
        axes[0, 0].hist(distances_no_diag, bins=50, alpha=0.7)
        axes[0, 0].set_title('Pairwise Distance Distribution')
        axes[0, 0].set_xlabel('Euclidean Distance')
        axes[0, 0].set_ylabel('Frequency')
        
        # Distance heatmap (subset)
        subset_size = min(20, num_samples)
        distance_subset = distances[:subset_size, :subset_size]
        im = axes[0, 1].imshow(distance_subset, cmap='viridis')
        axes[0, 1].set_title(f'Distance Matrix (First {subset_size} samples)')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Pixel intensity statistics
        pixel_means = np.mean(images_flat, axis=1)
        pixel_stds = np.std(images_flat, axis=1)
        
        axes[1, 0].scatter(pixel_means, pixel_stds, alpha=0.6)
        axes[1, 0].set_title('Pixel Statistics Distribution')
        axes[1, 0].set_xlabel('Mean Pixel Value')
        axes[1, 0].set_ylabel('Std Pixel Value')
        
        # Sample variance across dataset
        feature_variance = np.var(images_flat, axis=0)
        axes[1, 1].hist(feature_variance, bins=50, alpha=0.7)
        axes[1, 1].set_title('Feature Variance Distribution')
        axes[1, 1].set_xlabel('Variance')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save analysis
        diversity_path = self.output_dir / "analysis" / "diversity_analysis.png"
        plt.savefig(diversity_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate diversity metrics
        diversity_metrics = {
            'mean_pairwise_distance': float(np.mean(distances_no_diag)),
            'std_pairwise_distance': float(np.std(distances_no_diag)),
            'min_pairwise_distance': float(np.min(distances_no_diag)),
            'max_pairwise_distance': float(np.max(distances_no_diag)),
            'mean_pixel_mean': float(np.mean(pixel_means)),
            'std_pixel_mean': float(np.std(pixel_means)),
            'overall_feature_variance': float(np.mean(feature_variance))
        }
        
        # Save diversity metrics
        diversity_metrics_path = self.output_dir / "metrics" / "diversity_metrics.json"
        with open(diversity_metrics_path, 'w') as f:
            json.dump(diversity_metrics, f, indent=2)
        
        self.evaluation_results['diversity'] = diversity_metrics
        
        print(f"✅ Diversity analysis saved: {diversity_path}")
        print(f"📊 Diversity metrics: Mean distance = {diversity_metrics['mean_pairwise_distance']:.3f}")
        
        return diversity_path
    
    def generate_final_report(self) -> Path:
        """Generate comprehensive evaluation report"""
        print(f"\n📋 Generating final evaluation report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_path = self.output_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# CARS-FASTGAN Model Evaluation Report\n\n")
            f.write(f"**Generated**: {timestamp}\n")
            f.write(f"**Model**: {self.checkpoint_path.name}\n")
            f.write(f"**Data**: {self.data_path}\n")
            f.write(f"**Device**: {self.device}\n\n")
            
            # Model Summary
            f.write("## Model Summary\n\n")
            if self.model:
                f.write(f"- **Latent Dimension**: {self.model.hparams.latent_dim}\n")
                f.write(f"- **Generator Filters**: {self.model.hparams.ngf}\n")
                f.write(f"- **Discriminator Filters**: {self.model.hparams.ndf}\n")
                f.write(f"- **Generator Layers**: {self.model.hparams.generator_layers}\n")
                f.write(f"- **Discriminator Layers**: {self.model.hparams.discriminator_layers}\n")
            f.write("\n")
            
            # Dataset Summary
            f.write("## Dataset Summary\n\n")
            if self.datamodule:
                f.write(f"- **Training Images**: {len(self.datamodule.train_dataset)}\n")
                f.write(f"- **Validation Images**: {len(self.datamodule.val_dataset)}\n")
                f.write(f"- **Test Images**: {len(self.datamodule.test_dataset)}\n")
            f.write("\n")
            
            # Evaluation Metrics
            f.write("## Evaluation Metrics\n\n")
            if 'metrics' in self.evaluation_results:
                metrics = self.evaluation_results['metrics']
                
                f.write("### Image Quality Metrics\n")
                if 'fid_score' in metrics:
                    f.write(f"- **FID Score**: {metrics['fid_score']:.3f}\n")
                    if metrics['fid_score'] < 50:
                        f.write("  - ✅ Excellent quality\n")
                    elif metrics['fid_score'] < 100:
                        f.write("  - 👍 Good quality\n")
                    else:
                        f.write("  - 📈 Needs improvement\n")
                
                if 'lpips_score' in metrics:
                    f.write(f"- **LPIPS Score**: {metrics['lpips_score']:.3f}\n")
                
                if 'is_mean' in metrics:
                    f.write(f"- **Inception Score**: {metrics['is_mean']:.3f} ± {metrics.get('is_std', 0):.3f}\n")
                
                f.write("\n")
            
            # Diversity Analysis
            f.write("### Diversity Analysis\n")
            if 'diversity' in self.evaluation_results:
                diversity = self.evaluation_results['diversity']
                f.write(f"- **Mean Pairwise Distance**: {diversity['mean_pairwise_distance']:.3f}\n")
                f.write(f"- **Distance Std**: {diversity['std_pairwise_distance']:.3f}\n")
                f.write(f"- **Feature Variance**: {diversity['overall_feature_variance']:.6f}\n")
                
                # Diversity assessment
                if diversity['std_pairwise_distance'] > diversity['mean_pairwise_distance'] * 0.1:
                    f.write("- ✅ Good sample diversity\n")
                else:
                    f.write("- ⚠️  Low sample diversity - possible mode collapse\n")
                f.write("\n")
            
            # Medical Imaging Assessment
            f.write("## Medical Imaging Assessment\n\n")
            if 'metrics' in self.evaluation_results:
                metrics = self.evaluation_results['metrics']
                
                if 'real_texture' in metrics and 'fake_texture' in metrics:
                    f.write("### Texture Analysis\n")
                    real_texture = metrics['real_texture']
                    fake_texture = metrics['fake_texture']
                    
                    # Compare key texture metrics
                    texture_metrics = ['lbp_uniformity_mean', 'glcm_contrast_mean', 'entropy_mean']
                    for metric in texture_metrics:
                        if metric in real_texture and metric in fake_texture:
                            real_val = real_texture[metric]
                            fake_val = fake_texture[metric]
                            diff_pct = abs(real_val - fake_val) / real_val * 100
                            
                            f.write(f"- **{metric.replace('_', ' ').title()}**:\n")
                            f.write(f"  - Real: {real_val:.4f}\n")
                            f.write(f"  - Generated: {fake_val:.4f}\n")
                            f.write(f"  - Difference: {diff_pct:.1f}%\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Based on FID score
            if 'metrics' in self.evaluation_results and 'fid_score' in self.evaluation_results['metrics']:
                fid = self.evaluation_results['metrics']['fid_score']
                
                if fid < 30:
                    f.write("### 🎉 Excellent Model Performance\n")
                    f.write("- Model has achieved excellent image quality\n")
                    f.write("- Generated images are very close to real CARS microscopy\n")
                    f.write("- Ready for data augmentation use\n")
                elif fid < 70:
                    f.write("### ✅ Good Model Performance\n")
                    f.write("- Model generates reasonable quality images\n")
                    f.write("- Consider additional training for further improvement\n")
                    f.write("- Suitable for data augmentation with careful validation\n")
                else:
                    f.write("### ⚠️  Model Needs Improvement\n")
                    f.write("- Generated images quality is below optimal\n")
                    f.write("- Consider:\n")
                    f.write("  - Longer training\n")
                    f.write("  - Different model architecture\n")
                    f.write("  - Hyperparameter tuning\n")
                    f.write("  - Data quality assessment\n")
            
            f.write("\n### Next Steps\n")
            f.write("1. **Review generated samples** in `samples/` directory\n")
            f.write("2. **Analyze diversity plots** in `analysis/` directory\n")
            f.write("3. **Compare real vs fake** images in `comparisons/`\n")
            f.write("4. **Use for data augmentation** if quality is sufficient\n")
            f.write("5. **Consider classifier evaluation** on generated data\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("```\n")
            f.write("evaluation/\n")
            f.write("├── samples/                 # Generated image samples\n")
            f.write("├── comparisons/             # Real vs fake comparisons\n")
            f.write("├── metrics/                 # Quantitative metrics\n")
            f.write("├── analysis/                # Latent space & diversity analysis\n")
            f.write("└── evaluation_report.md     # This report\n")
            f.write("```\n")
            
            f.write("\n---\n")
            f.write("*Report generated by CARS-FASTGAN Evaluation Suite*\n")
        
        print(f"✅ Final report saved: {report_path}")
        return report_path
    
    def run_complete_evaluation(
        self,
        num_samples: int = 64,
        num_real_metrics: int = 500,
        num_fake_metrics: int = 1000,
        num_diversity_samples: int = 100
    ) -> bool:
        """Run complete evaluation pipeline"""
        
        print("🚀 Starting complete model evaluation...")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Setup data
        if not self.setup_data():
            return False
        
        # Generate sample grid
        self.generate_sample_grid(num_samples)
        
        # Create real vs fake comparison
        self.create_real_vs_fake_comparison()
        
        # Run comprehensive metrics
        self.run_comprehensive_metrics(num_real_metrics, num_fake_metrics)
        
        # Analyze latent space
        self.analyze_latent_space()
        
        # Analyze diversity
        self.create_diversity_analysis(num_diversity_samples)
        
        # Generate final report
        self.generate_final_report()
        
        print("\n" + "=" * 60)
        print("🎉 Complete evaluation finished!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("📋 Check evaluation_report.md for detailed analysis")
        print("=" * 60)
        
        return True


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of trained CARS-FASTGAN model')
    
    parser.add_argument('checkpoint_path', type=str,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('data_path', type=str,
                       help='Path to prepared CARS data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate for visualization')
    parser.add_argument('--num_real_metrics', type=int, default=500,
                       help='Number of real images for metrics calculation')
    parser.add_argument('--num_fake_metrics', type=int, default=1000,
                       help='Number of fake images for metrics calculation')
    parser.add_argument('--num_diversity_samples', type=int, default=100,
                       help='Number of samples for diversity analysis')
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False
    
    # Create evaluator and run
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    success = evaluator.run_complete_evaluation(
        num_samples=args.num_samples,
        num_real_metrics=args.num_real_metrics,
        num_fake_metrics=args.num_fake_metrics,
        num_diversity_samples=args.num_diversity_samples
    )
    
    if success:
        print("\n✨ Evaluation completed successfully!")
        
        # Print key results if available
        if evaluator.evaluation_results:
            print("\n📊 Key Results:")
            
            if 'metrics' in evaluator.evaluation_results:
                metrics = evaluator.evaluation_results['metrics']
                if 'fid_score' in metrics:
                    print(f"   🎯 FID Score: {metrics['fid_score']:.3f}")
                if 'lpips_score' in metrics:
                    print(f"   📏 LPIPS Score: {metrics['lpips_score']:.3f}")
                if 'is_mean' in metrics:
                    print(f"   📈 IS Score: {metrics['is_mean']:.3f}")
            
            if 'diversity' in evaluator.evaluation_results:
                diversity = evaluator.evaluation_results['diversity']
                print(f"   🎨 Sample Diversity: {diversity['mean_pairwise_distance']:.3f}")
        
        print(f"\n📁 Full results: {args.output_dir}")
        print("📋 Read evaluation_report.md for detailed analysis and recommendations")
        
    else:
        print("\n❌ Evaluation failed!")
        
    return success


if __name__ == "__main__":
    main()