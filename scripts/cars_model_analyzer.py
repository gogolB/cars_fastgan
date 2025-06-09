#!/usr/bin/env python3
"""
CARS-FASTGAN Model Analyzer - Complete Rewrite
A comprehensive tool for monitoring training and analyzing generated images

Features:
- Training monitoring from TensorBoard logs
- Image quality analysis
- Model checkpoint evaluation
- Detailed reporting with visualizations
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Image processing
from PIL import Image, ImageEnhance
import cv2
from skimage import filters, feature, measure
from scipy import ndimage, signal

# Deep learning
import torch
import torch.nn.functional as F

# Progress and UI
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# TensorBoard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")

# Project imports
try:
    from src.training.fastgan_module import FastGANModule
    from src.data.dataset import CARSDataModule
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError:
    PROJECT_IMPORTS_AVAILABLE = False
    print("⚠️ Project imports not available. Some features will be limited.")


# Initialize console for rich output
console = Console()


# ==================== Data Classes ====================

@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step"""
    epoch: int
    step: int
    d_loss: float = 0.0
    g_loss: float = 0.0
    d_loss_real: float = 0.0
    d_loss_fake: float = 0.0
    val_d_loss: Optional[float] = None
    val_g_loss: Optional[float] = None
    fid_score: Optional[float] = None
    lr_g: float = 0.0
    lr_d: float = 0.0


@dataclass
class AnalysisResults:
    """Container for all analysis results"""
    # Training metrics
    training_metrics: List[TrainingMetrics] = field(default_factory=list)
    training_health_score: float = 100.0
    training_stable: bool = True
    convergence_status: str = "unknown"
    
    # Image analysis
    image_quality_metrics: Dict[str, float] = field(default_factory=dict)
    noise_metrics: Dict[str, float] = field(default_factory=dict)
    structural_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations and issues
    recommendations: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    # Generated files
    plots: Dict[str, Path] = field(default_factory=dict)
    report_path: Optional[Path] = None


# ==================== Training Monitor ====================

class TrainingMonitor:
    """Monitors and analyzes training progress from TensorBoard logs"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def find_tensorboard_logs(self, experiment_dir: Path) -> List[Path]:
        """Find all TensorBoard event files"""
        console.print(f"\n[bold]🔍 Searching for TensorBoard logs in:[/bold] {experiment_dir}")
        
        event_files = []
        for file_path in experiment_dir.rglob("*"):
            if "events.out.tfevents" in file_path.name and file_path.is_file():
                event_files.append(file_path)
        
        if not event_files:
            console.print("[red]❌ No TensorBoard event files found![/red]")
            return []
        
        # Sort by modification time (newest first)
        event_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        console.print(f"[green]✅ Found {len(event_files)} event file(s)[/green]")
        for i, ef in enumerate(event_files[:3]):  # Show first 3
            size_kb = ef.stat().st_size / 1024
            console.print(f"  {i+1}. {ef.parent.name}/{ef.name} ({size_kb:.1f} KB)")
        
        return event_files
    
    def parse_event_file(self, event_file: Path) -> List[TrainingMetrics]:
        """Parse a single TensorBoard event file"""
        if not TENSORBOARD_AVAILABLE:
            console.print("[red]❌ TensorBoard not available![/red]")
            return []
        
        console.print(f"\n[bold]📊 Parsing event file:[/bold] {event_file.name}")
        
        try:
            # Use the parent directory for EventAccumulator
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            # Get scalar tags
            scalar_tags = ea.Tags()['scalars']
            console.print(f"[blue]Found {len(scalar_tags)} scalar metrics[/blue]")
            
            # Show available metrics
            if scalar_tags:
                console.print("\n[bold]Available metrics:[/bold]")
                for tag in sorted(scalar_tags)[:10]:  # Show first 10
                    console.print(f"  • {tag}")
                if len(scalar_tags) > 10:
                    console.print(f"  ... and {len(scalar_tags) - 10} more")
            
            # Parse metrics
            metrics_dict = {}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task("Parsing metrics...", total=len(scalar_tags))
                
                for tag in scalar_tags:
                    events = ea.Scalars(tag)
                    
                    for event in events:
                        step = event.step
                        
                        if step not in metrics_dict:
                            metrics_dict[step] = TrainingMetrics(
                                epoch=step // 100 if step > 0 else 0,  # Approximate
                                step=step
                            )
                        
                        metric = metrics_dict[step]
                        tag_lower = tag.lower()
                        
                        # Parse different metric types
                        if 'train/d_loss' in tag and 'real' not in tag and 'fake' not in tag:
                            metric.d_loss = event.value
                        elif 'train/g_loss' in tag:
                            metric.g_loss = event.value
                        elif 'train/d_loss_real' in tag:
                            metric.d_loss_real = event.value
                        elif 'train/d_loss_fake' in tag:
                            metric.d_loss_fake = event.value
                        elif 'val/d_loss' in tag:
                            metric.val_d_loss = event.value
                        elif 'val/g_loss' in tag:
                            metric.val_g_loss = event.value
                        elif 'fid' in tag_lower:
                            metric.fid_score = event.value
                        elif 'lr' in tag_lower:
                            if 'g' in tag_lower or 'gen' in tag_lower:
                                metric.lr_g = event.value
                            elif 'd' in tag_lower or 'disc' in tag_lower:
                                metric.lr_d = event.value
                    
                    progress.update(task, advance=1)
            
            # Convert to sorted list
            metrics = sorted(metrics_dict.values(), key=lambda m: m.step)
            
            if metrics:
                latest = metrics[-1]
                console.print(f"\n[green]✅ Successfully parsed {len(metrics)} metric entries[/green]")
                console.print(f"[blue]Latest: Epoch {latest.epoch}, Step {latest.step}[/blue]")
                console.print(f"[blue]D Loss: {latest.d_loss:.4f}, G Loss: {latest.g_loss:.4f}[/blue]")
            
            return metrics
            
        except Exception as e:
            console.print(f"[red]❌ Error parsing event file: {str(e)}[/red]")
            return []
    
    def analyze_training_health(self, metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """Analyze training health and stability"""
        if not metrics:
            return {
                'health_score': 0,
                'stable': False,
                'convergence': 'no_data',
                'recommendations': ['No training data available']
            }
        
        console.print("\n[bold]🏥 Analyzing training health...[/bold]")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'step': m.step,
            'epoch': m.epoch,
            'd_loss': m.d_loss,
            'g_loss': m.g_loss,
            'd_loss_real': m.d_loss_real,
            'd_loss_fake': m.d_loss_fake
        } for m in metrics])
        
        # Analyze recent performance (last 20%)
        recent_window = max(10, len(df) // 5)
        recent_df = df.tail(recent_window)
        
        health_score = 100
        recommendations = []
        
        # Check for loss explosion
        if recent_df['d_loss'].max() > 10 or recent_df['g_loss'].max() > 10:
            health_score -= 30
            recommendations.append("⚠️ Loss explosion detected - reduce learning rate")
        
        # Check for high variance
        d_std = recent_df['d_loss'].std()
        g_std = recent_df['g_loss'].std()
        
        if d_std > 2.0 or g_std > 2.0:
            health_score -= 20
            recommendations.append("⚠️ High loss variance - training may be unstable")
        
        # Check discriminator/generator balance
        d_mean = recent_df['d_loss'].mean()
        g_mean = recent_df['g_loss'].mean()
        
        if d_mean < 0.1:
            health_score -= 25
            recommendations.append("⚠️ Discriminator too strong - risk of mode collapse")
        elif g_mean < 0.1:
            health_score -= 25
            recommendations.append("⚠️ Generator too strong - discriminator failing")
        
        # Check convergence
        if len(df) > 50:
            # Linear regression on recent losses
            from scipy import stats
            slope_d, _, _, _, _ = stats.linregress(recent_df.index, recent_df['d_loss'])
            slope_g, _, _, _, _ = stats.linregress(recent_df.index, recent_df['g_loss'])
            
            if abs(slope_d) < 0.001 and abs(slope_g) < 0.001:
                convergence = 'converged'
                recommendations.append("✅ Training appears to have converged")
            elif slope_d > 0.01 and slope_g > 0.01:
                convergence = 'diverging'
                health_score -= 30
                recommendations.append("❌ Training appears to be diverging")
            else:
                convergence = 'training'
        else:
            convergence = 'early_stage'
        
        # Final recommendation
        if health_score >= 80:
            recommendations.append("✅ Training is healthy overall")
        elif health_score >= 60:
            recommendations.append("⚠️ Training has minor issues - monitor closely")
        else:
            recommendations.append("❌ Training has serious issues - intervention needed")
        
        return {
            'health_score': health_score,
            'stable': health_score >= 60,
            'convergence': convergence,
            'recommendations': recommendations,
            'd_loss_mean': d_mean,
            'g_loss_mean': g_mean,
            'd_loss_std': d_std,
            'g_loss_std': g_std
        }
    
    def create_training_plots(self, metrics: List[TrainingMetrics]) -> Path:
        """Create comprehensive training visualization plots"""
        if not metrics:
            return None
        
        console.print("\n[bold]📈 Creating training plots...[/bold]")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'step': m.step,
            'epoch': m.epoch,
            'd_loss': m.d_loss,
            'g_loss': m.g_loss,
            'd_loss_real': m.d_loss_real,
            'd_loss_fake': m.d_loss_fake,
            'val_d_loss': m.val_d_loss,
            'val_g_loss': m.val_g_loss
        } for m in metrics])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main losses over time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(df['step'], df['d_loss'], label='D Loss', alpha=0.8, linewidth=2)
        ax1.plot(df['step'], df['g_loss'], label='G Loss', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Discriminator components
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(df['step'], df['d_loss_real'], label='Real', alpha=0.8)
        ax2.plot(df['step'], df['d_loss_fake'], label='Fake', alpha=0.8)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Discriminator Components', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss distributions
        ax3 = fig.add_subplot(gs[1, 0])
        recent_window = max(100, len(df) // 10)
        recent_df = df.tail(recent_window)
        ax3.hist(recent_df['d_loss'], bins=30, alpha=0.6, label='D Loss', density=True)
        ax3.hist(recent_df['g_loss'], bins=30, alpha=0.6, label='G Loss', density=True)
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Loss Distributions (Last {recent_window} steps)', fontsize=12)
        ax3.legend()
        
        # 4. Moving averages
        ax4 = fig.add_subplot(gs[1, 1:])
        window = min(50, len(df) // 20)
        df['d_loss_ma'] = df['d_loss'].rolling(window=window, center=True).mean()
        df['g_loss_ma'] = df['g_loss'].rolling(window=window, center=True).mean()
        ax4.plot(df['step'], df['d_loss_ma'], label=f'D Loss (MA-{window})', linewidth=2)
        ax4.plot(df['step'], df['g_loss_ma'], label=f'G Loss (MA-{window})', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Smoothed Loss Trends', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Validation losses (if available)
        ax5 = fig.add_subplot(gs[2, 0])
        val_df = df[df['val_d_loss'].notna()]
        if not val_df.empty:
            ax5.plot(val_df['step'], val_df['val_d_loss'], 'o-', label='Val D Loss', markersize=6)
            ax5.plot(val_df['step'], val_df['val_g_loss'], 'o-', label='Val G Loss', markersize=6)
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Loss')
            ax5.set_title('Validation Losses', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No validation data', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Validation Losses', fontsize=12)
        
        # 6. Training dynamics
        ax6 = fig.add_subplot(gs[2, 1:])
        # Calculate loss ratio
        loss_ratio = df['g_loss'] / (df['d_loss'] + 1e-8)
        ax6.plot(df['step'], loss_ratio, alpha=0.8, color='purple')
        ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balance Line')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('G/D Loss Ratio')
        ax6.set_title('Generator/Discriminator Balance', fontsize=12)
        ax6.set_ylim(0, 5)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('CARS-FASTGAN Training Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f"training_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Saved training plots to: {plot_path}[/green]")
        
        return plot_path


# ==================== Image Analyzer ====================

class ImageAnalyzer:
    """Analyzes generated and real images for quality metrics"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.generated_images = []
        self.real_images = []
    
    def load_images(self, generated_dir: Optional[Path] = None, 
                   real_dir: Optional[Path] = None,
                   max_images: int = 50) -> Tuple[int, int]:
        """Load generated and real images for analysis"""
        console.print("\n[bold]🖼️ Loading images for analysis...[/bold]")
        
        # Load generated images
        if generated_dir and generated_dir.exists():
            self.generated_images = self._load_images_from_dir(generated_dir, "generated", max_images)
        else:
            console.print("[yellow]⚠️ No generated images directory provided[/yellow]")
        
        # Load real images
        if real_dir and real_dir.exists():
            self.real_images = self._load_images_from_dir(real_dir, "real", max_images)
        else:
            console.print("[yellow]⚠️ No real images directory provided[/yellow]")
        
        return len(self.generated_images), len(self.real_images)
    
    def _load_images_from_dir(self, image_dir: Path, image_type: str, max_images: int) -> List[np.ndarray]:
        """Load images from a directory"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        image_paths = []
        
        # Find all image files
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        # Remove duplicates and limit
        image_paths = list(set(image_paths))[:max_images]
        
        if not image_paths:
            console.print(f"[yellow]No images found in {image_dir}[/yellow]")
            return []
        
        # Load images with progress bar
        images = []
        for img_path in tqdm(image_paths, desc=f"Loading {image_type} images"):
            try:
                img = np.array(Image.open(img_path).convert('L'))
                
                # Handle potential normalization issues
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.min() < -0.5:  # Likely in [-1, 1]
                        img = ((img + 1) / 2 * 255).astype(np.uint8)
                    elif img.max() <= 1.0:  # Likely in [0, 1]
                        img = (img * 255).astype(np.uint8)
                
                images.append(img)
            except Exception as e:
                console.print(f"[red]Error loading {img_path.name}: {e}[/red]")
        
        console.print(f"[green]✅ Loaded {len(images)} {image_type} images[/green]")
        return images
    
    def load_from_checkpoint(self, checkpoint_path: Path, num_samples: int = 16) -> List[np.ndarray]:
        """Generate and load images from a model checkpoint"""
        if not PROJECT_IMPORTS_AVAILABLE:
            console.print("[red]❌ Project imports not available for checkpoint loading[/red]")
            return []
        
        console.print(f"\n[bold]🔄 Loading checkpoint: {checkpoint_path.name}[/bold]")
        
        try:
            # Load model
            model = FastGANModule.load_from_checkpoint(str(checkpoint_path))
            model.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 
                                'mps' if torch.backends.mps.is_available() else 'cpu')
            model = model.to(device)
            
            console.print(f"[blue]Model loaded on {device}[/blue]")
            
            # Generate samples
            with torch.no_grad():
                z = torch.randn(num_samples, model.hparams.latent_dim, device=device)
                fake_images = model.model.generator(z)
                
                # Denormalize from [-1, 1] to [0, 255]
                fake_images = (fake_images + 1) / 2
                fake_images = (fake_images * 255).cpu().numpy().astype(np.uint8)
            
            # Convert to list of 2D arrays
            self.generated_images = [fake_images[i, 0] for i in range(num_samples)]
            
            console.print(f"[green]✅ Generated {len(self.generated_images)} samples[/green]")
            return self.generated_images
            
        except Exception as e:
            console.print(f"[red]❌ Error loading checkpoint: {e}[/red]")
            return []
    
    def analyze_image_quality(self) -> Dict[str, float]:
        """Analyze image quality metrics"""
        console.print("\n[bold]📊 Analyzing image quality...[/bold]")
        
        metrics = {}
        
        # Dynamic range analysis
        for images, prefix in [(self.real_images, 'real'), (self.generated_images, 'generated')]:
            if images:
                all_pixels = np.concatenate([img.flatten() for img in images])
                metrics[f'{prefix}_mean'] = np.mean(all_pixels)
                metrics[f'{prefix}_std'] = np.std(all_pixels)
                metrics[f'{prefix}_min'] = np.min(all_pixels)
                metrics[f'{prefix}_max'] = np.max(all_pixels)
                metrics[f'{prefix}_range'] = metrics[f'{prefix}_max'] - metrics[f'{prefix}_min']
        
        # Noise analysis
        for images, prefix in [(self.real_images, 'real'), (self.generated_images, 'generated')]:
            if images:
                noise_estimates = []
                for img in images:
                    # Estimate noise using difference from median filtered version
                    denoised = cv2.medianBlur(img, 5)
                    noise = np.std(img.astype(float) - denoised.astype(float))
                    noise_estimates.append(noise)
                metrics[f'{prefix}_noise'] = np.mean(noise_estimates)
        
        # Structural analysis (follicle detection for CARS images)
        for images, prefix in [(self.real_images, 'real'), (self.generated_images, 'generated')]:
            if images:
                follicle_counts = []
                edge_densities = []
                
                for img in images:
                    # Edge detection
                    edges = feature.canny(img.astype(np.uint8), sigma=2)
                    edge_densities.append(np.sum(edges) / edges.size)
                    
                    # Simple circle detection for follicles
                    circles = cv2.HoughCircles(
                        img.astype(np.uint8),
                        cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist=30,
                        param1=50,
                        param2=30,
                        minRadius=10,
                        maxRadius=50
                    )
                    
                    if circles is not None:
                        follicle_counts.append(len(circles[0]))
                    else:
                        follicle_counts.append(0)
                
                metrics[f'{prefix}_edge_density'] = np.mean(edge_densities)
                metrics[f'{prefix}_follicle_count'] = np.mean(follicle_counts)
        
        # Print summary
        console.print("\n[bold]Image Quality Metrics:[/bold]")
        
        # Create comparison table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Real", style="green")
        table.add_column("Generated", style="yellow")
        table.add_column("Difference", style="blue")
        
        # Add rows for each metric type
        metric_types = ['mean', 'std', 'range', 'noise', 'edge_density', 'follicle_count']
        
        for metric_type in metric_types:
            real_val = metrics.get(f'real_{metric_type}', 0)
            gen_val = metrics.get(f'generated_{metric_type}', 0)
            diff = abs(real_val - gen_val)
            
            table.add_row(
                metric_type.replace('_', ' ').title(),
                f"{real_val:.2f}",
                f"{gen_val:.2f}",
                f"{diff:.2f}"
            )
        
        console.print(table)
        
        return metrics
    
    def create_comparison_plots(self) -> Path:
        """Create visual comparison of real vs generated images"""
        if not self.real_images or not self.generated_images:
            console.print("[yellow]⚠️ Need both real and generated images for comparison[/yellow]")
            return None
        
        console.print("\n[bold]🎨 Creating comparison plots...[/bold]")
        
        n_samples = min(8, len(self.real_images), len(self.generated_images))
        
        fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 3, 12))
        fig.suptitle('Real vs Generated CARS Images Analysis', fontsize=16, fontweight='bold')
        
        for i in range(n_samples):
            # Real images
            axes[0, i].imshow(self.real_images[i], cmap='gray')
            axes[0, i].set_title(f'Real {i+1}')
            axes[0, i].axis('off')
            
            # Generated images
            axes[1, i].imshow(self.generated_images[i], cmap='gray')
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].axis('off')
            
            # Enhanced contrast (real)
            real_enhanced = ImageEnhance.Contrast(Image.fromarray(self.real_images[i])).enhance(3.0)
            axes[2, i].imshow(np.array(real_enhanced), cmap='gray')
            axes[2, i].set_title('Real Enhanced')
            axes[2, i].axis('off')
            
            # Enhanced contrast (generated)
            gen_enhanced = ImageEnhance.Contrast(Image.fromarray(self.generated_images[i])).enhance(3.0)
            axes[3, i].imshow(np.array(gen_enhanced), cmap='gray')
            axes[3, i].set_title('Gen Enhanced')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f"image_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Saved comparison plots to: {plot_path}[/green]")
        
        return plot_path


# ==================== Main Analyzer ====================

class CARSModelAnalyzer:
    """Main analyzer that orchestrates all analysis components"""
    
    def __init__(self, output_dir: Path = Path("outputs/analysis")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.training_monitor = TrainingMonitor(output_dir)
        self.image_analyzer = ImageAnalyzer(output_dir)
        
        # Results container
        self.results = AnalysisResults()
    
    def analyze_training(self, experiment_dir: Path) -> bool:
        """Analyze training progress from TensorBoard logs"""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                    TRAINING ANALYSIS                       [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        
        # Find event files
        event_files = self.training_monitor.find_tensorboard_logs(experiment_dir)
        if not event_files:
            return False
        
        # Parse the most recent event file
        metrics = self.training_monitor.parse_event_file(event_files[0])
        if not metrics:
            return False
        
        self.results.training_metrics = metrics
        
        # Analyze training health
        health_analysis = self.training_monitor.analyze_training_health(metrics)
        self.results.training_health_score = health_analysis['health_score']
        self.results.training_stable = health_analysis['stable']
        self.results.convergence_status = health_analysis['convergence']
        self.results.recommendations.extend(health_analysis['recommendations'])
        
        # Create plots
        plot_path = self.training_monitor.create_training_plots(metrics)
        if plot_path:
            self.results.plots['training'] = plot_path
        
        # Display health summary
        console.print("\n[bold]Training Health Summary:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Health Score", f"{health_analysis['health_score']}/100")
        table.add_row("Stable", "✅ Yes" if health_analysis['stable'] else "❌ No")
        table.add_row("Convergence", health_analysis['convergence'].replace('_', ' ').title())
        table.add_row("D Loss (mean ± std)", f"{health_analysis['d_loss_mean']:.3f} ± {health_analysis['d_loss_std']:.3f}")
        table.add_row("G Loss (mean ± std)", f"{health_analysis['g_loss_mean']:.3f} ± {health_analysis['g_loss_std']:.3f}")
        
        console.print(table)
        
        return True
    
    def analyze_images(self, generated_dir: Optional[Path] = None, 
                      real_dir: Optional[Path] = None,
                      checkpoint_path: Optional[Path] = None) -> bool:
        """Analyze image quality"""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                     IMAGE ANALYSIS                         [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        
        # Load images from checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            self.image_analyzer.load_from_checkpoint(checkpoint_path)
        
        # Load images from directories
        n_gen, n_real = self.image_analyzer.load_images(generated_dir, real_dir)
        
        if n_gen == 0 and n_real == 0:
            console.print("[red]❌ No images loaded for analysis[/red]")
            return False
        
        # Analyze image quality
        quality_metrics = self.image_analyzer.analyze_image_quality()
        self.results.image_quality_metrics = quality_metrics
        
        # Detect issues
        self._detect_image_issues(quality_metrics)
        
        # Create comparison plots
        plot_path = self.image_analyzer.create_comparison_plots()
        if plot_path:
            self.results.plots['comparison'] = plot_path
        
        return True
    
    def _detect_image_issues(self, metrics: Dict[str, float]):
        """Detect issues in generated images"""
        # Check intensity mismatch
        if 'real_mean' in metrics and 'generated_mean' in metrics:
            mean_diff = abs(metrics['real_mean'] - metrics['generated_mean'])
            if mean_diff > 50:
                self.results.issues.append("⚠️ Severe intensity mismatch between real and generated")
                self.results.recommendations.append("Check normalization in data pipeline and model output")
        
        # Check dynamic range
        if 'real_range' in metrics and 'generated_range' in metrics:
            if metrics['generated_range'] < metrics['real_range'] * 0.5:
                self.results.issues.append("⚠️ Low dynamic range in generated images")
                self.results.recommendations.append("Increase model capacity or adjust loss weights")
        
        # Check structural features
        if 'real_follicle_count' in metrics and 'generated_follicle_count' in metrics:
            if metrics['generated_follicle_count'] < metrics['real_follicle_count'] * 0.3:
                self.results.issues.append("⚠️ Missing characteristic follicular structures")
                self.results.recommendations.append("Model may need more training or architectural changes")
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        console.print("\n[bold]📝 Generating analysis report...[/bold]")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"analysis_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# CARS-FASTGAN Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training section
            if self.results.training_metrics:
                f.write("## Training Analysis\n\n")
                f.write(f"- **Total Steps**: {self.results.training_metrics[-1].step}\n")
                f.write(f"- **Epochs**: {self.results.training_metrics[-1].epoch}\n")
                f.write(f"- **Health Score**: {self.results.training_health_score}/100\n")
                f.write(f"- **Stable**: {'Yes' if self.results.training_stable else 'No'}\n")
                f.write(f"- **Convergence**: {self.results.convergence_status}\n\n")
            
            # Image analysis section
            if self.results.image_quality_metrics:
                f.write("## Image Quality Analysis\n\n")
                f.write("| Metric | Real | Generated | Difference |\n")
                f.write("|--------|------|-----------|------------|\n")
                
                for metric in ['mean', 'std', 'range', 'noise', 'follicle_count']:
                    real = self.results.image_quality_metrics.get(f'real_{metric}', 0)
                    gen = self.results.image_quality_metrics.get(f'generated_{metric}', 0)
                    diff = abs(real - gen)
                    f.write(f"| {metric} | {real:.2f} | {gen:.2f} | {diff:.2f} |\n")
                f.write("\n")
            
            # Issues and recommendations
            if self.results.issues:
                f.write("## Issues Detected\n\n")
                for issue in self.results.issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            if self.results.recommendations:
                f.write("## Recommendations\n\n")
                for rec in self.results.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Generated files
            if self.results.plots:
                f.write("## Generated Plots\n\n")
                for name, path in self.results.plots.items():
                    f.write(f"- **{name}**: `{path}`\n")
        
        self.results.report_path = report_path
        console.print(f"[green]✅ Report saved to: {report_path}[/green]")


# ==================== CLI Interface ====================

def main():
    parser = argparse.ArgumentParser(
        description="CARS-FASTGAN Model Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze training logs
  python scripts/cars_model_analyzer.py --mode monitor --experiment_dir experiments/logs
  
  # Analyze generated images
  python scripts/cars_model_analyzer.py --mode analyze --generated_dir outputs/samples --real_dir data/real
  
  # Full analysis (training + images)
  python scripts/cars_model_analyzer.py --mode full --experiment_dir experiments/logs --real_dir data/real
  
  # Analyze from checkpoint
  python scripts/cars_model_analyzer.py --mode checkpoint --checkpoint_path model.ckpt --real_dir data/real
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['monitor', 'analyze', 'full', 'checkpoint'],
                       help='Analysis mode')
    parser.add_argument('--experiment_dir', type=Path, default=Path('experiments/logs'),
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--generated_dir', type=Path, default=None,
                       help='Directory containing generated images')
    parser.add_argument('--real_dir', type=Path, default=None,
                       help='Directory containing real images')
    parser.add_argument('--checkpoint_path', type=Path, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/analysis'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print header
    console.print("\n[bold magenta]╔══════════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║             CARS-FASTGAN MODEL ANALYZER                  ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════════════════════╝[/bold magenta]")
    console.print(f"\n[bold]Mode:[/bold] {args.mode}")
    console.print(f"[bold]Output:[/bold] {args.output_dir}")
    
    # Create analyzer
    analyzer = CARSModelAnalyzer(args.output_dir)
    
    # Run analysis based on mode
    success = False
    
    if args.mode == 'monitor':
        success = analyzer.analyze_training(args.experiment_dir)
    
    elif args.mode == 'analyze':
        success = analyzer.analyze_images(args.generated_dir, args.real_dir)
    
    elif args.mode == 'full':
        training_success = analyzer.analyze_training(args.experiment_dir)
        image_success = analyzer.analyze_images(args.generated_dir, args.real_dir)
        success = training_success or image_success
    
    elif args.mode == 'checkpoint':
        if not args.checkpoint_path:
            console.print("[red]❌ Checkpoint path required for checkpoint mode[/red]")
        else:
            success = analyzer.analyze_images(checkpoint_path=args.checkpoint_path, real_dir=args.real_dir)
    
    # Generate report
    if success:
        analyzer.generate_report()
        
        # Print summary
        console.print("\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        console.print("[bold green]                    ANALYSIS COMPLETE                       [/bold green]")
        console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • Health Score: {analyzer.results.training_health_score}/100")
        console.print(f"  • Issues Found: {len(analyzer.results.issues)}")
        console.print(f"  • Recommendations: {len(analyzer.results.recommendations)}")
        
        if analyzer.results.report_path:
            console.print(f"\n[bold]📄 Full report:[/bold] {analyzer.results.report_path}")
    else:
        console.print("\n[red]❌ Analysis failed - check error messages above[/red]")


if __name__ == "__main__":
    main()