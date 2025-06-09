#!/usr/bin/env python3
"""
CARS-FASTGAN Model Analyzer
Comprehensive tool for monitoring training, analyzing generated images, and providing insights

Consolidates:
- monitor_training.py functionality
- analyze_generated_images.py functionality
- Enhanced evaluation and diagnostic features

Usage:
    # Real-time monitoring
    python cars_model_analyzer.py --mode monitor --experiment_dir experiments
    
    # Post-training analysis
    python cars_model_analyzer.py --mode analyze --generated_dir outputs/samples --real_dir data/processed
    
    # Full analysis (monitor + analyze)
    python cars_model_analyzer.py --mode full --experiment_dir experiments --real_dir data/processed
    
    # Interactive dashboard
    python cars_model_analyzer.py --mode dashboard --experiment_dir experiments
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from sklearn.metrics import mean_squared_error

# Image processing
from PIL import Image, ImageEnhance
import cv2
from skimage import filters, feature, measure, exposure
from skimage.feature import local_binary_pattern

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Deep learning
import torch
import torch.nn.functional as F

# Progress tracking
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

# For web dashboard (optional imports)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")


# ==================== Configuration ====================

console = Console()


class AnalysisMode(Enum):
    """Analysis modes"""
    MONITOR = "monitor"
    ANALYZE = "analyze"
    FULL = "full"
    DASHBOARD = "dashboard"


@dataclass
class AnalyzerConfig:
    """Configuration for the analyzer"""
    # Mode
    mode: AnalysisMode = AnalysisMode.FULL
    
    # Paths
    experiment_dir: Path = Path("experiments")
    real_dir: Optional[Path] = None
    generated_dir: Optional[Path] = None
    output_dir: Path = Path("outputs/analysis")
    
    # Monitoring settings
    refresh_interval: int = 30
    continuous_monitoring: bool = True
    
    # Analysis settings
    sample_size: int = 50
    comprehensive_analysis: bool = True
    
    # Visualization
    plot_format: str = "png"
    plot_dpi: int = 150
    interactive_plots: bool = False
    
    # Reporting
    report_formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    
    # Integration
    enable_hooks: bool = True
    webhook_url: Optional[str] = None
    
    # Web dashboard
    dashboard_port: int = 8501
    dashboard_host: str = "localhost"


@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    epoch: int
    step: int
    timestamp: datetime
    d_loss: float
    g_loss: float
    d_loss_real: float
    d_loss_fake: float
    lr_g: float
    lr_d: float
    val_d_loss: Optional[float] = None
    val_g_loss: Optional[float] = None
    fid_score: Optional[float] = None
    epoch_time: Optional[float] = None
    is_score: Optional[float] = None
    lpips_score: Optional[float] = None


@dataclass
class AnalysisResults:
    """Container for all analysis results"""
    # Training metrics
    training_metrics: List[TrainingMetrics] = field(default_factory=list)
    health_score: float = 100.0
    training_stable: bool = True
    convergence_status: str = "training"
    
    # Image quality
    dynamic_range_stats: Dict[str, float] = field(default_factory=dict)
    noise_pattern_stats: Dict[str, float] = field(default_factory=dict)
    frequency_content_stats: Dict[str, float] = field(default_factory=dict)
    
    # Detected issues
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Artifacts
    plots: Dict[str, Path] = field(default_factory=dict)
    reports: Dict[str, Path] = field(default_factory=dict)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analyzer_version: str = "1.0.0"


# ==================== Base Components ====================

class BaseAnalyzer:
    """Base class for analysis components"""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.console = console
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.config.output_dir / "plots"
        self.reports_dir = self.config.output_dir / "reports"
        self.plots_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def log(self, message: str, style: str = "info"):
        """Log a message with style"""
        styles = {
            "info": "[blue]ℹ️[/blue]",
            "success": "[green]✅[/green]",
            "warning": "[yellow]⚠️[/yellow]",
            "error": "[red]❌[/red]",
            "debug": "[grey]🐛[/grey]"
        }
        prefix = styles.get(style, "")
        self.console.print(f"{prefix} {message}")
    
    def save_plot(self, fig: plt.Figure, name: str) -> Path:
        """Save a matplotlib figure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{self.config.plot_format}"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        
        return filepath


# ==================== Monitoring Module ====================

class TrainingMonitor(BaseAnalyzer):
    """Real-time training monitor component"""
    
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.logs_dir = config.experiment_dir / "logs"
        self.checkpoints_dir = config.experiment_dir / "checkpoints"
        
        if not TENSORBOARD_AVAILABLE:
            self.log("TensorBoard not available. Monitoring will be limited.", "warning")
    
    def find_latest_experiment(self) -> Optional[Path]:
        """Find the most recent experiment"""
        if not self.logs_dir.exists():
            return None
        
        log_dirs = [d for d in self.logs_dir.iterdir() if d.is_dir()]
        
        if not log_dirs:
            return None
        
        latest_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
        self.log(f"Found experiment: {latest_dir.name}")
        
        return latest_dir
    
    def parse_tensorboard_logs(self, log_dir: Path) -> List[TrainingMetrics]:
        """Parse TensorBoard event files to extract metrics"""
        if not TENSORBOARD_AVAILABLE:
            return []
        
        metrics = []
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        
        if not event_files:
            self.log("No event files found", "warning")
            return metrics
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                # Get all scalar tags
                scalar_tags = ea.Tags()['scalars']
                
                # Create a mapping of steps to metrics
                step_metrics = {}
                
                for tag in scalar_tags:
                    for event in ea.Scalars(tag):
                        step = event.step
                        
                        if step not in step_metrics:
                            step_metrics[step] = TrainingMetrics(
                                epoch=0,
                                step=step,
                                timestamp=datetime.fromtimestamp(event.wall_time),
                                d_loss=0.0,
                                g_loss=0.0,
                                d_loss_real=0.0,
                                d_loss_fake=0.0,
                                lr_g=1e-4,
                                lr_d=1e-4
                            )
                        
                        metric = step_metrics[step]
                        
                        # Map tags to metric fields
                        tag_lower = tag.lower()
                        if 'epoch' in tag_lower:
                            metric.epoch = int(event.value)
                        elif 'd_loss' in tag_lower and 'val' not in tag_lower:
                            if 'real' in tag_lower:
                                metric.d_loss_real = event.value
                            elif 'fake' in tag_lower:
                                metric.d_loss_fake = event.value
                            else:
                                metric.d_loss = event.value
                        elif 'g_loss' in tag_lower and 'val' not in tag_lower:
                            metric.g_loss = event.value
                        elif 'val' in tag_lower:
                            if 'd_loss' in tag_lower:
                                metric.val_d_loss = event.value
                            elif 'g_loss' in tag_lower:
                                metric.val_g_loss = event.value
                        elif 'fid' in tag_lower:
                            metric.fid_score = event.value
                        elif 'is_score' in tag_lower or 'inception' in tag_lower:
                            metric.is_score = event.value
                        elif 'lpips' in tag_lower:
                            metric.lpips_score = event.value
                        elif 'lr' in tag_lower:
                            if 'g' in tag_lower or 'gen' in tag_lower:
                                metric.lr_g = event.value
                            elif 'd' in tag_lower or 'disc' in tag_lower:
                                metric.lr_d = event.value
                
                metrics.extend(step_metrics.values())
                
            except Exception as e:
                self.log(f"Error parsing {event_file}: {e}", "error")
                continue
        
        # Sort by epoch
        metrics.sort(key=lambda x: (x.epoch, x.step))
        
        self.log(f"Parsed {len(metrics)} metric records", "success")
        return metrics
    
    def analyze_training_progress(self, metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """Analyze training progress and health"""
        if not metrics:
            return {
                'health_score': 0,
                'training_stable': False,
                'convergence_status': 'no_data',
                'recommendations': ['No metrics available for analysis']
            }
        
        analysis = {
            'total_epochs': metrics[-1].epoch if metrics else 0,
            'current_epoch': metrics[-1].epoch if metrics else 0,
            'training_stable': True,
            'convergence_status': 'training',
            'recommendations': [],
            'health_score': 100
        }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([{
            'epoch': m.epoch,
            'd_loss': m.d_loss,
            'g_loss': m.g_loss,
            'd_loss_real': m.d_loss_real,
            'd_loss_fake': m.d_loss_fake,
            'val_d_loss': m.val_d_loss,
            'val_g_loss': m.val_g_loss,
            'fid_score': m.fid_score,
            'lr_g': m.lr_g,
            'lr_d': m.lr_d
        } for m in metrics])
        
        if len(df) < 10:
            analysis['recommendations'].append("Too few epochs for meaningful analysis")
            return analysis
        
        # Analyze loss trends
        recent_window = min(50, len(df) // 4)
        recent_df = df.tail(recent_window)
        
        # Check for loss explosion
        if recent_df['d_loss'].max() > 10 or recent_df['g_loss'].max() > 10:
            analysis['training_stable'] = False
            analysis['health_score'] -= 30
            analysis['recommendations'].append("⚠️ Loss explosion detected - reduce learning rate")
        
        # Check for loss oscillation
        d_loss_std = recent_df['d_loss'].std()
        g_loss_std = recent_df['g_loss'].std()
        
        if d_loss_std > 2.0 or g_loss_std > 2.0:
            analysis['health_score'] -= 20
            analysis['recommendations'].append("⚠️ High loss oscillation - training unstable")
        
        # Check discriminator/generator balance
        recent_d_mean = recent_df['d_loss'].mean()
        recent_g_mean = recent_df['g_loss'].mean()
        
        if recent_d_mean < 0.1:
            analysis['health_score'] -= 25
            analysis['recommendations'].append("⚠️ Discriminator too strong - generator may collapse")
        elif recent_g_mean < 0.1:
            analysis['health_score'] -= 25
            analysis['recommendations'].append("⚠️ Generator too strong - discriminator failing")
        
        # Check for convergence
        if len(df) > 100:
            convergence_window = min(100, len(df) // 3)
            convergence_df = df.tail(convergence_window)
            
            d_loss_trend = np.polyfit(range(len(convergence_df)), convergence_df['d_loss'], 1)[0]
            g_loss_trend = np.polyfit(range(len(convergence_df)), convergence_df['g_loss'], 1)[0]
            
            if abs(d_loss_trend) < 0.001 and abs(g_loss_trend) < 0.001:
                analysis['convergence_status'] = 'converged'
                analysis['recommendations'].append("✅ Training appears to have converged")
            elif d_loss_trend > 0.01 and g_loss_trend > 0.01:
                analysis['convergence_status'] = 'diverging'
                analysis['health_score'] -= 40
                analysis['recommendations'].append("❌ Training appears to be diverging")
        
        # FID score analysis
        if 'fid_score' in df.columns and df['fid_score'].notna().any():
            fid_scores = df['fid_score'].dropna()
            if len(fid_scores) > 5:
                latest_fid = fid_scores.iloc[-1]
                best_fid = fid_scores.min()
                
                analysis['latest_fid'] = latest_fid
                analysis['best_fid'] = best_fid
                
                if latest_fid < 50:
                    analysis['recommendations'].append(f"✅ Excellent FID score: {latest_fid:.2f}")
                elif latest_fid < 100:
                    analysis['recommendations'].append(f"👍 Good FID score: {latest_fid:.2f}")
                else:
                    analysis['recommendations'].append(f"📈 FID needs improvement: {latest_fid:.2f}")
        
        # Overall health assessment
        if analysis['health_score'] > 80:
            analysis['recommendations'].append("✅ Training is healthy")
        elif analysis['health_score'] > 60:
            analysis['recommendations'].append("⚠️ Training has minor issues")
        else:
            analysis['recommendations'].append("❌ Training has serious issues - consider stopping")
        
        return analysis
    
    def create_training_plots(self, metrics: List[TrainingMetrics]) -> Optional[Path]:
        """Create comprehensive training plots"""
        if not metrics:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'epoch': m.epoch,
            'd_loss': m.d_loss,
            'g_loss': m.g_loss,
            'd_loss_real': m.d_loss_real,
            'd_loss_fake': m.d_loss_fake,
            'val_d_loss': m.val_d_loss,
            'val_g_loss': m.val_g_loss,
            'fid_score': m.fid_score,
            'lr_g': m.lr_g,
            'lr_d': m.lr_d
        } for m in metrics])
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Plot 1: Training Losses
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['epoch'], df['d_loss'], label='D Loss', alpha=0.8, linewidth=2)
        ax1.plot(df['epoch'], df['g_loss'], label='G Loss', alpha=0.8, linewidth=2)
        if df['val_d_loss'].notna().any():
            ax1.plot(df['epoch'], df['val_d_loss'], label='Val D Loss', 
                    linestyle='--', alpha=0.8, linewidth=2)
        if df['val_g_loss'].notna().any():
            ax1.plot(df['epoch'], df['val_g_loss'], label='Val G Loss', 
                    linestyle='--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Discriminator Components
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['epoch'], df['d_loss_real'], label='D Loss Real', alpha=0.8, linewidth=2)
        ax2.plot(df['epoch'], df['d_loss_fake'], label='D Loss Fake', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Discriminator Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FID Score
        ax3 = fig.add_subplot(gs[0, 2])
        if df['fid_score'].notna().any():
            fid_data = df[df['fid_score'].notna()]
            ax3.plot(fid_data['epoch'], fid_data['fid_score'], 
                    marker='o', markersize=8, alpha=0.8, linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('FID Score')
            ax3.set_title('FID Score Progress (Lower is Better)')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(fid_data) > 3:
                z = np.polyfit(fid_data['epoch'], fid_data['fid_score'], 2)
                p = np.poly1d(z)
                ax3.plot(fid_data['epoch'], p(fid_data['epoch']), 
                        'r--', alpha=0.5, label='Trend')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No FID data\navailable', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('FID Score Progress')
        
        # Plot 4: Learning Rates
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(df['epoch'], df['lr_g'], label='Generator LR', alpha=0.8, linewidth=2)
        ax4.plot(df['epoch'], df['lr_d'], label='Discriminator LR', alpha=0.8, linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Plot 5: Loss Smoothed (Moving Average)
        ax5 = fig.add_subplot(gs[1, 1])
        window = min(20, len(df) // 10) if len(df) > 20 else 1
        if window > 1:
            d_loss_smooth = df['d_loss'].rolling(window=window, center=True).mean()
            g_loss_smooth = df['g_loss'].rolling(window=window, center=True).mean()
            ax5.plot(df['epoch'], d_loss_smooth, label=f'D Loss (MA {window})', 
                    alpha=0.8, linewidth=2)
            ax5.plot(df['epoch'], g_loss_smooth, label=f'G Loss (MA {window})', 
                    alpha=0.8, linewidth=2)
        else:
            ax5.plot(df['epoch'], df['d_loss'], label='D Loss', alpha=0.8, linewidth=2)
            ax5.plot(df['epoch'], df['g_loss'], label='G Loss', alpha=0.8, linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Smoothed Loss Trends')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Training Health Score
        ax6 = fig.add_subplot(gs[1, 2])
        if len(df) > 10:
            window_size = min(10, len(df) // 5)
            health_scores = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                
                # Health based on loss variance and magnitude
                d_var = window_data['d_loss'].var()
                g_var = window_data['g_loss'].var()
                d_mean = window_data['d_loss'].mean()
                g_mean = window_data['g_loss'].mean()
                
                # Simple health score (higher is better)
                health = 100 / (1 + d_var + g_var + abs(d_mean - g_mean))
                health_scores.append(min(100, health * 10))  # Scale and cap
            
            epochs_subset = df['epoch'].iloc[window_size:]
            ax6.plot(epochs_subset, health_scores, color='green', alpha=0.8, linewidth=3)
            ax6.fill_between(epochs_subset, health_scores, alpha=0.3, color='green')
            ax6.set_ylim(0, 100)
            ax6.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Good')
            ax6.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Warning')
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Health Score')
        ax6.set_title('Training Health Score')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7-9: Loss ratio and balance metrics
        ax7 = fig.add_subplot(gs[2, 0])
        if len(df) > 1:
            loss_ratio = df['g_loss'] / (df['d_loss'] + 1e-8)
            ax7.plot(df['epoch'], loss_ratio, alpha=0.8, linewidth=2)
            ax7.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('G/D Loss Ratio')
            ax7.set_title('Generator/Discriminator Balance')
            ax7.set_yscale('log')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Additional metrics if available
        ax8 = fig.add_subplot(gs[2, 1])
        metrics_to_plot = []
        if df.get('is_score') is not None and df['is_score'].notna().any():
            metrics_to_plot.append(('is_score', 'Inception Score', 'blue'))
        if df.get('lpips_score') is not None and df['lpips_score'].notna().any():
            metrics_to_plot.append(('lpips_score', 'LPIPS Score', 'red'))
        
        if metrics_to_plot:
            for metric, label, color in metrics_to_plot:
                data = df[df[metric].notna()]
                ax8.plot(data['epoch'], data[metric], label=label, 
                        color=color, marker='o', alpha=0.8)
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Score')
            ax8.set_title('Additional Metrics')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No additional\nmetrics available', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=14)
            ax8.set_title('Additional Metrics')
        
        # Plot 9: Training time per epoch
        ax9 = fig.add_subplot(gs[2, 2])
        if len(metrics) > 1:
            epoch_times = []
            for i in range(1, len(metrics)):
                if metrics[i].epoch != metrics[i-1].epoch:
                    time_diff = (metrics[i].timestamp - metrics[i-1].timestamp).total_seconds() / 60
                    if time_diff < 1000:  # Reasonable epoch time
                        epoch_times.append((metrics[i].epoch, time_diff))
            
            if epoch_times:
                epochs, times = zip(*epoch_times)
                ax9.plot(epochs, times, marker='o', alpha=0.8)
                ax9.set_xlabel('Epoch')
                ax9.set_ylabel('Time (minutes)')
                ax9.set_title('Training Time per Epoch')
                ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_plot(fig, 'training_progress')
        return plot_path


# ==================== Image Analysis Module ====================

class ImageAnalyzer(BaseAnalyzer):
    """Comprehensive image analysis component"""
    
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.generated_images = []
        self.real_images = []
    
    def load_images(self, generated_dir: Optional[Path] = None, 
                   real_dir: Optional[Path] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images for analysis"""
        # Use config paths if not provided
        generated_dir = generated_dir or self.config.generated_dir
        real_dir = real_dir or self.config.real_dir
        
        # Load generated images
        if generated_dir and generated_dir.exists():
            self.generated_images = self._load_images_from_dir(generated_dir, "generated")
        else:
            self.log("No generated images directory found", "warning")
        
        # Load real images
        if real_dir and real_dir.exists():
            self.real_images = self._load_images_from_dir(real_dir, "real")
        else:
            self.log("No real images directory found", "warning")
        
        return self.generated_images, self.real_images
    
    def _load_images_from_dir(self, image_dir: Path, image_type: str) -> List[np.ndarray]:
        """Load images from directory"""
        image_paths = []
        
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_paths.extend(list(image_dir.glob(ext)))
            image_paths.extend(list(image_dir.rglob(ext)))
        
        if not image_paths:
            self.log(f"No images found in {image_dir}", "warning")
            return []
        
        images = []
        max_images = min(self.config.sample_size, len(image_paths))
        
        for img_path in tqdm(image_paths[:max_images], 
                           desc=f"Loading {image_type} images"):
            try:
                img = np.array(Image.open(img_path).convert('L'))
                images.append(img)
            except Exception as e:
                self.log(f"Error loading {img_path}: {e}", "error")
        
        self.log(f"Loaded {len(images)} {image_type} images", "success")
        return images
    
    def analyze_dynamic_range(self) -> Dict[str, float]:
        """Analyze dynamic range and intensity distributions"""
        self.log("Analyzing dynamic range...", "info")
        
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
        
        stats = {**real_stats, **gen_stats}
        
        # Create comparison plot
        if self.real_images and self.generated_images:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Dynamic Range Analysis', fontsize=16)
            
            real_pixels = np.concatenate([img.flatten() for img in self.real_images])
            gen_pixels = np.concatenate([img.flatten() for img in self.generated_images])
            
            # Histograms
            axes[0, 0].hist(real_pixels, bins=100, alpha=0.7, label='Real', density=True)
            axes[0, 0].hist(gen_pixels, bins=100, alpha=0.7, label='Generated', density=True)
            axes[0, 0].set_title('Pixel Intensity Histograms')
            axes[0, 0].set_xlabel('Pixel Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            
            # Cumulative histograms
            axes[0, 1].hist(real_pixels, bins=100, alpha=0.7, label='Real', 
                          density=True, cumulative=True)
            axes[0, 1].hist(gen_pixels, bins=100, alpha=0.7, label='Generated', 
                          density=True, cumulative=True)
            axes[0, 1].set_title('Cumulative Histograms')
            axes[0, 1].set_xlabel('Pixel Value')
            axes[0, 1].set_ylabel('Cumulative Density')
            axes[0, 1].legend()
            
            # Box plots
            data_for_box = [real_pixels[::1000], gen_pixels[::1000]]
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
            
            # Q-Q plot
            real_sorted = np.sort(real_pixels[::1000])
            gen_sorted = np.sort(gen_pixels[::1000])
            min_len = min(len(real_sorted), len(gen_sorted))
            axes[1, 1].scatter(real_sorted[:min_len], gen_sorted[:min_len], alpha=0.5, s=1)
            axes[1, 1].plot([0, 255], [0, 255], 'r--', label='Perfect Match')
            axes[1, 1].set_title('Q-Q Plot: Real vs Generated')
            axes[1, 1].set_xlabel('Real Quantiles')
            axes[1, 1].set_ylabel('Generated Quantiles')
            axes[1, 1].legend()
            
            # Dynamic range comparison
            ranges = [real_stats['real_dynamic_range'], gen_stats['generated_dynamic_range']]
            colors = ['blue', 'orange']
            bars = axes[1, 2].bar(['Real', 'Generated'], ranges, alpha=0.7, color=colors)
            axes[1, 2].set_title('Dynamic Range Comparison')
            axes[1, 2].set_ylabel('Dynamic Range')
            
            # Add value labels on bars
            for bar, value in zip(bars, ranges):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.save_plot(fig, 'dynamic_range_analysis')
            stats['plot_path'] = str(plot_path)
        
        self.log("Dynamic range analysis complete", "success")
        return stats
    
    def detect_noise_patterns(self) -> Dict[str, float]:
        """Detect and analyze noise patterns"""
        self.log("Detecting noise patterns...", "info")
        
        def analyze_noise(images, name):
            if not images:
                return {}
            
            noise_metrics = {
                'high_freq_energy': [],
                'edge_density': [],
                'texture_uniformity': [],
                'checkerboard_score': [],
                'salt_pepper_score': [],
                'fft_checkerboard_score': [],
                'structural_complexity': [],
                'follicle_like_structures': []
            }
            
            for img in tqdm(images[:20], desc=f"Analyzing {name} noise"):
                # High frequency energy
                fft = np.fft.fft2(img)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                h, w = img.shape
                center_h, center_w = h // 2, w // 2
                mask = np.zeros_like(magnitude)
                cv2.circle(mask, (center_w, center_h), min(h, w) // 4, 1, -1)
                
                high_freq = magnitude * (1 - mask)
                noise_metrics['high_freq_energy'].append(np.sum(high_freq) / np.sum(magnitude))
                
                # Edge density
                edges = cv2.Canny(img.astype(np.uint8), 50, 150)
                noise_metrics['edge_density'].append(np.mean(edges > 0))
                
                # Texture uniformity (using LBP)
                lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                uniformity = np.sum(hist ** 2)
                noise_metrics['texture_uniformity'].append(uniformity)
                
                # Checkerboard detection (spatial)
                kernel = np.array([[1, -1], [-1, 1]])
                conv = cv2.filter2D(img, -1, kernel)
                checkerboard = np.var(conv)
                noise_metrics['checkerboard_score'].append(checkerboard)
                
                # Salt and pepper noise
                img_float = img.astype(float)
                noise_est = np.abs(img_float - cv2.medianBlur(img, 3))
                salt_pepper = np.mean(noise_est > 50)
                noise_metrics['salt_pepper_score'].append(salt_pepper)
                
                # FFT-based checkerboard detection
                fft_log = np.log(magnitude + 1)
                peaks = feature.peak_local_max(fft_log, min_distance=5, num_peaks=10)
                
                if len(peaks) > 0:
                    distances = []
                    for peak in peaks:
                        dist = np.sqrt((peak[0] - center_h)**2 + (peak[1] - center_w)**2)
                        distances.append(dist)
                    
                    distances = np.array(distances)
                    periodic_score = np.std(distances) / (np.mean(distances) + 1e-8)
                else:
                    periodic_score = 0
                
                noise_metrics['fft_checkerboard_score'].append(periodic_score)
                
                # Structural complexity
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
                structural_complexity = -np.sum(lbp_hist * np.log(lbp_hist + 1e-8))
                noise_metrics['structural_complexity'].append(structural_complexity)
                
                # Follicle-like structure detection
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
        if self.real_images and self.generated_images:
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            fig.suptitle('Comprehensive Noise & Structure Analysis', fontsize=16)
            
            metrics = [
                'high_freq_energy', 'edge_density', 'texture_uniformity',
                'checkerboard_score', 'salt_pepper_score', 'fft_checkerboard_score',
                'structural_complexity', 'follicle_like_structures'
            ]
            
            for i, metric in enumerate(metrics):
                if i < 8:  # We have 8 metrics but 9 plots
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
                            bars[1].set_color('red')
                        elif metric == 'follicle_like_structures' and noise_stats[gen_key] < noise_stats[real_key] * 0.5:
                            bars[1].set_color('orange')
                        
                        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                        axes[row, col].set_ylabel('Score')
                        
                        # Add value labels
                        for bar, value in zip(bars, values):
                            height = bar.get_height()
                            axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                              f'{value:.3f}', ha='center', va='bottom')
            
            # Use the last subplot for overall noise score
            ax_overall = axes[2, 2]
            if noise_stats:
                # Calculate overall noise scores
                real_noise_score = np.mean([v for k, v in noise_stats.items() 
                                           if k.startswith('real_') and '_mean' in k])
                gen_noise_score = np.mean([v for k, v in noise_stats.items() 
                                         if k.startswith('generated_') and '_mean' in k])
                
                bars = ax_overall.bar(['Real', 'Generated'], 
                                    [real_noise_score, gen_noise_score], 
                                    alpha=0.7, color=['blue', 'orange'])
                ax_overall.set_title('Overall Noise Score')
                ax_overall.set_ylabel('Average Score')
                
                for bar, value in zip(bars, [real_noise_score, gen_noise_score]):
                    height = bar.get_height()
                    ax_overall.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.save_plot(fig, 'noise_pattern_analysis')
            noise_stats['plot_path'] = str(plot_path)
        
        self.log("Noise pattern analysis complete", "success")
        return noise_stats
    
    def analyze_frequency_content(self) -> Dict[str, float]:
        """Analyze frequency content and artifacts"""
        self.log("Analyzing frequency content...", "info")
        
        def get_frequency_stats(images, name):
            if not images:
                return {}
            
            freq_stats = {
                'low_freq_energy': [],
                'mid_freq_energy': [],
                'high_freq_energy': [],
                'freq_concentration': [],
                'spectral_entropy': []
            }
            
            for img in tqdm(images[:10], desc=f"Analyzing {name} frequency"):
                # FFT analysis
                f_transform = np.fft.fft2(img)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                
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
                
                # Frequency concentration
                mag_flat = magnitude_spectrum.flatten()
                mag_sorted = np.sort(mag_flat)[::-1]
                cum_energy = np.cumsum(mag_sorted)
                total = cum_energy[-1]
                
                # Find how many frequencies contain 90% of energy
                idx_90 = np.where(cum_energy >= 0.9 * total)[0][0]
                concentration = idx_90 / len(mag_flat)
                freq_stats['freq_concentration'].append(concentration)
                
                # Spectral entropy
                mag_norm = magnitude_spectrum / (np.sum(magnitude_spectrum) + 1e-10)
                entropy = -np.sum(mag_norm * np.log(mag_norm + 1e-10))
                freq_stats['spectral_entropy'].append(entropy)
            
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
            
            # Show sample FFTs
            for i, (images, name) in enumerate([(self.real_images, 'Real'), 
                                                (self.generated_images, 'Generated')]):
                if images:
                    img = images[0]
                    
                    # Original image
                    axes[i, 0].imshow(img, cmap='gray')
                    axes[i, 0].set_title(f'{name} Sample')
                    axes[i, 0].axis('off')
                    
                    # FFT magnitude
                    f_transform = np.fft.fft2(img)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude = np.log(np.abs(f_shift) + 1)
                    
                    axes[i, 1].imshow(magnitude, cmap='hot')
                    axes[i, 1].set_title(f'{name} FFT Magnitude')
                    axes[i, 1].axis('off')
                    
                    # Radial average
                    h, w = img.shape
                    center = (h // 2, w // 2)
                    Y, X = np.ogrid[:h, :w]
                    r = np.sqrt((X - center[1])**2 + (Y - center[0])**2).astype(int)
                    
                    radial_prof = ndimage.mean(magnitude, labels=r, 
                                              index=np.arange(0, r.max()))
                    
                    axes[i, 2].plot(radial_prof)
                    axes[i, 2].set_xlabel('Frequency')
                    axes[i, 2].set_ylabel('Magnitude')
                    axes[i, 2].set_title(f'{name} Radial Profile')
                    axes[i, 2].grid(True, alpha=0.3)
                    
                    # Frequency band energies
                    bands = ['Low', 'Mid', 'High']
                    energies = [
                        freq_stats.get(f'{name.lower()}_low_freq_energy_mean', 0),
                        freq_stats.get(f'{name.lower()}_mid_freq_energy_mean', 0),
                        freq_stats.get(f'{name.lower()}_high_freq_energy_mean', 0)
                    ]
                    
                    bars = axes[i, 3].bar(bands, energies, alpha=0.7)
                    axes[i, 3].set_title(f'{name} Frequency Energy')
                    axes[i, 3].set_ylabel('Relative Energy')
                    
                    for bar, energy in zip(bars, energies):
                        height = bar.get_height()
                        axes[i, 3].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{energy:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.save_plot(fig, 'frequency_analysis')
            freq_stats['plot_path'] = str(plot_path)
        
        self.log("Frequency analysis complete", "success")
        return freq_stats
    
    def create_enhanced_visualizations(self) -> Optional[Path]:
        """Create enhanced contrast visualizations"""
        self.log("Creating enhanced visualizations...", "info")
        
        if not self.generated_images and not self.real_images:
            self.log("No images available for visualization", "warning")
            return None
        
        # Create comparison grid
        n_samples = min(8, len(self.generated_images) if self.generated_images else 0,
                       len(self.real_images) if self.real_images else 0)
        
        if n_samples == 0:
            return None
        
        fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 3, 12))
        fig.suptitle('Enhanced Contrast Comparison', fontsize=16)
        
        def enhance_image(img, factor=3.0):
            """Enhance contrast of an image"""
            pil_img = Image.fromarray(img)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(factor)
            return np.array(enhanced)
        
        for i in range(n_samples):
            # Real images
            if self.real_images and i < len(self.real_images):
                # Original
                axes[0, i].imshow(self.real_images[i], cmap='gray')
                axes[0, i].set_title(f'Real {i+1}')
                axes[0, i].axis('off')
                
                # Enhanced
                enhanced = enhance_image(self.real_images[i], 3.0)
                axes[1, i].imshow(enhanced, cmap='gray')
                axes[1, i].set_title(f'Real Enhanced')
                axes[1, i].axis('off')
            
            # Generated images
            if self.generated_images and i < len(self.generated_images):
                # Original
                axes[2, i].imshow(self.generated_images[i], cmap='gray')
                axes[2, i].set_title(f'Gen {i+1}')
                axes[2, i].axis('off')
                
                # Enhanced
                enhanced = enhance_image(self.generated_images[i], 3.0)
                axes[3, i].imshow(enhanced, cmap='gray')
                axes[3, i].set_title(f'Gen Enhanced')
                axes[3, i].axis('off')
        
        plt.tight_layout()
        plot_path = self.save_plot(fig, 'enhanced_comparison')
        
        self.log("Enhanced visualizations created", "success")
        return plot_path


# ==================== Main Analyzer Class ====================

class CARSModelAnalyzer:
    """Main analyzer combining all components"""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.console = console
        
        # Initialize components
        self.monitor = TrainingMonitor(config)
        self.image_analyzer = ImageAnalyzer(config)
        
        # Results container
        self.results = AnalysisResults()
        
        # Integration hooks
        self.hooks = {} if config.enable_hooks else None
    
    def run(self) -> AnalysisResults:
        """Run analysis based on configured mode"""
        self.console.print(Panel.fit(
            f"[bold]CARS-FASTGAN Model Analyzer[/bold]\n"
            f"Mode: {self.config.mode.value}\n"
            f"Output: {self.config.output_dir}",
            title="Starting Analysis"
        ))
        
        if self.config.mode == AnalysisMode.MONITOR:
            self._run_monitoring()
        elif self.config.mode == AnalysisMode.ANALYZE:
            self._run_image_analysis()
        elif self.config.mode == AnalysisMode.FULL:
            self._run_full_analysis()
        elif self.config.mode == AnalysisMode.DASHBOARD:
            self._run_dashboard()
        
        # Generate reports
        self._generate_reports()
        
        return self.results
    
    def _run_monitoring(self):
        """Run training monitoring"""
        self.console.print("\n[bold]Running Training Monitor[/bold]")
        
        while True:
            # Find latest experiment
            experiment_dir = self.monitor.find_latest_experiment()
            
            if experiment_dir is None:
                self.monitor.log("No experiments found", "warning")
                if not self.config.continuous_monitoring:
                    break
                time.sleep(self.config.refresh_interval)
                continue
            
            # Parse metrics
            metrics = self.monitor.parse_tensorboard_logs(experiment_dir)
            self.results.training_metrics = metrics
            
            if metrics:
                # Analyze progress
                analysis = self.monitor.analyze_training_progress(metrics)
                self.results.health_score = analysis.get('health_score', 100)
                self.results.training_stable = analysis.get('training_stable', True)
                self.results.convergence_status = analysis.get('convergence_status', 'training')
                self.results.recommendations.extend(analysis.get('recommendations', []))
                
                # Create plots
                plot_path = self.monitor.create_training_plots(metrics)
                if plot_path:
                    self.results.plots['training_progress'] = plot_path
                
                # Display status
                self._display_monitoring_status(analysis)
            
            if not self.config.continuous_monitoring:
                break
            
            self.console.print(f"\n⏳ Next check in {self.config.refresh_interval}s...")
            time.sleep(self.config.refresh_interval)
    
    def _run_image_analysis(self):
        """Run image analysis"""
        self.console.print("\n[bold]Running Image Analysis[/bold]")
        
        # Load images
        self.image_analyzer.load_images()
        
        if not self.image_analyzer.generated_images and not self.image_analyzer.real_images:
            self.console.print("[red]No images found for analysis![/red]")
            return
        
        # Run analyses
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Dynamic range
            task = progress.add_task("Analyzing dynamic range...", total=1)
            dr_stats = self.image_analyzer.analyze_dynamic_range()
            self.results.dynamic_range_stats = dr_stats
            progress.update(task, completed=1)
            
            # Noise patterns
            task = progress.add_task("Detecting noise patterns...", total=1)
            noise_stats = self.image_analyzer.detect_noise_patterns()
            self.results.noise_pattern_stats = noise_stats
            progress.update(task, completed=1)
            
            # Frequency content
            task = progress.add_task("Analyzing frequency content...", total=1)
            freq_stats = self.image_analyzer.analyze_frequency_content()
            self.results.frequency_content_stats = freq_stats
            progress.update(task, completed=1)
            
            # Enhanced visualizations
            task = progress.add_task("Creating visualizations...", total=1)
            viz_path = self.image_analyzer.create_enhanced_visualizations()
            if viz_path:
                self.results.plots['enhanced_comparison'] = viz_path
            progress.update(task, completed=1)
        
        # Detect issues
        self._detect_issues()
        
        # Display results
        self._display_analysis_results()
    
    def _run_full_analysis(self):
        """Run both monitoring and image analysis"""
        self.console.print("\n[bold]Running Full Analysis[/bold]")
        
        # First run monitoring (once)
        self.config.continuous_monitoring = False
        self._run_monitoring()
        
        # Then run image analysis
        # Try to find generated images from the experiment
        if self.config.generated_dir is None:
            # Look for generated images in common locations
            possible_dirs = [
                self.config.experiment_dir / "generated_images",
                self.config.output_dir.parent / "evaluation" / "samples",
                Path("outputs/images")
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    self.config.generated_dir = dir_path
                    self.console.print(f"Found generated images at: {dir_path}")
                    break
        
        self._run_image_analysis()
    
    def _run_dashboard(self):
        """Run interactive web dashboard"""
        if not STREAMLIT_AVAILABLE:
            self.console.print("[red]Streamlit not available![/red]")
            self.console.print("Install with: pip install streamlit")
            return
        
        # Create dashboard script
        dashboard_script = self.config.output_dir / "dashboard.py"
        self._create_dashboard_script(dashboard_script)
        
        # Launch streamlit
        import subprocess
        self.console.print(f"\n[green]Launching dashboard at http://{self.config.dashboard_host}:{self.config.dashboard_port}[/green]")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", str(self.config.dashboard_port),
            "--server.address", self.config.dashboard_host
        ])
    
    def _detect_issues(self):
        """Detect issues based on analysis results"""
        issues = []
        recommendations = []
        
        # Check dynamic range
        if self.results.dynamic_range_stats:
            dr = self.results.dynamic_range_stats
            if dr.get('generated_dynamic_range', 0) < dr.get('real_dynamic_range', 0) * 0.7:
                issues.append("Low dynamic range in generated images")
                recommendations.append("Increase model capacity or adjust normalization")
            
            real_mean = dr.get('real_mean', 0)
            gen_mean = dr.get('generated_mean', 0)
            if abs(real_mean - gen_mean) > 50:
                issues.append("Severe intensity mismatch")
                recommendations.append("Fix output normalization - check tanh scaling")
        
        # Check noise patterns
        if self.results.noise_pattern_stats:
            noise = self.results.noise_pattern_stats
            
            # FFT checkerboard
            real_fft_cb = noise.get('real_fft_checkerboard_score_mean', 0)
            gen_fft_cb = noise.get('generated_fft_checkerboard_score_mean', 0)
            if gen_fft_cb > real_fft_cb * 2:
                issues.append("FFT checkerboard artifacts detected")
                recommendations.append("Replace upsampling layers with learned upsampling")
            
            # Structural complexity
            real_struct = noise.get('real_structural_complexity_mean', 0)
            gen_struct = noise.get('generated_structural_complexity_mean', 0)
            if gen_struct < real_struct * 0.7:
                issues.append("Low structural complexity")
                recommendations.append("Model not learning tissue patterns - check architecture")
            
            # Follicular structures
            real_foll = noise.get('real_follicle_like_structures_mean', 0)
            gen_foll = noise.get('generated_follicle_like_structures_mean', 0)
            if gen_foll < real_foll * 0.3:
                issues.append("Missing follicular structures")
                recommendations.append("Generated images lack characteristic thyroid follicles")
        
        self.results.issues = issues
        self.results.recommendations.extend(recommendations)
    
    def _display_monitoring_status(self, analysis: Dict[str, Any]):
        """Display monitoring status in console"""
        table = Table(title="Training Status")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Epochs", str(analysis.get('total_epochs', 0)))
        table.add_row("Current Epoch", str(analysis.get('current_epoch', 0)))
        table.add_row("Health Score", f"{analysis.get('health_score', 0):.1f}/100")
        table.add_row("Training Stable", "✅ Yes" if analysis.get('training_stable', True) else "❌ No")
        table.add_row("Convergence", analysis.get('convergence_status', 'Unknown'))
        
        if 'latest_fid' in analysis:
            table.add_row("Latest FID", f"{analysis['latest_fid']:.2f}")
        if 'best_fid' in analysis:
            table.add_row("Best FID", f"{analysis['best_fid']:.2f}")
        
        self.console.print(table)
        
        # Print recommendations
        if analysis.get('recommendations'):
            self.console.print("\n[bold]Recommendations:[/bold]")
            for rec in analysis['recommendations']:
                self.console.print(f"  {rec}")
    
    def _display_analysis_results(self):
        """Display image analysis results"""
        table = Table(title="Image Analysis Results")
        
        table.add_column("Analysis", style="cyan", no_wrap=True)
        table.add_column("Real", style="green")
        table.add_column("Generated", style="yellow")
        table.add_column("Status", style="magenta")
        
        # Dynamic range
        if self.results.dynamic_range_stats:
            dr = self.results.dynamic_range_stats
            real_dr = dr.get('real_dynamic_range', 0)
            gen_dr = dr.get('generated_dynamic_range', 0)
            status = "✅ OK" if gen_dr >= real_dr * 0.7 else "⚠️ Low"
            table.add_row("Dynamic Range", f"{real_dr:.1f}", f"{gen_dr:.1f}", status)
            
            real_mean = dr.get('real_mean', 0)
            gen_mean = dr.get('generated_mean', 0)
            status = "✅ OK" if abs(real_mean - gen_mean) < 20 else "⚠️ Mismatch"
            table.add_row("Mean Intensity", f"{real_mean:.1f}", f"{gen_mean:.1f}", status)
        
        # Noise patterns
        if self.results.noise_pattern_stats:
            noise = self.results.noise_pattern_stats
            
            real_cb = noise.get('real_fft_checkerboard_score_mean', 0)
            gen_cb = noise.get('generated_fft_checkerboard_score_mean', 0)
            status = "✅ OK" if gen_cb <= real_cb * 2 else "❌ Artifacts"
            table.add_row("FFT Checkerboard", f"{real_cb:.3f}", f"{gen_cb:.3f}", status)
            
            real_struct = noise.get('real_structural_complexity_mean', 0)
            gen_struct = noise.get('generated_structural_complexity_mean', 0)
            status = "✅ OK" if gen_struct >= real_struct * 0.7 else "⚠️ Low"
            table.add_row("Structural Complexity", f"{real_struct:.2f}", f"{gen_struct:.2f}", status)
        
        self.console.print(table)
        
        # Print issues and recommendations
        if self.results.issues:
            self.console.print("\n[bold red]Issues Detected:[/bold red]")
            for issue in self.results.issues:
                self.console.print(f"  ⚠️ {issue}")
        
        if self.results.recommendations:
            self.console.print("\n[bold green]Recommendations:[/bold green]")
            for rec in self.results.recommendations:
                self.console.print(f"  💡 {rec}")
    
    def _generate_reports(self):
        """Generate reports in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Markdown report
        if "markdown" in self.config.report_formats:
            report_path = self.config.output_dir / "reports" / f"analysis_report_{timestamp}.md"
            self._generate_markdown_report(report_path)
            self.results.reports['markdown'] = report_path
        
        # JSON report
        if "json" in self.config.report_formats:
            report_path = self.config.output_dir / "reports" / f"analysis_report_{timestamp}.json"
            self._generate_json_report(report_path)
            self.results.reports['json'] = report_path
        
        # HTML report
        if "html" in self.config.report_formats:
            report_path = self.config.output_dir / "reports" / f"analysis_report_{timestamp}.html"
            self._generate_html_report(report_path)
            self.results.reports['html'] = report_path
    
    def _generate_markdown_report(self, report_path: Path):
        """Generate markdown report"""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# CARS-FASTGAN Model Analysis Report\n\n")
            f.write(f"**Generated**: {self.results.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode**: {self.config.mode.value}\n")
            f.write(f"**Version**: {self.results.analyzer_version}\n\n")
            
            # Training Status
            if self.results.training_metrics:
                f.write("## Training Status\n\n")
                f.write(f"- **Total Epochs**: {self.results.training_metrics[-1].epoch if self.results.training_metrics else 0}\n")
                f.write(f"- **Health Score**: {self.results.health_score:.1f}/100\n")
                f.write(f"- **Training Stable**: {'Yes' if self.results.training_stable else 'No'}\n")
                f.write(f"- **Convergence**: {self.results.convergence_status}\n\n")
                
                # Latest metrics
                if self.results.training_metrics:
                    latest = self.results.training_metrics[-1]
                    f.write("### Latest Metrics\n\n")
                    f.write(f"- **D Loss**: {latest.d_loss:.4f}\n")
                    f.write(f"- **G Loss**: {latest.g_loss:.4f}\n")
                    if latest.fid_score is not None:
                        f.write(f"- **FID Score**: {latest.fid_score:.2f}\n")
                    f.write("\n")
            
            # Image Analysis
            if self.results.dynamic_range_stats or self.results.noise_pattern_stats:
                f.write("## Image Analysis\n\n")
                
                # Dynamic Range
                if self.results.dynamic_range_stats:
                    dr = self.results.dynamic_range_stats
                    f.write("### Dynamic Range\n\n")
                    f.write("| Metric | Real | Generated | Status |\n")
                    f.write("|--------|------|-----------|--------|\n")
                    
                    real_dr = dr.get('real_dynamic_range', 0)
                    gen_dr = dr.get('generated_dynamic_range', 0)
                    status = '✅' if gen_dr >= real_dr * 0.7 else '⚠️'
                    f.write(f"| Dynamic Range | {real_dr:.1f} | {gen_dr:.1f} | {status} |\n")
                    
                    real_mean = dr.get('real_mean', 0)
                    gen_mean = dr.get('generated_mean', 0)
                    status = '✅' if abs(real_mean - gen_mean) < 20 else '⚠️'
                    f.write(f"| Mean Intensity | {real_mean:.1f} | {gen_mean:.1f} | {status} |\n")
                    
                    real_std = dr.get('real_std', 0)
                    gen_std = dr.get('generated_std', 0)
                    status = '✅' if gen_std >= real_std * 0.7 else '⚠️'
                    f.write(f"| Std Deviation | {real_std:.1f} | {gen_std:.1f} | {status} |\n")
                    f.write("\n")
                
                # Noise Patterns
                if self.results.noise_pattern_stats:
                    noise = self.results.noise_pattern_stats
                    f.write("### Noise Pattern Analysis\n\n")
                    
                    # Key metrics
                    metrics_to_report = [
                        ('FFT Checkerboard', 'fft_checkerboard_score', 2.0, '❌'),
                        ('Structural Complexity', 'structural_complexity', 0.7, '⚠️'),
                        ('Follicular Structures', 'follicle_like_structures', 0.3, '⚠️'),
                        ('High Freq Energy', 'high_freq_energy', 1.5, '⚠️')
                    ]
                    
                    for metric_name, metric_key, threshold, warning in metrics_to_report:
                        real_val = noise.get(f'real_{metric_key}_mean', 0)
                        gen_val = noise.get(f'generated_{metric_key}_mean', 0)
                        
                        if metric_key in ['structural_complexity', 'follicle_like_structures']:
                            status = '✅' if gen_val >= real_val * threshold else warning
                        else:
                            status = '✅' if gen_val <= real_val * threshold else warning
                        
                        f.write(f"- **{metric_name}**: Real {real_val:.3f}, Generated {gen_val:.3f} {status}\n")
                    f.write("\n")
            
            # Issues and Recommendations
            if self.results.issues:
                f.write("## Issues Detected\n\n")
                for issue in self.results.issues:
                    f.write(f"- ⚠️ {issue}\n")
                f.write("\n")
            
            if self.results.recommendations:
                f.write("## Recommendations\n\n")
                for rec in self.results.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Generated Files
            f.write("## Generated Files\n\n")
            if self.results.plots:
                f.write("### Plots\n")
                for name, path in self.results.plots.items():
                    f.write(f"- `{name}`: {path}\n")
                f.write("\n")
            
            if self.results.reports:
                f.write("### Reports\n")
                for format_type, path in self.results.reports.items():
                    f.write(f"- `{format_type}`: {path}\n")
                f.write("\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            if self.results.health_score > 80:
                f.write("- ✅ Training is progressing well - continue monitoring\n")
                f.write("- Consider evaluating on test set\n")
            elif self.results.health_score > 60:
                f.write("- ⚠️ Monitor training closely for potential issues\n")
                f.write("- Consider adjusting hyperparameters if problems persist\n")
            else:
                f.write("- ❌ Serious issues detected\n")
                f.write("- **Recommendation**: Review model architecture and training settings\n")
                f.write("- Consider stopping and restarting with fixes\n")
        
        self.console.print(f"[green]✅ Markdown report saved: {report_path}[/green]")
    
    def _generate_json_report(self, report_path: Path):
        """Generate JSON report"""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        report_data = {
            'metadata': {
                'timestamp': self.results.analysis_timestamp.isoformat(),
                'version': self.results.analyzer_version,
                'mode': self.config.mode.value
            },
            'training': {
                'health_score': self.results.health_score,
                'training_stable': self.results.training_stable,
                'convergence_status': self.results.convergence_status,
                'total_epochs': self.results.training_metrics[-1].epoch if self.results.training_metrics else 0,
                'metrics_count': len(self.results.training_metrics)
            },
            'image_analysis': {
                'dynamic_range': self.results.dynamic_range_stats,
                'noise_patterns': self.results.noise_pattern_stats,
                'frequency_content': self.results.frequency_content_stats
            },
            'diagnostics': {
                'issues': self.results.issues,
                'recommendations': self.results.recommendations
            },
            'artifacts': {
                'plots': {k: str(v) for k, v in self.results.plots.items()},
                'reports': {k: str(v) for k, v in self.results.reports.items()}
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.console.print(f"[green]✅ JSON report saved: {report_path}[/green]")
    
    def _generate_html_report(self, report_path: Path):
        """Generate HTML report with embedded plots"""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simple HTML template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CARS-FASTGAN Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .bad {{ color: red; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>CARS-FASTGAN Model Analysis Report</h1>
    <p><strong>Generated:</strong> {self.results.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Mode:</strong> {self.config.mode.value}</p>
    
    <h2>Summary</h2>
    <div class="metric">
        <strong>Health Score:</strong> 
        <span class="{'good' if self.results.health_score > 80 else 'warning' if self.results.health_score > 60 else 'bad'}">
            {self.results.health_score:.1f}/100
        </span>
    </div>
    <div class="metric">
        <strong>Training Stable:</strong> 
        <span class="{'good' if self.results.training_stable else 'bad'}">
            {'Yes' if self.results.training_stable else 'No'}
        </span>
    </div>
"""
        
        # Add issues if any
        if self.results.issues:
            html_content += """
    <h2>Issues Detected</h2>
    <ul>
"""
            for issue in self.results.issues:
                html_content += f"        <li class='warning'>{issue}</li>\n"
            html_content += "    </ul>\n"
        
        # Add recommendations
        if self.results.recommendations:
            html_content += """
    <h2>Recommendations</h2>
    <ul>
"""
            for rec in self.results.recommendations:
                html_content += f"        <li>{rec}</li>\n"
            html_content += "    </ul>\n"
        
        # Close HTML
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.console.print(f"[green]✅ HTML report saved: {report_path}[/green]")
    
    def _create_dashboard_script(self, script_path: Path):
        """Create Streamlit dashboard script"""
        dashboard_code = '''
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="CARS-FASTGAN Analysis Dashboard", layout="wide")

st.title("🔬 CARS-FASTGAN Model Analysis Dashboard")

# Load the latest analysis results
analysis_dir = Path(__file__).parent
reports_dir = analysis_dir / "reports"
plots_dir = analysis_dir / "plots"

# Find latest JSON report
json_reports = list(reports_dir.glob("analysis_report_*.json"))
if json_reports:
    latest_report = max(json_reports, key=lambda x: x.stat().st_mtime)
    
    with open(latest_report) as f:
        data = json.load(f)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_score = data['training']['health_score']
        st.metric("Health Score", f"{health_score:.1f}/100", 
                  delta="Good" if health_score > 80 else "Warning" if health_score > 60 else "Bad")
    
    with col2:
        st.metric("Total Epochs", data['training']['total_epochs'])
    
    with col3:
        st.metric("Training Stable", "Yes" if data['training']['training_stable'] else "No")
    
    with col4:
        st.metric("Convergence", data['training']['convergence_status'])
    
    # Issues and Recommendations
    if data['diagnostics']['issues']:
        st.warning("### Issues Detected")
        for issue in data['diagnostics']['issues']:
            st.write(f"- {issue}")
    
    if data['diagnostics']['recommendations']:
        st.info("### Recommendations")
        for rec in data['diagnostics']['recommendations']:
            st.write(f"- {rec}")
    
    # Display plots
    st.header("Analysis Plots")
    
    plot_files = list(plots_dir.glob("*.png"))
    if plot_files:
        for plot_file in plot_files:
            st.subheader(plot_file.stem.replace('_', ' ').title())
            st.image(str(plot_file))
    
else:
    st.error("No analysis reports found. Please run the analyzer first.")

# Add refresh button
if st.button("Refresh"):
    st.experimental_rerun()
'''
        
        with open(script_path, 'w') as f:
            f.write(dashboard_code)
    
    # Integration API
    def register_hook(self, name: str, callback):
        """Register a callback hook for integration"""
        if self.hooks is not None:
            self.hooks[name] = callback
    
    def trigger_hook(self, name: str, *args, **kwargs):
        """Trigger a registered hook"""
        if self.hooks and name in self.hooks:
            return self.hooks[name](*args, **kwargs)
    
    def get_integration_api(self):
        """Get integration API for cars_training_manager.py"""
        return {
            'analyze': self.run,
            'get_health_score': lambda: self.results.health_score,
            'get_recommendations': lambda: self.results.recommendations,
            'get_issues': lambda: self.results.issues,
            'register_hook': self.register_hook
        }


# ==================== CLI Interface ====================

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='CARS-FASTGAN Model Analyzer - Comprehensive training and generation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor training in real-time
  python cars_model_analyzer.py --mode monitor --experiment_dir experiments
  
  # Analyze generated images
  python cars_model_analyzer.py --mode analyze --generated_dir outputs/samples --real_dir data/processed
  
  # Run full analysis
  python cars_model_analyzer.py --mode full --experiment_dir experiments --real_dir data/processed
  
  # Launch interactive dashboard
  python cars_model_analyzer.py --mode dashboard --experiment_dir experiments
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['monitor', 'analyze', 'full', 'dashboard'],
                       help='Analysis mode (default: full)')
    
    # Paths
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiments directory containing logs and checkpoints')
    parser.add_argument('--real_dir', type=str, default=None,
                       help='Directory containing real images')
    parser.add_argument('--generated_dir', type=str, default=None,
                       help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str, default='outputs/analysis',
                       help='Output directory for analysis results')
    
    # Monitoring settings
    parser.add_argument('--refresh_interval', type=int, default=30,
                       help='Refresh interval for monitoring in seconds')
    parser.add_argument('--once', action='store_true',
                       help='Run monitoring once instead of continuously')
    
    # Analysis settings
    parser.add_argument('--sample_size', type=int, default=50,
                       help='Number of images to analyze')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (fewer samples)')
    
    # Visualization
    parser.add_argument('--plot_format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Format for saving plots')
    parser.add_argument('--plot_dpi', type=int, default=150,
                       help='DPI for saving plots')
    
    # Reporting
    parser.add_argument('--report_formats', nargs='+', 
                       default=['markdown', 'json'],
                       choices=['markdown', 'json', 'html'],
                       help='Report formats to generate')
    
    # Dashboard
    parser.add_argument('--dashboard_port', type=int, default=8501,
                       help='Port for web dashboard')
    parser.add_argument('--dashboard_host', type=str, default='localhost',
                       help='Host for web dashboard')
    
    # Integration
    parser.add_argument('--webhook_url', type=str, default=None,
                       help='Webhook URL for notifications')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalyzerConfig(
        mode=AnalysisMode(args.mode),
        experiment_dir=Path(args.experiment_dir),
        real_dir=Path(args.real_dir) if args.real_dir else None,
        generated_dir=Path(args.generated_dir) if args.generated_dir else None,
        output_dir=Path(args.output_dir),
        refresh_interval=args.refresh_interval,
        continuous_monitoring=not args.once,
        sample_size=args.sample_size if not args.quick else 10,
        comprehensive_analysis=not args.quick,
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
        report_formats=args.report_formats,
        webhook_url=args.webhook_url,
        dashboard_port=args.dashboard_port,
        dashboard_host=args.dashboard_host
    )
    
    # Create and run analyzer
    analyzer = CARSModelAnalyzer(config)
    results = analyzer.run()
    
    # Print summary
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(f"Health Score: {results.health_score:.1f}/100")
    console.print(f"Issues Found: {len(results.issues)}")
    console.print(f"Reports Generated: {len(results.reports)}")
    
    # Print file locations
    if results.reports:
        console.print("\n[bold]Generated Reports:[/bold]")
        for format_type, path in results.reports.items():
            console.print(f"  {format_type}: {path}")
    
    if results.plots:
        console.print("\n[bold]Generated Plots:[/bold]")
        for name, path in results.plots.items():
            console.print(f"  {name}: {path}")
    
    return 0 if results.health_score > 60 else 1


if __name__ == "__main__":
    sys.exit(main())