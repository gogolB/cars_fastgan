"""
Real-time Training Monitor for CARS-FASTGAN
Monitors training progress, generates reports, and provides early stopping recommendations
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
import torch
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


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


class TrainingMonitor:
    """Real-time training monitor"""
    
    def __init__(self, experiment_dir: str = "experiments", refresh_interval: int = 30):
        self.experiment_dir = Path(experiment_dir)
        self.refresh_interval = refresh_interval
        
        self.logs_dir = self.experiment_dir / "logs"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.outputs_dir = Path("outputs")
        
        # Create monitoring output directory
        self.monitor_dir = self.outputs_dir / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 CARS-FASTGAN Training Monitor")
        print(f"📁 Monitoring: {self.logs_dir}")
        print(f"📊 Outputs: {self.monitor_dir}")
        print(f"🔄 Refresh: {refresh_interval}s")
        print("=" * 50)
    
    def find_latest_experiment(self) -> Optional[Path]:
        """Find the most recent experiment"""
        if not self.logs_dir.exists():
            return None
        
        # Look for tensorboard log directories
        log_dirs = [d for d in self.logs_dir.iterdir() if d.is_dir()]
        
        if not log_dirs:
            return None
        
        # Sort by modification time (most recent first)
        latest_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
        
        print(f"📊 Found experiment: {latest_dir.name}")
        return latest_dir
    
    def parse_tensorboard_logs(self, log_dir: Path) -> List[TrainingMetrics]:
        """Parse TensorBoard event files to extract metrics"""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            print("❌ TensorBoard not available. Install with: pip install tensorboard")
            return []
        
        metrics = []
        
        # Find event files
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        
        if not event_files:
            print("⚠️  No TensorBoard event files found")
            return []
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                # Get available tags
                scalar_tags = ea.Tags()['scalars']
                
                # Extract metrics
                epochs = set()
                
                # Get epoch numbers from any available metric
                for tag in scalar_tags:
                    if 'train' in tag or 'val' in tag:
                        scalar_events = ea.Scalars(tag)
                        for event in scalar_events:
                            epochs.add(event.step)
                
                # For each epoch, collect all available metrics
                for epoch in sorted(epochs):
                    metric = TrainingMetrics(
                        epoch=epoch,
                        step=epoch,
                        timestamp=datetime.now(),  # Approximate
                        d_loss=0.0,
                        g_loss=0.0,
                        d_loss_real=0.0,
                        d_loss_fake=0.0,
                        lr_g=0.0002,
                        lr_d=0.0002
                    )
                    
                    # Extract specific metrics
                    for tag in scalar_tags:
                        try:
                            scalar_events = ea.Scalars(tag)
                            # Find the event for this epoch
                            for event in scalar_events:
                                if event.step == epoch:
                                    if 'train/d_loss' in tag:
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
                                    elif 'val/fid_score' in tag:
                                        metric.fid_score = event.value
                                    elif 'train/lr_g' in tag:
                                        metric.lr_g = event.value
                                    elif 'train/lr_d' in tag:
                                        metric.lr_d = event.value
                                    break
                        except:
                            continue
                    
                    metrics.append(metric)
                    
            except Exception as e:
                print(f"⚠️  Error parsing {event_file}: {e}")
                continue
        
        # Sort by epoch
        metrics.sort(key=lambda x: x.epoch)
        
        print(f"📈 Parsed {len(metrics)} metric records")
        return metrics
    
    def analyze_training_progress(self, metrics: List[TrainingMetrics]) -> Dict:
        """Analyze training progress and health"""
        if not metrics:
            return {}
        
        analysis = {
            'total_epochs': len(metrics),
            'current_epoch': metrics[-1].epoch if metrics else 0,
            'training_stable': True,
            'convergence_status': 'training',
            'recommendations': [],
            'health_score': 100
        }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
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
            }
            for m in metrics
        ])
        
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
            analysis['recommendations'].append("⚠️  Loss explosion detected - consider reducing learning rate")
        
        # Check for loss oscillation
        d_loss_std = recent_df['d_loss'].std()
        g_loss_std = recent_df['g_loss'].std()
        
        if d_loss_std > 2.0 or g_loss_std > 2.0:
            analysis['health_score'] -= 20
            analysis['recommendations'].append("⚠️  High loss oscillation - training may be unstable")
        
        # Check discriminator/generator balance
        recent_d_mean = recent_df['d_loss'].mean()
        recent_g_mean = recent_df['g_loss'].mean()
        
        if recent_d_mean < 0.1:
            analysis['health_score'] -= 25
            analysis['recommendations'].append("⚠️  Discriminator too strong - generator may collapse")
        elif recent_g_mean < 0.1:
            analysis['health_score'] -= 25
            analysis['recommendations'].append("⚠️  Generator too strong - discriminator failing")
        
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
        
        # Performance recommendations
        if analysis['health_score'] > 80:
            analysis['recommendations'].append("✅ Training is healthy")
        elif analysis['health_score'] > 60:
            analysis['recommendations'].append("⚠️  Training has minor issues")
        else:
            analysis['recommendations'].append("❌ Training has serious issues - consider stopping")
        
        return analysis
    
    def create_training_plots(self, metrics: List[TrainingMetrics]) -> Path:
        """Create comprehensive training plots"""
        if not metrics:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
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
            }
            for m in metrics
        ])
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CARS-FASTGAN Training Progress', fontsize=16)
        
        # Plot 1: Training Losses
        ax1 = axes[0, 0]
        ax1.plot(df['epoch'], df['d_loss'], label='D Loss', alpha=0.8)
        ax1.plot(df['epoch'], df['g_loss'], label='G Loss', alpha=0.8)
        if df['val_d_loss'].notna().any():
            ax1.plot(df['epoch'], df['val_d_loss'], label='Val D Loss', linestyle='--', alpha=0.8)
        if df['val_g_loss'].notna().any():
            ax1.plot(df['epoch'], df['val_g_loss'], label='Val G Loss', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Discriminator Components
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['d_loss_real'], label='D Loss Real', alpha=0.8)
        ax2.plot(df['epoch'], df['d_loss_fake'], label='D Loss Fake', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Discriminator Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FID Score
        ax3 = axes[0, 2]
        if df['fid_score'].notna().any():
            fid_data = df[df['fid_score'].notna()]
            ax3.plot(fid_data['epoch'], fid_data['fid_score'], marker='o', alpha=0.8)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('FID Score')
            ax3.set_title('FID Score Progress (Lower is Better)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No FID data\navailable', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('FID Score Progress')
        
        # Plot 4: Learning Rates
        ax4 = axes[1, 0]
        ax4.plot(df['epoch'], df['lr_g'], label='Generator LR', alpha=0.8)
        ax4.plot(df['epoch'], df['lr_d'], label='Discriminator LR', alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Plot 5: Loss Smoothed (Moving Average)
        ax5 = axes[1, 1]
        window = min(20, len(df) // 10) if len(df) > 20 else 1
        if window > 1:
            d_loss_smooth = df['d_loss'].rolling(window=window, center=True).mean()
            g_loss_smooth = df['g_loss'].rolling(window=window, center=True).mean()
            ax5.plot(df['epoch'], d_loss_smooth, label=f'D Loss (MA {window})', alpha=0.8)
            ax5.plot(df['epoch'], g_loss_smooth, label=f'G Loss (MA {window})', alpha=0.8)
        else:
            ax5.plot(df['epoch'], df['d_loss'], label='D Loss', alpha=0.8)
            ax5.plot(df['epoch'], df['g_loss'], label='G Loss', alpha=0.8)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Smoothed Loss Trends')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Training Health Score
        ax6 = axes[1, 2]
        # Calculate a simple health score based on loss stability
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
            ax6.plot(epochs_subset, health_scores, color='green', alpha=0.8)
            ax6.fill_between(epochs_subset, health_scores, alpha=0.3, color='green')
            ax6.set_ylim(0, 100)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Health Score')
        ax6.set_title('Training Health Score')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.monitor_dir / f"training_progress_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def check_generated_images(self, experiment_dir: Path) -> Optional[Path]:
        """Find and display latest generated images"""
        # Look for generated images in various locations
        possible_dirs = [
            experiment_dir / "generated_images",
            self.outputs_dir / "images",
            Path("outputs/images")
        ]
        
        latest_images = None
        latest_time = 0
        
        for img_dir in possible_dirs:
            if img_dir.exists():
                for img_file in img_dir.glob("*.png"):
                    mtime = img_file.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_images = img_file
        
        return latest_images
    
    def generate_report(self, metrics: List[TrainingMetrics], analysis: Dict) -> Path:
        """Generate comprehensive training report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_path = self.monitor_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# CARS-FASTGAN Training Report\n\n")
            f.write(f"**Generated**: {timestamp}\n")
            f.write(f"**Total Epochs**: {analysis.get('total_epochs', 'N/A')}\n")
            f.write(f"**Current Epoch**: {analysis.get('current_epoch', 'N/A')}\n")
            f.write(f"**Health Score**: {analysis.get('health_score', 'N/A')}/100\n\n")
            
            # Training Status
            f.write("## Training Status\n\n")
            f.write(f"- **Stability**: {'✅ Stable' if analysis.get('training_stable', True) else '❌ Unstable'}\n")
            f.write(f"- **Convergence**: {analysis.get('convergence_status', 'Unknown').capitalize()}\n")
            
            if 'latest_fid' in analysis:
                f.write(f"- **Latest FID Score**: {analysis['latest_fid']:.2f}\n")
            if 'best_fid' in analysis:
                f.write(f"- **Best FID Score**: {analysis['best_fid']:.2f}\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            else:
                f.write("- No specific recommendations at this time\n")
            
            f.write("\n")
            
            # Latest Metrics
            if metrics:
                latest = metrics[-1]
                f.write("## Latest Metrics\n\n")
                f.write(f"- **Epoch**: {latest.epoch}\n")
                f.write(f"- **Discriminator Loss**: {latest.d_loss:.4f}\n")
                f.write(f"- **Generator Loss**: {latest.g_loss:.4f}\n")
                f.write(f"- **D Loss Real**: {latest.d_loss_real:.4f}\n")
                f.write(f"- **D Loss Fake**: {latest.d_loss_fake:.4f}\n")
                f.write(f"- **Generator LR**: {latest.lr_g:.6f}\n")
                f.write(f"- **Discriminator LR**: {latest.lr_d:.6f}\n")
                
                if latest.val_d_loss is not None:
                    f.write(f"- **Val D Loss**: {latest.val_d_loss:.4f}\n")
                if latest.val_g_loss is not None:
                    f.write(f"- **Val G Loss**: {latest.val_g_loss:.4f}\n")
                if latest.fid_score is not None:
                    f.write(f"- **FID Score**: {latest.fid_score:.2f}\n")
                
                f.write("\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            health_score = analysis.get('health_score', 100)
            
            if health_score > 80:
                f.write("- ✅ Training is progressing well - continue monitoring\n")
                f.write("- Consider evaluating generated samples\n")
            elif health_score > 60:
                f.write("- ⚠️  Monitor training closely for potential issues\n")
                f.write("- Consider adjusting hyperparameters if problems persist\n")
            else:
                f.write("- ❌ Serious training issues detected\n")
                f.write("- **Recommendation**: Stop training and investigate\n")
                f.write("- Check learning rates, model architecture, or data quality\n")
            
            f.write("\n---\n")
            f.write("*Report generated by CARS-FASTGAN Training Monitor*\n")
        
        return report_path
    
    def monitor_loop(self, continuous: bool = True):
        """Main monitoring loop"""
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n{'='*50}")
            print(f"📊 Monitoring Iteration {iteration}")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Find latest experiment
            experiment_dir = self.find_latest_experiment()
            
            if experiment_dir is None:
                print("❌ No experiments found")
                if not continuous:
                    break
                print(f"⏳ Waiting {self.refresh_interval}s...")
                time.sleep(self.refresh_interval)
                continue
            
            # Parse metrics
            metrics = self.parse_tensorboard_logs(experiment_dir)
            
            if not metrics:
                print("⚠️  No metrics found")
                if not continuous:
                    break
                print(f"⏳ Waiting {self.refresh_interval}s...")
                time.sleep(self.refresh_interval)
                continue
            
            # Analyze progress
            analysis = self.analyze_training_progress(metrics)
            
            # Print status
            print(f"📈 Total epochs: {analysis.get('total_epochs', 0)}")
            print(f"🏃 Current epoch: {analysis.get('current_epoch', 0)}")
            print(f"❤️  Health score: {analysis.get('health_score', 0)}/100")
            
            if 'latest_fid' in analysis:
                print(f"🎯 Latest FID: {analysis['latest_fid']:.2f}")
            
            print("\n💡 Recommendations:")
            for rec in analysis.get('recommendations', []):
                print(f"   {rec}")
            
            # Create plots
            plot_path = self.create_training_plots(metrics)
            if plot_path:
                print(f"\n📊 Plots saved: {plot_path}")
            
            # Generate report
            report_path = self.generate_report(metrics, analysis)
            print(f"📋 Report saved: {report_path}")
            
            # Check for generated images
            latest_images = self.check_generated_images(experiment_dir)
            if latest_images:
                print(f"🖼️  Latest images: {latest_images}")
            
            if not continuous:
                break
            
            print(f"\n⏳ Next check in {self.refresh_interval}s...")
            print("   Press Ctrl+C to stop monitoring")
            
            try:
                time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped by user")
                break
        
        print("\n✅ Monitoring session complete")


def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description='Monitor CARS-FASTGAN training progress')
    
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiments directory path')
    parser.add_argument('--refresh_interval', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--once', action='store_true',
                       help='Run once instead of continuous monitoring')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        experiment_dir=args.experiment_dir,
        refresh_interval=args.refresh_interval
    )
    
    monitor.monitor_loop(continuous=not args.once)


if __name__ == "__main__":
    main()