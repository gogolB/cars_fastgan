"""
M2 Mac Optimization Script for CARS-FASTGAN
Finds optimal batch sizes, model configurations, and memory settings
"""

import torch
import time
import psutil
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fastgan import FastGAN
from src.training.fastgan_module import FastGANModule


class M2Optimizer:
    """Optimize FASTGAN settings for M2 Mac"""
    
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Optimizing for device: {self.device}")
        
        # Create output directory
        self.output_dir = Path("scripts/optimization_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # System info
        self.system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        print(f"System RAM: {self.system_memory:.1f} GB")
        
        # Results storage
        self.results = {
            'batch_size_tests': [],
            'model_size_tests': [],
            'memory_efficiency': [],
            'speed_benchmarks': []
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    def benchmark_batch_sizes(self, image_size: int = 512) -> Dict:
        """Find optimal batch size for different configurations"""
        print(f"\n{'='*50}")
        print(f"Benchmarking Batch Sizes (Image Size: {image_size})")
        print(f"{'='*50}")
        
        configs = [
            {"name": "Micro", "ngf": 32, "ndf": 32, "g_layers": 3, "d_layers": 2},
            {"name": "Standard", "ngf": 64, "ndf": 64, "g_layers": 4, "d_layers": 3},
            {"name": "Large", "ngf": 128, "ndf": 128, "g_layers": 5, "d_layers": 4}
        ]
        
        batch_sizes = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration:")
            config_results = {
                'config_name': config['name'],
                'image_size': image_size,
                'batch_results': []
            }
            
            for batch_size in batch_sizes:
                try:
                    # Clear memory
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    
                    baseline_memory = self.get_memory_usage()
                    
                    # Create model
                    model = FastGAN(
                        latent_dim=256,
                        ngf=config['ngf'],
                        ndf=config['ndf'],
                        generator_layers=config['g_layers'],
                        discriminator_layers=config['d_layers'],
                        image_size=image_size,
                        channels=1
                    ).to(self.device)
                    
                    model_memory = self.get_memory_usage()
                    
                    # Test forward pass
                    start_time = time.time()
                    
                    # Generator forward
                    z = torch.randn(batch_size, 256, device=self.device)
                    fake_images = model.generator(z)
                    
                    # Discriminator forward
                    real_images = torch.randn(batch_size, 1, image_size, image_size, device=self.device)
                    _ = model.discriminator(real_images)
                    _ = model.discriminator(fake_images)
                    
                    forward_time = time.time() - start_time
                    peak_memory = self.get_memory_usage()
                    
                    # Test backward pass
                    start_time = time.time()
                    
                    # Simulate training step
                    fake_pred = model.discriminator(fake_images)
                    if isinstance(fake_pred, tuple):
                        fake_pred = fake_pred[0]
                    
                    loss = fake_pred.mean()
                    loss.backward()
                    
                    backward_time = time.time() - start_time
                    
                    result = {
                        'batch_size': batch_size,
                        'forward_time': forward_time,
                        'backward_time': backward_time,
                        'total_time': forward_time + backward_time,
                        'baseline_memory': baseline_memory,
                        'model_memory': model_memory,
                        'peak_memory': peak_memory,
                        'memory_increase': peak_memory - baseline_memory,
                        'samples_per_second': batch_size / (forward_time + backward_time),
                        'success': True
                    }
                    
                    config_results['batch_results'].append(result)
                    
                    print(f"  Batch {batch_size:2d}: "
                          f"{forward_time:.3f}s forward, "
                          f"{backward_time:.3f}s backward, "
                          f"{peak_memory - baseline_memory:.2f}GB memory, "
                          f"{result['samples_per_second']:.1f} samples/s")
                    
                    # Clean up
                    del model, fake_images, real_images, z
                    
                except Exception as e:
                    print(f"  Batch {batch_size:2d}: ❌ Failed - {str(e)[:50]}...")
                    
                    config_results['batch_results'].append({
                        'batch_size': batch_size,
                        'success': False,
                        'error': str(e)
                    })
                    
                    # Clean up after failure
                    try:
                        del model, fake_images, real_images, z
                    except:
                        pass
                    
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            
            self.results['batch_size_tests'].append(config_results)
        
        return self.results['batch_size_tests']
    
    def benchmark_model_sizes(self, batch_size: int = 8) -> Dict:
        """Test different model architectures"""
        print(f"\n{'='*50}")
        print(f"Benchmarking Model Sizes (Batch Size: {batch_size})")
        print(f"{'='*50}")
        
        configs = [
            {"name": "Tiny", "ngf": 16, "ndf": 16, "g_layers": 2, "d_layers": 2},
            {"name": "Micro", "ngf": 32, "ndf": 32, "g_layers": 3, "d_layers": 2},
            {"name": "Small", "ngf": 48, "ndf": 48, "g_layers": 3, "d_layers": 3},
            {"name": "Standard", "ngf": 64, "ndf": 64, "g_layers": 4, "d_layers": 3},
            {"name": "Large", "ngf": 96, "ndf": 96, "g_layers": 4, "d_layers": 4},
            {"name": "XLarge", "ngf": 128, "ndf": 128, "g_layers": 5, "d_layers": 4}
        ]
        
        for config in configs:
            print(f"\nTesting {config['name']} model:")
            
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                baseline_memory = self.get_memory_usage()
                
                # Create model
                model = FastGAN(
                    latent_dim=256,
                    ngf=config['ngf'],
                    ndf=config['ndf'],
                    generator_layers=config['g_layers'],
                    discriminator_layers=config['d_layers'],
                    image_size=512,
                    channels=1
                ).to(self.device)
                
                # Count parameters
                g_params = sum(p.numel() for p in model.generator.parameters())
                d_params = sum(p.numel() for p in model.discriminator.parameters())
                total_params = g_params + d_params
                
                model_memory = self.get_memory_usage()
                
                # Benchmark
                start_time = time.time()
                
                z = torch.randn(batch_size, 256, device=self.device)
                fake_images = model.generator(z)
                real_images = torch.randn(batch_size, 1, 512, 512, device=self.device)
                
                _ = model.discriminator(real_images)
                fake_pred = model.discriminator(fake_images)
                
                if isinstance(fake_pred, tuple):
                    fake_pred = fake_pred[0]
                
                loss = fake_pred.mean()
                loss.backward()
                
                total_time = time.time() - start_time
                peak_memory = self.get_memory_usage()
                
                result = {
                    'config_name': config['name'],
                    'ngf': config['ngf'],
                    'ndf': config['ndf'],
                    'g_layers': config['g_layers'],
                    'd_layers': config['d_layers'],
                    'g_params': g_params,
                    'd_params': d_params,
                    'total_params': total_params,
                    'baseline_memory': baseline_memory,
                    'model_memory': model_memory,
                    'peak_memory': peak_memory,
                    'memory_increase': peak_memory - baseline_memory,
                    'total_time': total_time,
                    'samples_per_second': batch_size / total_time,
                    'success': True
                }
                
                self.results['model_size_tests'].append(result)
                
                print(f"  ✅ {config['name']}: "
                      f"{total_params:,} params, "
                      f"{peak_memory - baseline_memory:.2f}GB memory, "
                      f"{total_time:.3f}s, "
                      f"{result['samples_per_second']:.1f} samples/s")
                
                del model, fake_images, real_images, z
                
            except Exception as e:
                print(f"  ❌ {config['name']}: Failed - {str(e)[:50]}...")
                
                self.results['model_size_tests'].append({
                    'config_name': config['name'],
                    'success': False,
                    'error': str(e)
                })
                
                try:
                    del model, fake_images, real_images, z
                except:
                    pass
                
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        return self.results['model_size_tests']
    
    def find_optimal_settings(self) -> Dict:
        """Find optimal settings based on test results"""
        print(f"\n{'='*50}")
        print("Finding Optimal Settings")
        print(f"{'='*50}")
        
        recommendations = {}
        
        # Find best batch sizes for each config
        for config_test in self.results['batch_size_tests']:
            config_name = config_test['config_name']
            successful_results = [r for r in config_test['batch_results'] if r.get('success', False)]
            
            if successful_results:
                # Find optimal batch size (best samples/second with reasonable memory usage)
                optimal = max(
                    [r for r in successful_results if r['memory_increase'] < self.system_memory * 0.7],
                    key=lambda x: x['samples_per_second'],
                    default=successful_results[0]
                )
                
                recommendations[f'{config_name}_optimal_batch'] = {
                    'batch_size': optimal['batch_size'],
                    'memory_usage': optimal['memory_increase'],
                    'samples_per_second': optimal['samples_per_second'],
                    'total_time': optimal['total_time']
                }
        
        # Find best model size
        successful_models = [r for r in self.results['model_size_tests'] if r.get('success', False)]
        if successful_models:
            # Balance between performance and memory usage
            best_model = max(
                [r for r in successful_models if r['memory_increase'] < self.system_memory * 0.6],
                key=lambda x: x['samples_per_second'] / (x['total_params'] / 1e6),  # Efficiency score
                default=successful_models[0]
            )
            
            recommendations['optimal_model'] = {
                'config_name': best_model['config_name'],
                'ngf': best_model['ngf'],
                'ndf': best_model['ndf'],
                'g_layers': best_model['g_layers'],
                'd_layers': best_model['d_layers'],
                'total_params': best_model['total_params'],
                'memory_usage': best_model['memory_increase'],
                'samples_per_second': best_model['samples_per_second']
            }
        
        # Print recommendations
        print("\n🎯 Optimal Settings for Your M2 Mac:")
        print("-" * 40)
        
        if 'optimal_model' in recommendations:
            model = recommendations['optimal_model']
            print(f"📐 Model Architecture: {model['config_name']}")
            print(f"   - Generator filters: {model['ngf']}")
            print(f"   - Discriminator filters: {model['ndf']}")
            print(f"   - Generator layers: {model['g_layers']}")
            print(f"   - Discriminator layers: {model['d_layers']}")
            print(f"   - Total parameters: {model['total_params']:,}")
            print(f"   - Memory usage: {model['memory_usage']:.2f} GB")
            print(f"   - Performance: {model['samples_per_second']:.1f} samples/s")
        
        for config_name in ['Micro', 'Standard', 'Large']:
            batch_key = f'{config_name}_optimal_batch'
            if batch_key in recommendations:
                batch = recommendations[batch_key]
                print(f"\n📦 {config_name} Optimal Batch Size: {batch['batch_size']}")
                print(f"   - Memory usage: {batch['memory_usage']:.2f} GB")
                print(f"   - Performance: {batch['samples_per_second']:.1f} samples/s")
        
        return recommendations
    
    def create_visualizations(self):
        """Create performance visualization plots"""
        print(f"\nCreating performance visualizations...")
        
        # Batch size performance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('M2 Mac Performance Analysis', fontsize=16)
        
        # Plot 1: Batch Size vs Samples/Second
        ax1 = axes[0, 0]
        for config_test in self.results['batch_size_tests']:
            config_name = config_test['config_name']
            successful_results = [r for r in config_test['batch_results'] if r.get('success', False)]
            
            if successful_results:
                batch_sizes = [r['batch_size'] for r in successful_results]
                samples_per_sec = [r['samples_per_second'] for r in successful_results]
                ax1.plot(batch_sizes, samples_per_sec, marker='o', label=config_name)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Batch Size vs Memory Usage
        ax2 = axes[0, 1]
        for config_test in self.results['batch_size_tests']:
            config_name = config_test['config_name']
            successful_results = [r for r in config_test['batch_results'] if r.get('success', False)]
            
            if successful_results:
                batch_sizes = [r['batch_size'] for r in successful_results]
                memory_usage = [r['memory_increase'] for r in successful_results]
                ax2.plot(batch_sizes, memory_usage, marker='s', label=config_name)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Size vs Performance
        ax3 = axes[1, 0]
        successful_models = [r for r in self.results['model_size_tests'] if r.get('success', False)]
        if successful_models:
            model_names = [r['config_name'] for r in successful_models]
            total_params = [r['total_params'] / 1e6 for r in successful_models]  # Millions
            samples_per_sec = [r['samples_per_second'] for r in successful_models]
            
            bars = ax3.bar(model_names, samples_per_sec, alpha=0.7)
            ax3.set_ylabel('Samples/Second')
            ax3.set_title('Performance by Model Size')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add parameter count on bars
            for bar, params in zip(bars, total_params):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{params:.1f}M', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Memory Efficiency
        ax4 = axes[1, 1]
        if successful_models:
            memory_usage = [r['memory_increase'] for r in successful_models]
            efficiency = [r['samples_per_second'] / r['memory_increase'] for r in successful_models]
            
            bars = ax4.bar(model_names, efficiency, alpha=0.7, color='orange')
            ax4.set_ylabel('Samples/Second per GB')
            ax4.set_title('Memory Efficiency')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {self.output_dir}/performance_analysis.png")
    
    def save_results(self, recommendations: Dict):
        """Save all results to files"""
        # Save detailed results as JSON
        results_file = self.output_dir / 'optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save recommendations as JSON
        recommendations_file = self.output_dir / 'recommendations.json'
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Create summary report
        report_file = self.output_dir / 'optimization_report.md'
        with open(report_file, 'w') as f:
            f.write("# M2 Mac Optimization Report\n\n")
            f.write(f"Generated for system with {self.system_memory:.1f} GB RAM\n\n")
            
            f.write("## Recommended Settings\n\n")
            
            if 'optimal_model' in recommendations:
                model = recommendations['optimal_model']
                f.write(f"### Optimal Model Configuration\n")
                f.write(f"- **Architecture**: {model['config_name']}\n")
                f.write(f"- **Generator filters**: {model['ngf']}\n")
                f.write(f"- **Discriminator filters**: {model['ndf']}\n")
                f.write(f"- **Generator layers**: {model['g_layers']}\n")
                f.write(f"- **Discriminator layers**: {model['d_layers']}\n")
                f.write(f"- **Total parameters**: {model['total_params']:,}\n")
                f.write(f"- **Memory usage**: {model['memory_usage']:.2f} GB\n")
                f.write(f"- **Performance**: {model['samples_per_second']:.1f} samples/s\n\n")
            
            f.write("### Optimal Batch Sizes\n\n")
            for config_name in ['Micro', 'Standard', 'Large']:
                batch_key = f'{config_name}_optimal_batch'
                if batch_key in recommendations:
                    batch = recommendations[batch_key]
                    f.write(f"- **{config_name}**: Batch size {batch['batch_size']} ")
                    f.write(f"({batch['memory_usage']:.2f} GB, {batch['samples_per_second']:.1f} samples/s)\n")
            
            f.write(f"\n## Configuration for Hydra\n\n")
            f.write("```yaml\n")
            if 'optimal_model' in recommendations:
                model = recommendations['optimal_model']
                f.write(f"model:\n")
                f.write(f"  generator:\n")
                f.write(f"    ngf: {model['ngf']}\n")
                f.write(f"    n_layers: {model['g_layers']}\n")
                f.write(f"  discriminator:\n")
                f.write(f"    ndf: {model['ndf']}\n")
                f.write(f"    n_layers: {model['d_layers']}\n")
                
                # Find corresponding batch size
                batch_key = f"{model['config_name']}_optimal_batch"
                if batch_key in recommendations:
                    f.write(f"\ndata:\n")
                    f.write(f"  batch_size: {recommendations[batch_key]['batch_size']}\n")
            f.write("```\n")
        
        print(f"✅ Results saved to:")
        print(f"   - {results_file}")
        print(f"   - {recommendations_file}")
        print(f"   - {report_file}")
    
    def run_optimization(self):
        """Run complete optimization suite"""
        print("🚀 Starting M2 Mac Optimization for CARS-FASTGAN")
        print("=" * 60)
        
        # Run benchmarks
        self.benchmark_batch_sizes(image_size=512)
        self.benchmark_model_sizes(batch_size=8)
        
        # Find optimal settings
        recommendations = self.find_optimal_settings()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results(recommendations)
        
        print("\n" + "=" * 60)
        print("🎉 Optimization complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main optimization function"""
    optimizer = M2Optimizer()
    optimizer.run_optimization()


if __name__ == "__main__":
    main()