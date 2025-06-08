"""
Architecture Testing Script for CARS-FASTGAN
Tests model components, memory usage, and training stability
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fastgan import FastGAN, FastGANGenerator, FastGANDiscriminator
from src.training.fastgan_module import FastGANModule
from src.data.dataset import CARSDataModule


class ArchitectureTester:
    """Comprehensive architecture testing"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Testing on device: {self.device}")
        
        # Create output directory
        self.output_dir = Path("tests/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_generator_shapes(self):
        """Test generator output shapes for different configurations"""
        print("\n" + "="*50)
        print("Testing Generator Shapes")
        print("="*50)
        
        configs = [
            {"name": "Micro", "ngf": 32, "n_layers": 3, "image_size": 256},
            {"name": "Standard", "ngf": 64, "n_layers": 4, "image_size": 512},
            {"name": "Large", "ngf": 128, "n_layers": 5, "image_size": 512},
        ]
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration:")
            print(f"  - NGF: {config['ngf']}")
            print(f"  - Layers: {config['n_layers']}")
            print(f"  - Image size: {config['image_size']}")
            
            try:
                generator = FastGANGenerator(
                    latent_dim=256,
                    ngf=config['ngf'],
                    n_layers=config['n_layers'],
                    image_size=config['image_size'],
                    channels=1
                ).to(self.device)
                
                # Test with different batch sizes
                batch_sizes = [1, 4, 8]
                for batch_size in batch_sizes:
                    z = torch.randn(batch_size, 256, device=self.device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        output = generator(z)
                    generation_time = time.time() - start_time
                    
                    expected_shape = (batch_size, 1, config['image_size'], config['image_size'])
                    
                    if output.shape == expected_shape:
                        print(f"    ✅ Batch {batch_size}: {output.shape} - {generation_time:.3f}s")
                    else:
                        print(f"    ❌ Batch {batch_size}: Expected {expected_shape}, got {output.shape}")
                
                # Count parameters
                total_params = sum(p.numel() for p in generator.parameters())
                trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
                print(f"    Parameters: {total_params:,} total, {trainable_params:,} trainable")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    def test_discriminator_shapes(self):
        """Test discriminator input/output shapes"""
        print("\n" + "="*50)
        print("Testing Discriminator Shapes")
        print("="*50)
        
        configs = [
            {"name": "Micro", "ndf": 32, "n_layers": 2, "image_size": 256},
            {"name": "Standard", "ndf": 64, "n_layers": 3, "image_size": 512},
            {"name": "Large", "ndf": 128, "n_layers": 4, "image_size": 512},
        ]
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration:")
            
            try:
                discriminator = FastGANDiscriminator(
                    ndf=config['ndf'],
                    n_layers=config['n_layers'],
                    image_size=config['image_size'],
                    channels=1,
                    use_multiscale=True,
                    num_scales=2
                ).to(self.device)
                
                # Test with different batch sizes
                batch_sizes = [1, 4, 8]
                for batch_size in batch_sizes:
                    x = torch.randn(batch_size, 1, config['image_size'], config['image_size'], device=self.device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        output = discriminator(x, return_features=True)
                    discrimination_time = time.time() - start_time
                    
                    if isinstance(output, tuple):
                        main_output, scale_outputs, features = output
                        print(f"    ✅ Batch {batch_size}: Main {main_output.shape}, "
                              f"Scales: {len(scale_outputs)}, Features: {len(features)} - {discrimination_time:.3f}s")
                    else:
                        print(f"    ✅ Batch {batch_size}: {output.shape} - {discrimination_time:.3f}s")
                
                # Count parameters
                total_params = sum(p.numel() for p in discriminator.parameters())
                print(f"    Parameters: {total_params:,}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    def test_full_model_forward(self):
        """Test complete FastGAN forward pass"""
        print("\n" + "="*50)
        print("Testing Complete FastGAN Model")
        print("="*50)
        
        try:
            model = FastGAN(
                latent_dim=256,
                ngf=64,
                ndf=64,
                generator_layers=4,
                discriminator_layers=3,
                image_size=512,
                channels=1,
                use_skip_connections=True,
                use_multiscale=True
            ).to(self.device)
            
            batch_size = 4
            
            # Test generation
            print("Testing generation...")
            fake_images = model.generate(batch_size, self.device)
            print(f"✅ Generated images shape: {fake_images.shape}")
            
            # Test discrimination
            print("Testing discrimination...")
            real_images = torch.randn(batch_size, 1, 512, 512, device=self.device)
            real_pred = model.discriminate(real_images, return_features=True)
            fake_pred = model.discriminate(fake_images, return_features=True)
            
            print(f"✅ Real prediction shape: {real_pred[0].shape}")
            print(f"✅ Fake prediction shape: {fake_pred[0].shape}")
            print(f"✅ Feature extraction working: {len(real_pred[2])} features")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                print(f"✅ GPU Memory allocated: {memory_allocated:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error in complete model test: {e}")
            import traceback
            traceback.print_exc()
    
    def test_image_generation_quality(self):
        """Test and visualize generated image quality"""
        print("\n" + "="*50)
        print("Testing Image Generation Quality")
        print("="*50)
        
        try:
            model = FastGAN(
                latent_dim=256,
                ngf=64,
                ndf=64,
                generator_layers=4,
                discriminator_layers=3,
                image_size=512,
                channels=1
            ).to(self.device)
            
            # Generate samples
            num_samples = 16
            with torch.no_grad():
                fake_images = model.generate(num_samples, self.device)
            
            # Convert to numpy for visualization
            fake_images_np = fake_images.cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle('Generated CARS Microscopy Images (Untrained)', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                if i < num_samples:
                    img = fake_images_np[i, 0]  # Remove channel dimension
                    # Normalize to [0, 1] for display
                    img = (img + 1) / 2
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Sample {i+1}')
                ax.axis('off')
            
            plt.tight_layout()
            output_path = self.output_dir / 'generated_samples_untrained.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated {num_samples} samples")
            print(f"✅ Image shape: {fake_images.shape}")
            print(f"✅ Value range: [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}]")
            print(f"✅ Visualization saved to: {output_path}")
            
            # Basic quality checks
            mean_val = fake_images.mean().item()
            std_val = fake_images.std().item()
            
            print(f"  Mean pixel value: {mean_val:.3f}")
            print(f"  Std pixel value: {std_val:.3f}")
            
            if abs(mean_val) < 0.5 and std_val > 0.3:
                print("✅ Generated images show reasonable distribution")
            else:
                print("⚠️  Generated images may have distribution issues")
            
        except Exception as e:
            print(f"❌ Image generation error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧪 CARS-FASTGAN Architecture Testing")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available")
        
        print("=" * 60)
        
        # Run all tests
        self.test_generator_shapes()
        self.test_discriminator_shapes()
        self.test_full_model_forward()
        self.test_image_generation_quality()
        
        print("\n" + "="*60)
        print("🎉 Architecture testing completed!")
        print(f"📁 Outputs saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CARS-FASTGAN Architecture')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to run tests on')
    
    args = parser.parse_args()
    
    tester = ArchitectureTester(device=args.device)
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()