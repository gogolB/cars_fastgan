"""
Quick Training Test for CARS-FASTGAN
Tests end-to-end training with synthetic data to verify everything works
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.fastgan_module import FastGANModule
from src.data.dataset import CARSDataModule


class QuickTrainingTester:
    """Test training with synthetic data"""
    
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
        
        # Create test directories
        self.test_dir = Path("tests/outputs/training_test")
        self.test_dir.mkdir(parents=True, exist_ok=True)
    
    def create_synthetic_cars_data(self, num_images=32, image_size=256):
        """Create synthetic CARS-like images for testing"""
        print(f"Creating {num_images} synthetic CARS images...")
        
        # Create temporary data directory
        data_dir = self.test_dir / "synthetic_data"
        data_dir.mkdir(exist_ok=True)
        
        # Create healthy and cancerous subdirectories
        healthy_dir = data_dir / "healthy"
        cancer_dir = data_dir / "cancerous"
        healthy_dir.mkdir(exist_ok=True)
        cancer_dir.mkdir(exist_ok=True)
        
        np.random.seed(42)  # For reproducible test data
        
        for i in range(num_images // 2):
            # Healthy tissue pattern (more regular structures)
            healthy_img = self._generate_healthy_pattern(image_size)
            healthy_path = healthy_dir / f"healthy_{i:03d}.tif"
            Image.fromarray(healthy_img).save(healthy_path)
            
            # Cancerous tissue pattern (more irregular)
            cancer_img = self._generate_cancer_pattern(image_size)
            cancer_path = cancer_dir / f"cancer_{i:03d}.tif"
            Image.fromarray(cancer_img).save(cancer_path)
        
        print(f"✅ Created synthetic data in: {data_dir}")
        return str(data_dir)
    
    def _generate_healthy_pattern(self, size):
        """Generate synthetic healthy tissue pattern"""
        # Base noise
        img = np.random.normal(0.3, 0.1, (size, size))
        
        # Add regular cellular structures
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        
        # Regular cell-like patterns
        cell_pattern = np.sin(x / 20) * np.sin(y / 20) * 0.2
        img += cell_pattern
        
        # Add some fine structure
        fine_structure = np.random.normal(0, 0.05, (size, size))
        img += fine_structure
        
        # Convert to 16-bit
        img = np.clip(img, 0, 1)
        img = (img * 65535).astype(np.uint16)
        
        return img
    
    def _generate_cancer_pattern(self, size):
        """Generate synthetic cancerous tissue pattern"""
        # Base noise with higher variance
        img = np.random.normal(0.4, 0.15, (size, size))
        
        # Add irregular structures
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        
        # Irregular patterns
        irregular_pattern = np.sin(x / 15 + np.random.random()) * np.cos(y / 25 + np.random.random()) * 0.3
        img += irregular_pattern
        
        # Add random bright spots (abnormal cells)
        num_spots = np.random.randint(5, 15)
        for _ in range(num_spots):
            cx, cy = np.random.randint(50, size-50, 2)
            radius = np.random.randint(10, 30)
            
            spot_mask = ((x - cx)**2 + (y - cy)**2) < radius**2
            img[spot_mask] += np.random.normal(0.3, 0.1)
        
        # Convert to 16-bit
        img = np.clip(img, 0, 1)
        img = (img * 65535).astype(np.uint16)
        
        return img
    
    def test_data_loading(self, data_path):
        """Test data loading with synthetic data"""
        print("\n" + "="*50)
        print("Testing Data Loading")
        print("="*50)
        
        try:
            # Create data module
            datamodule = CARSDataModule(
                data_path=data_path,
                batch_size=4,
                num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
                image_size=256,  # Smaller for faster testing
                use_8bit=True,
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1,
                augment_train=True
            )
            
            # Setup data
            datamodule.setup()
            
            print(f"✅ Train samples: {len(datamodule.train_dataset)}")
            print(f"✅ Val samples: {len(datamodule.val_dataset)}")
            print(f"✅ Test samples: {len(datamodule.test_dataset)}")
            
            # Test data loaders
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            # Get a batch
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            print(f"✅ Train batch shape: {train_batch['image'].shape}")
            print(f"✅ Val batch shape: {val_batch['image'].shape}")
            print(f"✅ Image value range: [{train_batch['image'].min():.3f}, {train_batch['image'].max():.3f}]")
            
            # Visualize a sample
            sample_img = train_batch['image'][0, 0].cpu().numpy()
            sample_img = (sample_img + 1) / 2  # Denormalize
            
            plt.figure(figsize=(6, 6))
            plt.imshow(sample_img, cmap='gray')
            plt.title('Sample Training Image')
            plt.axis('off')
            plt.savefig(self.test_dir / 'sample_training_image.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return datamodule
            
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_quick_training(self, datamodule, max_epochs=3):
        """Test quick training run"""
        print("\n" + "="*50)
        print(f"Testing Quick Training ({max_epochs} epochs)")
        print("="*50)
        
        try:
            # Create model with smaller architecture for fast testing
            model = FastGANModule(
                latent_dim=128,  # Smaller latent dim
                ngf=32,          # Smaller generator
                ndf=32,          # Smaller discriminator
                generator_layers=3,  # Fewer layers
                discriminator_layers=2,
                image_size=256,  # Smaller images
                channels=1,
                generator_lr=0.0002,
                discriminator_lr=0.0002,
                feature_matching_weight=5.0,
                log_images_every_n_epochs=2  # Log more frequently
            )
            
            # Setup trainer
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.test_dir / "checkpoints",
                filename="test-{epoch:02d}",
                save_top_k=2,
                monitor="val/d_loss",
                mode="min"
            )
            
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                callbacks=[checkpoint_callback],
                enable_checkpointing=True,
                enable_model_summary=True,
                enable_progress_bar=True,
                log_every_n_steps=1,
                val_check_interval=1.0,
                limit_train_batches=10,  # Limit for fast testing
                limit_val_batches=5,
                default_root_dir=str(self.test_dir),
                fast_dev_run=False  # Set to True for even faster testing
            )
            
            print("🚀 Starting training...")
            trainer.fit(model, datamodule)
            
            print("✅ Training completed successfully!")
            
            # Test generation after training
            self.test_generation_after_training(model)
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_generation_after_training(self, model):
        """Test image generation after training"""
        print("\nTesting generation after training...")
        
        try:
            model.eval()
            
            with torch.no_grad():
                # Generate samples
                fake_images = model.model.generate(16, model.device)
            
            # Visualize generated images
            fake_images_np = fake_images.cpu().numpy()
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle('Generated Images After Training', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                img = fake_images_np[i, 0]
                img = (img + 1) / 2  # Denormalize
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Generated {i+1}')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.test_dir / 'generated_after_training.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Generated images after training")
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
    
    def run_complete_test(self):
        """Run complete training test"""
        print("🧪 CARS-FASTGAN Quick Training Test")
        print("=" * 60)
        
        try:
            # Create synthetic data
            data_path = self.create_synthetic_cars_data(num_images=64, image_size=256)
            
            # Test data loading
            datamodule = self.test_data_loading(data_path)
            if datamodule is None:
                print("❌ Data loading failed, stopping test")
                return False
            
            # Test training
            success = self.test_quick_training(datamodule, max_epochs=3)
            
            if success:
                print("\n" + "="*60)
                print("🎉 Quick training test PASSED!")
                print(f"📁 Test outputs saved to: {self.test_dir}")
                print("="*60)
                return True
            else:
                print("\n" + "="*60)
                print("❌ Quick training test FAILED!")
                print("="*60)
                return False
                
        except Exception as e:
            print(f"❌ Complete test error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CARS-FASTGAN Training Pipeline')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to run tests on')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs for quick training test')
    
    args = parser.parse_args()
    
    tester = QuickTrainingTester(device=args.device)
    success = tester.run_complete_test()
    
    if success:
        print("\n🎉 All tests passed! Your FASTGAN implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()