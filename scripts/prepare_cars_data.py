"""
CARS Data Preparation Script
Prepares real CARS microscopy data for FASTGAN training
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.analyze import CARSDataAnalyzer


class CARSDataPreparator:
    """Comprehensive CARS data preparation"""
    
    def __init__(self, input_path: str, output_path: str = "data/processed"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "train").mkdir(exist_ok=True)
        (self.output_path / "val").mkdir(exist_ok=True)
        (self.output_path / "test").mkdir(exist_ok=True)
        (self.output_path / "analysis").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        
        print(f"Input path: {self.input_path}")
        print(f"Output path: {self.output_path}")
        
        # Data tracking
        self.image_stats = {
            'total_images': 0,
            'healthy_images': 0,
            'cancerous_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'bit_depth_recommendation': None
        }
        
        self.failed_files = []
    
    def discover_images(self) -> List[Path]:
        """Discover all CARS images in input directory"""
        print("🔍 Discovering CARS images...")
        
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(self.input_path.rglob(f'*{ext}')))
            image_paths.extend(list(self.input_path.rglob(f'*{ext.upper()}')))
        
        # Sort for consistent ordering
        image_paths = sorted(image_paths)
        
        print(f"✅ Found {len(image_paths)} images")
        
        # Try to infer classes from directory structure or filenames
        healthy_count = 0
        cancer_count = 0
        
        for img_path in image_paths:
            path_str = str(img_path).lower()
            if any(term in path_str for term in ['healthy', 'normal', 'benign']):
                healthy_count += 1
            elif any(term in path_str for term in ['cancer', 'malignant', 'tumor']):
                cancer_count += 1
        
        print(f"   - Detected healthy images: {healthy_count}")
        print(f"   - Detected cancerous images: {cancer_count}")
        print(f"   - Unclassified images: {len(image_paths) - healthy_count - cancer_count}")
        
        self.image_stats['total_images'] = len(image_paths)
        self.image_stats['healthy_images'] = healthy_count
        self.image_stats['cancerous_images'] = cancer_count
        
        return image_paths
    
    def analyze_data_quality(self, image_paths: List[Path]) -> Dict:
        """Run comprehensive data analysis"""
        print("\n📊 Analyzing data quality...")
        
        # Use the existing analysis tool
        analyzer = CARSDataAnalyzer(
            data_path=str(self.input_path),
            output_dir=str(self.output_path / "analysis")
        )
        
        # Run dataset characteristics analysis
        characteristics = analyzer.analyze_dataset_characteristics()
        
        # Run bit depth analysis
        bit_depth_analysis = analyzer.analyze_bit_depth_impact(sample_size=min(50, len(image_paths)))
        
        # Store recommendation
        self.image_stats['bit_depth_recommendation'] = bit_depth_analysis['recommendation']
        
        # Generate comprehensive report
        analyzer.generate_report()
        
        print("✅ Data quality analysis complete")
        print(f"   - Recommended bit depth: {'8-bit' if bit_depth_analysis['recommendation']['use_8bit'] else '16-bit'}")
        print(f"   - Confidence: {bit_depth_analysis['recommendation']['confidence']}")
        
        return {
            'characteristics': characteristics,
            'bit_depth_analysis': bit_depth_analysis
        }
    
    def create_train_val_test_splits(
        self, 
        image_paths: List[Path],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, List[Path]]:
        """Create balanced train/validation/test splits"""
        print(f"\n📂 Creating data splits ({train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%})...")
        
        # Separate by class if possible
        healthy_images = []
        cancer_images = []
        unclassified_images = []
        
        for img_path in image_paths:
            path_str = str(img_path).lower()
            if any(term in path_str for term in ['healthy', 'normal', 'benign']):
                healthy_images.append(img_path)
            elif any(term in path_str for term in ['cancer', 'malignant', 'tumor']):
                cancer_images.append(img_path)
            else:
                unclassified_images.append(img_path)
        
        splits = {'train': [], 'val': [], 'test': []}
        
        # Split each class separately to maintain balance
        for class_name, class_images in [('healthy', healthy_images), ('cancer', cancer_images), ('unclassified', unclassified_images)]:
            if len(class_images) == 0:
                continue
                
            print(f"   Splitting {len(class_images)} {class_name} images...")
            
            # First split: train+val vs test
            train_val, test = train_test_split(
                class_images, 
                test_size=test_ratio,
                random_state=seed
            )
            
            # Second split: train vs val
            train, val = train_test_split(
                train_val,
                test_size=val_ratio / (train_ratio + val_ratio),
                random_state=seed
            )
            
            splits['train'].extend(train)
            splits['val'].extend(val)
            splits['test'].extend(test)
            
            print(f"     - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Shuffle each split
        np.random.seed(seed)
        for split_name in splits:
            np.random.shuffle(splits[split_name])
        
        print(f"✅ Final splits:")
        print(f"   - Train: {len(splits['train'])} images")
        print(f"   - Val: {len(splits['val'])} images") 
        print(f"   - Test: {len(splits['test'])} images")
        
        return splits
    
    def process_image(
        self, 
        image_path: Path, 
        target_size: int = 512,
        use_8bit: bool = True,
        quality_check: bool = True
    ) -> Optional[np.ndarray]:
        """Process a single CARS image"""
        try:
            # Load image
            img = np.array(Image.open(image_path))
            
            # Quality checks
            if quality_check:
                if img.size == 0:
                    raise ValueError("Empty image")
                
                if len(img.shape) not in [2, 3]:
                    raise ValueError(f"Invalid image dimensions: {img.shape}")
                
                # Check for corrupted images (all zeros, all same value)
                if np.all(img == 0) or np.all(img == img.flat[0]):
                    raise ValueError("Image appears corrupted (uniform values)")
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bit depth conversion
            if img.dtype == np.uint16 and use_8bit:
                img = (img / 65535.0 * 255).astype(np.uint8)
            elif img.dtype == np.uint8 and not use_8bit:
                img = img.astype(np.uint16) * 257
            
            # Resize if needed
            if img.shape[0] != target_size or img.shape[1] != target_size:
                img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            return img
            
        except Exception as e:
            self.failed_files.append({
                'path': str(image_path),
                'error': str(e)
            })
            return None
    
    def copy_processed_images(
        self, 
        splits: Dict[str, List[Path]],
        target_size: int = 512,
        use_8bit: bool = True
    ):
        """Copy and process images to output directories"""
        print(f"\n🔄 Processing and copying images...")
        print(f"   Target size: {target_size}x{target_size}")
        print(f"   Using {'8-bit' if use_8bit else '16-bit'} format")
        
        total_images = sum(len(paths) for paths in splits.values())
        processed_count = 0
        failed_count = 0
        
        with tqdm(total=total_images, desc="Processing images") as pbar:
            for split_name, image_paths in splits.items():
                split_dir = self.output_path / split_name
                
                for img_path in image_paths:
                    # Process image
                    processed_img = self.process_image(
                        img_path, 
                        target_size=target_size,
                        use_8bit=use_8bit
                    )
                    
                    if processed_img is not None:
                        # Generate output filename
                        output_filename = f"{img_path.stem}_processed{img_path.suffix}"
                        output_path = split_dir / output_filename
                        
                        # Save processed image
                        if use_8bit:
                            Image.fromarray(processed_img).save(output_path)
                        else:
                            Image.fromarray(processed_img.astype(np.uint16)).save(output_path)
                        
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
        
        self.image_stats['processed_images'] = processed_count
        self.image_stats['failed_images'] = failed_count
        
        print(f"✅ Processing complete:")
        print(f"   - Successfully processed: {processed_count}")
        print(f"   - Failed: {failed_count}")
        
        if failed_count > 0:
            print(f"   - Failed files saved to: {self.output_path}/failed_files.json")
            with open(self.output_path / "failed_files.json", 'w') as f:
                json.dump(self.failed_files, f, indent=2)
    
    def create_dataset_visualization(self, splits: Dict[str, List[Path]]):
        """Create visualization of the prepared dataset"""
        print("\n📈 Creating dataset visualizations...")
        
        # Sample images from each split
        fig, axes = plt.subplots(3, 6, figsize=(18, 9))
        fig.suptitle('CARS Dataset Overview', fontsize=16)
        
        for row, (split_name, image_paths) in enumerate(splits.items()):
            # Sample 6 images from this split
            sample_paths = np.random.choice(
                image_paths, 
                min(6, len(image_paths)), 
                replace=False
            )
            
            for col, img_path in enumerate(sample_paths):
                # Load processed image
                split_dir = self.output_path / split_name
                processed_path = split_dir / f"{img_path.stem}_processed{img_path.suffix}"
                
                if processed_path.exists():
                    img = np.array(Image.open(processed_path))
                    
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].set_title(f'{split_name.upper()}\n{img_path.stem[:15]}...')
                    axes[row, col].axis('off')
                else:
                    axes[row, col].text(0.5, 0.5, 'Image\nNot Found', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "visualizations" / "dataset_overview.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create dataset statistics visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Split distribution
        split_counts = [len(splits[name]) for name in ['train', 'val', 'test']]
        axes[0].pie(split_counts, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
        axes[0].set_title('Dataset Split Distribution')
        
        # Class distribution
        if self.image_stats['healthy_images'] > 0 or self.image_stats['cancerous_images'] > 0:
            class_counts = [
                self.image_stats['healthy_images'],
                self.image_stats['cancerous_images'],
                self.image_stats['total_images'] - self.image_stats['healthy_images'] - self.image_stats['cancerous_images']
            ]
            class_labels = ['Healthy', 'Cancerous', 'Unclassified']
            
            # Remove zero counts
            non_zero_counts = [(count, label) for count, label in zip(class_counts, class_labels) if count > 0]
            if non_zero_counts:
                counts, labels = zip(*non_zero_counts)
                axes[1].pie(counts, labels=labels, autopct='%1.1f%%')
            axes[1].set_title('Class Distribution')
        else:
            axes[1].text(0.5, 0.5, 'No class\ninformation\navailable', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Class Distribution')
        
        # Processing results
        processing_counts = [
            self.image_stats['processed_images'],
            self.image_stats['failed_images']
        ]
        processing_labels = ['Processed', 'Failed']
        axes[2].pie(processing_counts, labels=processing_labels, autopct='%1.1f%%')
        axes[2].set_title('Processing Results')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "visualizations" / "dataset_statistics.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {self.output_path}/visualizations/")
    
    def create_data_config(
        self, 
        target_size: int = 512,
        use_8bit: bool = True,
        batch_size: int = 8
    ):
        """Create Hydra configuration file for the prepared dataset"""
        print("\n⚙️  Creating Hydra configuration...")
        
        config = {
            'dataset_name': 'cars_microscopy_prepared',
            'data_path': str(self.output_path.resolve()),
            'image_size': target_size,
            'channels': 1,
            'use_8bit': use_8bit,
            'batch_size': batch_size,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': True,
            
            'train_ratio': 0.8,  # These are now pre-split
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'split_seed': 42,
            
            'augment_train': True,
            'augment_val': False,
            'augment_test': False,
            
            'normalize_to_unit_range': True,
            'mean': 0.5,
            'std': 0.5,
            
            'validate_images': True,
            'use_cache': True,
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
        
        # Save as YAML
        config_path = self.output_path / "cars_dataset_prepared.yaml"
        
        # Convert to YAML format manually (to avoid external dependency)
        yaml_content = "# Prepared CARS Dataset Configuration\n"
        yaml_content += f"# Generated from: {self.input_path}\n"
        yaml_content += f"# Total images: {self.image_stats['total_images']}\n"
        yaml_content += f"# Processed images: {self.image_stats['processed_images']}\n\n"
        
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
        
        print(f"✅ Configuration saved to: {config_path}")
        
        return config_path
    
    def create_summary_report(self, analysis_results: Dict):
        """Create comprehensive summary report"""
        print("\n📋 Creating summary report...")
        
        report_path = self.output_path / "preparation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# CARS Data Preparation Report\n\n")
            f.write(f"**Source**: {self.input_path}\n")
            f.write(f"**Output**: {self.output_path}\n")
            f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset statistics
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total images found**: {self.image_stats['total_images']}\n")
            f.write(f"- **Successfully processed**: {self.image_stats['processed_images']}\n")
            f.write(f"- **Failed processing**: {self.image_stats['failed_images']}\n")
            f.write(f"- **Success rate**: {self.image_stats['processed_images']/self.image_stats['total_images']*100:.1f}%\n\n")
            
            # Class distribution
            if self.image_stats['healthy_images'] > 0 or self.image_stats['cancerous_images'] > 0:
                f.write("### Class Distribution\n")
                f.write(f"- **Healthy images**: {self.image_stats['healthy_images']}\n")
                f.write(f"- **Cancerous images**: {self.image_stats['cancerous_images']}\n")
                f.write(f"- **Unclassified images**: {self.image_stats['total_images'] - self.image_stats['healthy_images'] - self.image_stats['cancerous_images']}\n\n")
            
            # Bit depth recommendation
            if self.image_stats['bit_depth_recommendation']:
                rec = self.image_stats['bit_depth_recommendation']
                f.write("## Bit Depth Analysis\n\n")
                f.write(f"**Recommendation**: {'Use 8-bit' if rec['use_8bit'] else 'Use 16-bit'}\n")
                f.write(f"**Confidence**: {rec['confidence']}\n\n")
                f.write("**Reasoning**:\n")
                for reason in rec['reasons']:
                    f.write(f"- {reason}\n")
                f.write("\n")
            
            # Training recommendations
            f.write("## Training Recommendations\n\n")
            f.write("Based on your dataset size and the analysis results:\n\n")
            
            if self.image_stats['processed_images'] < 100:
                f.write("⚠️  **Very small dataset** (<100 images):\n")
                f.write("- Use micro model configuration\n")
                f.write("- Enable heavy data augmentation\n")
                f.write("- Consider StyleGAN2-ADA instead\n")
                f.write("- Train for many epochs (1000+)\n\n")
            elif self.image_stats['processed_images'] < 500:
                f.write("📊 **Small dataset** (<500 images):\n")
                f.write("- Use standard FASTGAN configuration\n")
                f.write("- Enable data augmentation\n")
                f.write("- Train for 500-1000 epochs\n")
                f.write("- Monitor for mode collapse\n\n")
            else:
                f.write("✅ **Adequate dataset** (500+ images):\n")
                f.write("- Standard or large configuration should work\n")
                f.write("- Moderate augmentation needed\n")
                f.write("- Train for 300-800 epochs\n\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Review the analysis results** in `analysis/cars_data_analysis_report.md`\n")
            f.write("2. **Check the visualizations** in `visualizations/`\n")
            f.write("3. **Use the generated config** `cars_dataset_prepared.yaml`\n")
            f.write("4. **Start training** with:\n")
            f.write(f"   ```bash\n")
            f.write(f"   python main.py data_path={self.output_path}\n")
            f.write(f"   ```\n\n")
            
            # File structure
            f.write("## Output Structure\n\n")
            f.write("```\n")
            f.write(f"{self.output_path.name}/\n")
            f.write("├── train/              # Training images\n")
            f.write("├── val/                # Validation images\n")
            f.write("├── test/               # Test images\n")
            f.write("├── analysis/           # Data analysis results\n")
            f.write("├── visualizations/     # Dataset visualizations\n")
            f.write("├── cars_dataset_prepared.yaml  # Hydra config\n")
            f.write("└── preparation_report.md       # This report\n")
            f.write("```\n")
        
        print(f"✅ Summary report saved to: {report_path}")
        
        return report_path
    
    def prepare_dataset(
        self,
        target_size: int = 512,
        use_8bit: Optional[bool] = None,
        batch_size: int = 8,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """Complete dataset preparation pipeline"""
        print("🚀 Starting CARS dataset preparation...")
        print("=" * 60)
        
        # Discover images
        image_paths = self.discover_images()
        
        if len(image_paths) == 0:
            print("❌ No images found! Please check your input path.")
            return False
        
        # Analyze data quality
        analysis_results = self.analyze_data_quality(image_paths)
        
        # Use analysis recommendation for bit depth if not specified
        if use_8bit is None:
            use_8bit = analysis_results['bit_depth_analysis']['recommendation']['use_8bit']
            print(f"📊 Using analysis recommendation: {'8-bit' if use_8bit else '16-bit'}")
        
        # Create splits
        splits = self.create_train_val_test_splits(
            image_paths, train_ratio, val_ratio, test_ratio, seed
        )
        
        # Process and copy images
        self.copy_processed_images(splits, target_size, use_8bit)
        
        # Create visualizations
        self.create_dataset_visualization(splits)
        
        # Create configuration
        config_path = self.create_data_config(target_size, use_8bit, batch_size)
        
        # Create summary report
        report_path = self.create_summary_report(analysis_results)
        
        print("\n" + "=" * 60)
        print("🎉 Dataset preparation complete!")
        print(f"📁 Output directory: {self.output_path}")
        print(f"📊 Processed {self.image_stats['processed_images']} images")
        print(f"⚙️  Config file: {config_path}")
        print(f"📋 Report: {report_path}")
        print("=" * 60)
        
        return True


def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description='Prepare CARS microscopy data for FASTGAN training')
    
    parser.add_argument('input_path', type=str, 
                       help='Path to directory containing CARS images')
    parser.add_argument('--output_path', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--target_size', type=int, default=512,
                       help='Target image size (default: 512)')
    parser.add_argument('--use_8bit', type=bool, default=None,
                       help='Force 8-bit conversion (None=auto based on analysis)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training config')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("❌ Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return False
    
    # Create preparator and run
    preparator = CARSDataPreparator(args.input_path, args.output_path)
    
    success = preparator.prepare_dataset(
        target_size=args.target_size,
        use_8bit=args.use_8bit,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    if success:
        print("\n✨ Ready to start training!")
        print(f"Run: python main.py data_path={args.output_path}")
    else:
        print("\n❌ Data preparation failed!")
    
    return success


if __name__ == "__main__":
    main()