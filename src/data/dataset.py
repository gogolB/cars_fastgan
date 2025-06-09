"""
CARS Microscopy Dataset Module
Handles loading, preprocessing, and augmentation of CARS microscopy images

This module provides:
- Efficient data loading with caching support
- Comprehensive augmentation pipeline for small datasets
- Support for 8-bit and 16-bit images
- Train/validation/test splitting with stratification support
- PyTorch Lightning DataModule for easy integration
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from PIL import Image
import random
from typing import Optional, Tuple, List, Dict, Any, Union
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from functools import lru_cache
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class CARSMicroscopyDataset(Dataset):
    """Dataset class for CARS microscopy images
    
    Features:
    - Automatic detection of image bit depth
    - Optional caching for faster loading
    - Comprehensive augmentation pipeline
    - Support for multiple image formats
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 512,
        use_8bit: bool = True,
        augment: bool = True,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        cache_images: bool = False,
        normalize_method: str = 'standard',
        target_mean: float = 0.5,
        target_std: float = 0.5
    ):
        """
        Initialize CARS microscopy dataset
        
        Args:
            data_path: Path to directory containing images
            image_size: Target image size (assumes square images)
            use_8bit: Whether to convert to 8-bit (recommended after analysis)
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'test')
            transform: Custom transforms to apply
            cache_images: Whether to cache images in memory
            normalize_method: Normalization method ('standard', 'minmax', 'custom')
            target_mean: Target mean for normalization (in [0, 1])
            target_std: Target std for normalization (in [0, 1])
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.use_8bit = use_8bit
        self.augment = augment and (split == 'train')
        self.split = split
        self.cache_images = cache_images
        self.normalize_method = normalize_method
        self.target_mean = target_mean
        self.target_std = target_std
        
        # Find all image files
        self.image_paths = self._find_image_files()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
            
        logger.info(f"Found {len(self.image_paths)} images for {split} split")
        
        # Set up transforms
        self.transform = transform or self._get_default_transforms()
        
        # Cache for loaded images
        self._image_cache = {} if cache_images else None
        
        # Precompute normalization statistics if needed
        if normalize_method == 'custom':
            self._compute_dataset_statistics()
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the data directory
        
        Returns:
            List of paths to image files
        """
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        image_paths = []
        
        for ext in image_extensions:
            # Case-insensitive search
            image_paths.extend(list(self.data_path.rglob(f'*{ext}')))
            image_paths.extend(list(self.data_path.rglob(f'*{ext.upper()}')))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        return image_paths
    
    def _compute_dataset_statistics(self):
        """Compute dataset statistics for normalization"""
        logger.info("Computing dataset statistics for normalization...")
        
        means = []
        stds = []
        
        # Sample a subset for efficiency
        sample_size = min(100, len(self.image_paths))
        sample_indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
        
        for idx in sample_indices:
            img = self._load_and_preprocess_image(self.image_paths[idx])
            if self.use_8bit:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32) / 65535.0
                
            means.append(np.mean(img))
            stds.append(np.std(img))
        
        self.dataset_mean = np.mean(means)
        self.dataset_std = np.mean(stds)
        
        logger.info(f"Dataset statistics: mean={self.dataset_mean:.4f}, std={self.dataset_std:.4f}")
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default image transforms using Albumentations
        
        Returns:
            Composition of transforms
        """
        transform_list = []
        
        # Always ensure correct size first
        transform_list.append(
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_AREA)
        )
        
        # Augmentations for training
        if self.augment:
            transform_list.extend([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # More conservative rotation for medical images
                A.Rotate(
                    limit=15, 
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7
                ),
                
                # Slight shifts and scaling
                A.ShiftScaleRotate(
                    shift_limit=0.0625,  # More conservative than default
                    scale_limit=0.1, 
                    rotate_limit=0,  # Rotation handled separately
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7
                ),
                
                # Intensity transforms (conservative for medical images)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    brightness_by_max=True,
                    p=0.5
                ),
                
                # Gamma correction
                A.RandomGamma(
                    gamma_limit=(90, 110),  # Conservative range
                    p=0.5
                ),
                
                # Noise (very light for medical images)
                A.GaussNoise(
                    var_limit=(10.0, 30.0),  # Reduced noise
                    mean=0,
                    p=0.3
                ),
                
                # Blur (occasional light blur)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                
                # Medical imaging specific - elastic deformation
                A.ElasticTransform(
                    alpha=120,  # Reduced from default
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.3
                ),
                
                # Grid distortion (simulates slight optical distortions)
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.2
                ),
            ])
        
        # Normalization based on method
        if self.normalize_method == 'standard':
            # Standard ImageNet-style normalization adapted for grayscale
            transform_list.append(
                A.Normalize(mean=[self.target_mean], std=[self.target_std])
            )
        elif self.normalize_method == 'minmax':
            # Min-max normalization to [-1, 1]
            transform_list.append(
                A.Normalize(mean=[0.5], std=[0.5])
            )
        elif self.normalize_method == 'custom':
            # Use computed dataset statistics
            transform_list.append(
                A.Normalize(
                    mean=[getattr(self, 'dataset_mean', 0.5)],
                    std=[getattr(self, 'dataset_std', 0.5)]
                )
            )
        
        # Convert to tensor
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)
    
    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Check cache first
        if self.cache_images and str(image_path) in self._image_cache:
            return self._image_cache[str(image_path)].copy()
        
        try:
            # Load image
            img = np.array(Image.open(image_path))
            
            # Handle different bit depths
            if img.dtype == np.uint16:
                if self.use_8bit:
                    # Convert 16-bit to 8-bit using percentile scaling
                    p_low, p_high = np.percentile(img, (0.1, 99.9))
                    img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
                    img = (img * 255).astype(np.uint8)
                else:
                    # Keep as 16-bit float32
                    img = img.astype(np.float32)
            elif img.dtype != np.uint8:
                # Convert other types to uint8
                img = img.astype(np.uint8)
            
            # Ensure single channel
            if len(img.shape) == 3:
                # Convert to grayscale if needed
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    img = img[:, :, 0]  # Take first channel
            
            # Cache if enabled
            if self.cache_images:
                self._image_cache[str(image_path)] = img.copy()
            
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image of the correct size and type
            if self.use_8bit:
                return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing image tensor and metadata
        """
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Ensure tensor format and correct shape
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure shape is [C, H, W]
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
            # Channel last to channel first
            image = image.permute(2, 0, 1)
        
        return {
            'image': image,
            'path': str(image_path),
            'filename': image_path.name,
            'index': idx
        }
    
    def get_sample_batch(self, batch_size: int = 8) -> torch.Tensor:
        """Get a sample batch for visualization
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Batch of images
        """
        indices = np.random.choice(len(self), min(batch_size, len(self)), replace=False)
        images = []
        
        for idx in indices:
            item = self[idx]
            images.append(item['image'])
        
        return torch.stack(images)


class CARSDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CARS microscopy data
    
    This DataModule handles:
    - Automatic train/val/test splitting
    - Efficient data loading with multiple workers
    - Consistent preprocessing across splits
    - Easy integration with PyTorch Lightning
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 512,
        use_8bit: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        drop_last: bool = True,
        augment_train: bool = True,
        augment_val: bool = False,
        augment_test: bool = False,
        cache_images: bool = False,
        normalize_method: str = 'standard',
        persistent_workers: bool = True
    ):
        """
        Initialize CARS DataModule
        
        Args:
            data_path: Path to directory containing all images
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            image_size: Target image size
            use_8bit: Whether to use 8-bit conversion
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducible splits
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            augment_train: Whether to augment training data
            augment_val: Whether to augment validation data
            augment_test: Whether to augment test data
            cache_images: Whether to cache images in memory
            normalize_method: Normalization method
            persistent_workers: Whether to keep workers alive between epochs
        """
        super().__init__()
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Data path
        self.data_path = Path(data_path)
        
        # Attributes that will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Store paths for reference
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
    
    def prepare_data(self):
        """Download or prepare data if needed
        
        This method is called only once and on a single GPU.
        Use it for downloading or preparing data.
        """
        # For CARS data, we assume it's already available
        # This method could be used for downloading from a remote source
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        # Find all image files
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        all_image_paths = []
        
        for ext in image_extensions:
            all_image_paths.extend(list(self.data_path.rglob(f'*{ext}')))
            all_image_paths.extend(list(self.data_path.rglob(f'*{ext.upper()}')))
        
        # Remove duplicates and sort
        all_image_paths = sorted(list(set(all_image_paths)))
        
        if len(all_image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")
        
        logger.info(f"Found {len(all_image_paths)} total images")
        
        # Split data
        # First split: train+val vs test
        train_val_paths, test_paths = train_test_split(
            all_image_paths, 
            test_size=self.hparams.test_ratio,
            random_state=self.hparams.seed
        )
        
        # Second split: train vs val
        val_size_adjusted = self.hparams.val_ratio / (self.hparams.train_ratio + self.hparams.val_ratio)
        train_paths, val_paths = train_test_split(
            train_val_paths,
            test_size=val_size_adjusted,
            random_state=self.hparams.seed
        )
        
        logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
        
        # Store paths
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(train_paths, "train", self.hparams.augment_train)
            self.val_dataset = self._create_dataset(val_paths, "val", self.hparams.augment_val)
        
        if stage == "validate":
            self.val_dataset = self._create_dataset(val_paths, "val", self.hparams.augment_val)
        
        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(test_paths, "test", self.hparams.augment_test)
    
    def _create_dataset(self, paths: List[Path], split: str, augment: bool) -> CARSMicroscopyDataset:
        """Create a dataset from a list of image paths
        
        Args:
            paths: List of image paths
            split: Dataset split name
            augment: Whether to apply augmentation
            
        Returns:
            CARSMicroscopyDataset instance
        """
        # Create a temporary directory structure for compatibility
        # with the existing dataset class
        dataset = PathBasedCARSDataset(
            image_paths=paths,
            image_size=self.hparams.image_size,
            use_8bit=self.hparams.use_8bit,
            augment=augment,
            split=split,
            cache_images=self.hparams.cache_images,
            normalize_method=self.hparams.normalize_method
        )
        
        return dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (uses test dataset)"""
        return self.test_dataloader()


class PathBasedCARSDataset(CARSMicroscopyDataset):
    """Dataset that works with pre-defined image paths
    
    This is a convenience class for when you already have
    lists of paths for each split.
    """
    
    def __init__(
        self, 
        image_paths: List[Path],
        image_size: int,
        use_8bit: bool,
        augment: bool,
        split: str,
        cache_images: bool = False,
        normalize_method: str = 'standard'
    ):
        """
        Initialize with pre-defined paths
        
        Args:
            image_paths: List of image paths
            image_size: Target image size
            use_8bit: Whether to convert to 8-bit
            augment: Whether to apply augmentation
            split: Dataset split name
            cache_images: Whether to cache images
            normalize_method: Normalization method
        """
        self.image_paths = image_paths
        self.image_size = image_size
        self.use_8bit = use_8bit
        self.augment = augment
        self.split = split
        self.cache_images = cache_images
        self.normalize_method = normalize_method
        self.target_mean = 0.5
        self.target_std = 0.5
        
        # Set up transforms
        self.transform = self._get_default_transforms()
        
        # Cache for loaded images
        self._image_cache = {} if cache_images else None
        
        logger.info(f"Created {split} dataset with {len(image_paths)} images")