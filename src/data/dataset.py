"""
CARS Microscopy Dataset Module
Handles loading, preprocessing, and augmentation of CARS microscopy images
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
from typing import Optional, Tuple, List, Dict, Any
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CARSMicroscopyDataset(Dataset):
    """Dataset class for CARS microscopy images"""
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 512,
        use_8bit: bool = True,
        augment: bool = True,
        split: str = 'train',
        transform: Optional[A.Compose] = None
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
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.use_8bit = use_8bit
        self.augment = augment and (split == 'train')
        self.split = split
        
        # Find all image files
        self.image_paths = self._find_image_files()
        print(f"Found {len(self.image_paths)} images for {split} split")
        
        # Set up transforms
        self.transform = transform or self._get_default_transforms()
        
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the data directory"""
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(self.data_path.rglob(f'*{ext}')))
            image_paths.extend(list(self.data_path.rglob(f'*{ext.upper()}')))
        
        return sorted(image_paths)
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default image transforms using Albumentations"""
        transform_list = []
        
        # Resize if needed
        if self.image_size != 512:  # Assuming original is 512x512
            transform_list.append(A.Resize(self.image_size, self.image_size))
        
        # Augmentations for training
        if self.augment:
            transform_list.extend([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.7
                ),
                
                # Intensity transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                
                # Medical imaging specific
                A.ElasticTransform(
                    alpha=100, 
                    sigma=10, 
                    alpha_affine=10, 
                    p=0.3
                ),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            A.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ToTensorV2()
        ])
        
        return A.Compose(transform_list)
    
    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image
            img = np.array(Image.open(image_path))
            
            # Convert to 8-bit if needed
            if img.dtype == np.uint16 and self.use_8bit:
                img = (img / 65535.0 * 255).astype(np.uint8)
            elif img.dtype == np.uint16 and not self.use_8bit:
                # Keep as 16-bit but normalize
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.uint8)
            
            # Ensure single channel
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return img
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image of the correct size
            if self.use_8bit:
                return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path)
        
        # Apply transforms
        if self.transform:
            # Albumentations expects image in HWC format
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Ensure tensor format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure correct shape [C, H, W]
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        
        return {
            'image': image,
            'path': str(image_path),
            'index': idx
        }


class CARSDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CARS microscopy data"""
    
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
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_8bit = use_8bit
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.augment_train = augment_train
        self.augment_val = augment_val
        self.augment_test = augment_test
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
    
    def prepare_data(self):
        """Download or prepare data if needed"""
        # For CARS data, this would be where we'd download or prepare the data
        # In this case, we assume the data is already available
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage"""
        # Find all image files
        data_path = Path(self.data_path)
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        all_image_paths = []
        
        for ext in image_extensions:
            all_image_paths.extend(list(data_path.rglob(f'*{ext}')))
            all_image_paths.extend(list(data_path.rglob(f'*{ext.upper()}')))
        
        all_image_paths = sorted(all_image_paths)
        print(f"Found {len(all_image_paths)} total images")
        
        # Split data
        if len(all_image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")
        
        # First split: train+val vs test
        train_val_paths, test_paths = train_test_split(
            all_image_paths, 
            test_size=self.test_ratio,
            random_state=self.seed
        )
        
        # Second split: train vs val
        train_paths, val_paths = train_test_split(
            train_val_paths,
            test_size=self.val_ratio / (self.train_ratio + self.val_ratio),
            random_state=self.seed
        )
        
        print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
        
        # Store paths for reference
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        
        # Create temporary directories with split data (or use custom dataset)
        # For simplicity, we'll pass the paths directly to the dataset
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset_from_paths(
                train_paths, "train", self.augment_train
            )
            self.val_dataset = self._create_dataset_from_paths(
                val_paths, "val", self.augment_val
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset_from_paths(
                test_paths, "test", self.augment_test
            )
    
    def _create_dataset_from_paths(self, paths: List[Path], split: str, augment: bool):
        """Create a dataset from a list of image paths"""
        # Create a temporary dataset class that uses the provided paths
        class PathBasedDataset(CARSMicroscopyDataset):
            def __init__(self, paths, image_size, use_8bit, augment, split):
                self.image_paths = paths
                self.image_size = image_size
                self.use_8bit = use_8bit
                self.augment = augment
                self.split = split
                self.transform = self._get_default_transforms()
        
        return PathBasedDataset(paths, self.image_size, self.use_8bit, augment, split)
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )