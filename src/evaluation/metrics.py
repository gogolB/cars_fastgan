"""
Evaluation Metrics for FASTGAN
Implements FID, LPIPS, IS, and custom medical imaging metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import cv2
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from sklearn.metrics import pairwise_distances
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")


class FIDScore:
    """Frechet Inception Distance (FID) metric"""
    
    def __init__(self, feature_dim: int = 2048, device: str = 'auto'):
        if not FID_AVAILABLE:
            raise ImportError("pytorch-fid is required for FID calculation")
        
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()
        
    def compute_statistics(self, images: torch.Tensor, batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features"""
        if images.dim() == 3:  # Single channel
            images = images.unsqueeze(1)
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(self.device)
                features = self.model(batch)[0]
                
                # Handle different output shapes
                if features.dim() > 2:
                    features = features.squeeze(-1).squeeze(-1)
                
                features_list.append(features.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor,
        batch_size: int = 50
    ) -> float:
        """Calculate FID between real and fake images"""
        mu1, sigma1 = self.compute_statistics(real_images, batch_size)
        mu2, sigma2 = self.compute_statistics(fake_images, batch_size)
        
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


class LPIPSScore:
    """Learned Perceptual Image Patch Similarity (LPIPS) metric"""
    
    def __init__(self, net: str = 'alex', device: str = 'auto'):
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips is required for LPIPS calculation")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()
    
    def calculate_lpips(
        self, 
        images1: torch.Tensor, 
        images2: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """Calculate LPIPS between two sets of images"""
        if images1.dim() == 3:
            images1 = images1.unsqueeze(1)
        if images2.dim() == 3:
            images2 = images2.unsqueeze(1)
        
        # Convert to RGB if grayscale
        if images1.shape[1] == 1:
            images1 = images1.repeat(1, 3, 1, 1)
        if images2.shape[1] == 1:
            images2 = images2.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] if needed
        if images1.max() <= 1:
            images1 = images1 * 2 - 1
        if images2.max() <= 1:
            images2 = images2 * 2 - 1
        
        scores = []
        
        with torch.no_grad():
            for i in range(0, min(len(images1), len(images2)), batch_size):
                batch1 = images1[i:i + batch_size].to(self.device)
                batch2 = images2[i:i + batch_size].to(self.device)
                
                score = self.model(batch1, batch2)
                scores.append(score.cpu())
        
        return torch.cat(scores).mean().item()


class InceptionScore:
    """Inception Score (IS) metric"""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load pretrained Inception v3
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
    
    def calculate_is(
        self, 
        images: torch.Tensor, 
        batch_size: int = 50, 
        splits: int = 10
    ) -> Tuple[float, float]:
        """Calculate Inception Score"""
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Get predictions
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(self.device)
                pred = F.softmax(self.model(batch), dim=1)
                preds.append(pred.cpu())
        
        preds = torch.cat(preds, dim=0).numpy()
        
        # Calculate IS
        scores = []
        for i in range(splits):
            part = preds[i * len(preds) // splits:(i + 1) * len(preds) // splits]
            kl_divs = []
            
            for p in part:
                kl_div = p * (np.log(p) - np.log(np.mean(part, axis=0)))
                kl_divs.append(np.sum(kl_div))
            
            scores.append(np.exp(np.mean(kl_divs)))
        
        return np.mean(scores), np.std(scores)


class MedicalImagingMetrics:
    """Custom metrics for medical imaging quality assessment"""
    
    def __init__(self):
        pass
    
    def texture_analysis(self, images: torch.Tensor) -> Dict[str, float]:
        """Analyze texture properties of images"""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.ndim == 4:  # Batch dimension
            images = images.squeeze(1)  # Remove channel dimension if single channel
        
        metrics = {
            'lbp_uniformity': [],
            'glcm_contrast': [],
            'glcm_homogeneity': [],
            'glcm_energy': [],
            'gabor_response': [],
            'entropy': []
        }
        
        for img in images:
            # Normalize to 0-255
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Local Binary Pattern
            lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
            hist_lbp, _ = np.histogram(lbp.ravel(), bins=10)
            uniformity = np.sum(hist_lbp ** 2) / (np.sum(hist_lbp) ** 2)
            metrics['lbp_uniformity'].append(uniformity)
            
            # Gray Level Co-occurrence Matrix
            glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            
            metrics['glcm_contrast'].append(contrast)
            metrics['glcm_homogeneity'].append(homogeneity)
            metrics['glcm_energy'].append(energy)
            
            # Gabor filter response
            gabor_real, _ = gabor(img, frequency=0.1)
            gabor_response = np.mean(np.abs(gabor_real))
            metrics['gabor_response'].append(gabor_response)
            
            # Shannon entropy
            entropy = shannon_entropy(img)
            metrics['entropy'].append(entropy)
        
        # Calculate means
        result = {}
        for key, values in metrics.items():
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
        
        return result
    
    def morphological_analysis(self, images: torch.Tensor) -> Dict[str, float]:
        """Analyze morphological properties relevant to thyroid tissue"""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.ndim == 4:
            images = images.squeeze(1)
        
        metrics = {
            'cell_density': [],
            'structure_regularity': [],
            'intensity_variation': []
        }
        
        for img in images:
            # Normalize to 0-255
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Cell density estimation (simplified)
            # Use edge detection to estimate cellular structures
            edges = cv2.Canny(img, 50, 150)
            cell_density = np.sum(edges > 0) / edges.size
            metrics['cell_density'].append(cell_density)
            
            # Structure regularity
            # Calculate standard deviation of local means
            kernel_size = 16
            local_means = []
            h, w = img.shape
            for i in range(0, h - kernel_size, kernel_size // 2):
                for j in range(0, w - kernel_size, kernel_size // 2):
                    patch = img[i:i + kernel_size, j:j + kernel_size]
                    local_means.append(np.mean(patch))
            
            regularity = 1.0 / (1.0 + np.std(local_means))
            metrics['structure_regularity'].append(regularity)
            
            # Intensity variation
            intensity_var = np.std(img) / (np.mean(img) + 1e-8)
            metrics['intensity_variation'].append(intensity_var)
        
        # Calculate means
        result = {}
        for key, values in metrics.items():
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
        
        return result
    
    def diversity_metrics(self, images: torch.Tensor) -> Dict[str, float]:
        """Calculate diversity metrics for generated images"""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.ndim == 4:
            images = images.reshape(images.shape[0], -1)  # Flatten
        
        # Pairwise distances
        distances = pairwise_distances(images, metric='euclidean')
        
        # Remove diagonal (self-distances)
        mask = np.eye(distances.shape[0], dtype=bool)
        distances_no_diag = distances[~mask]
        
        return {
            'mean_pairwise_distance': np.mean(distances_no_diag),
            'std_pairwise_distance': np.std(distances_no_diag),
            'min_pairwise_distance': np.min(distances_no_diag),
            'max_pairwise_distance': np.max(distances_no_diag)
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining multiple metrics"""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Initialize metrics
        if FID_AVAILABLE:
            self.fid_metric = FIDScore(device=device)
        else:
            self.fid_metric = None
            
        if LPIPS_AVAILABLE:
            self.lpips_metric = LPIPSScore(device=device)
        else:
            self.lpips_metric = None
            
        self.is_metric = InceptionScore(device=device)
        self.medical_metrics = MedicalImagingMetrics()
    
    def evaluate(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of generated images"""
        results = {}
        
        print("Computing evaluation metrics...")
        
        # FID Score
        if self.fid_metric is not None:
            try:
                fid_score = self.fid_metric.calculate_fid(real_images, fake_images, batch_size)
                results['fid_score'] = fid_score
                print(f"FID Score: {fid_score:.3f}")
            except Exception as e:
                print(f"Error computing FID: {e}")
                results['fid_score'] = None
        
        # LPIPS Score
        if self.lpips_metric is not None:
            try:
                # Sample pairs for LPIPS
                n_pairs = min(500, len(real_images), len(fake_images))
                real_sample = real_images[:n_pairs]
                fake_sample = fake_images[:n_pairs]
                
                lpips_score = self.lpips_metric.calculate_lpips(real_sample, fake_sample, batch_size)
                results['lpips_score'] = lpips_score
                print(f"LPIPS Score: {lpips_score:.3f}")
            except Exception as e:
                print(f"Error computing LPIPS: {e}")
                results['lpips_score'] = None
        
        # Inception Score
        try:
            is_mean, is_std = self.is_metric.calculate_is(fake_images, batch_size)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            print(f"IS Score: {is_mean:.3f} ± {is_std:.3f}")
        except Exception as e:
            print(f"Error computing IS: {e}")
            results['is_mean'] = None
            results['is_std'] = None
        
        # Medical imaging metrics
        try:
            # Texture analysis
            real_texture = self.medical_metrics.texture_analysis(real_images)
            fake_texture = self.medical_metrics.texture_analysis(fake_images)
            
            results['real_texture'] = real_texture
            results['fake_texture'] = fake_texture
            
            # Morphological analysis
            real_morphology = self.medical_metrics.morphological_analysis(real_images)
            fake_morphology = self.medical_metrics.morphological_analysis(fake_images)
            
            results['real_morphology'] = real_morphology
            results['fake_morphology'] = fake_morphology
            
            # Diversity metrics
            diversity = self.medical_metrics.diversity_metrics(fake_images)
            results['diversity'] = diversity
            
            print("Medical imaging metrics computed successfully")
            
        except Exception as e:
            print(f"Error computing medical metrics: {e}")
        
        return results