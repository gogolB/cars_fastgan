# CARS-FASTGAN: Thyroid Tissue Image Generation

A FASTGAN implementation optimized for generating synthetic CARS (Coherent Anti-Stokes Raman Scattering) microscopy images of thyroid tissue from limited datasets.

## 🔬 Project Overview

This project implements FASTGAN for generating high-quality synthetic CARS microscopy images to augment a limited dataset of 448 thyroid tissue images. FASTGAN is specifically designed for training on small datasets (100-1000 images) and includes key features like skip connections and multi-scale discrimination.

### Key Features

- **FASTGAN Architecture**: Optimized for small datasets with skip connections
- **Multi-scale Discrimination**: Improved training stability
- **Comprehensive Evaluation**: FID, LPIPS, IS, and custom medical imaging metrics
- **Medical Imaging Optimizations**: Texture analysis, morphological assessment
- **M2 Mac Optimized**: Efficient training on Apple Silicon
- **Hydra Configuration**: Flexible experiment management
- **PyTorch Lightning**: Clean, scalable training code

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- 16GB+ RAM recommended

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd cars_fastgan

# Install dependencies
pip install -r requirements.txt

# Install additional packages for evaluation
pip install albumentations pytorch-fid lpips pyiqa
```

### 2. Data Analysis (Recommended First Step)

Before training, analyze your data to determine optimal preprocessing:

```bash
python src/data/analyze.py \
    --data_path /path/to/your/cars/images \
    --output_dir ./outputs/analysis \
    --sample_size 50
```

This will:
- Analyze 16-bit vs 8-bit conversion impact
- Generate visualization comparisons
- Provide recommendations for preprocessing
- Create comprehensive analysis report

### 3. Basic Training

```bash
# Basic training with default settings
python main.py data_path=/path/to/your/cars/images

# Training with custom settings
python main.py \
    data_path=/path/to/your/cars/images \
    data.use_8bit=true \
    data.batch_size=8 \
    model.generator.ngf=64 \
    max_epochs=1000
```

### 4. Monitor Training

```bash
# View tensorboard logs
tensorboard --logdir experiments/logs

# If using Weights & Biases
python main.py use_wandb=true wandb.project=your_project_name
```

## 📊 Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/data/cars_dataset.yaml`: Data loading and augmentation
- `configs/model/fastgan.yaml`: Model architecture
- `configs/training/default.yaml`: Training parameters
- `configs/evaluation/default.yaml`: Evaluation metrics

### Key Configuration Options

```yaml
# Data settings
data:
  batch_size: 8          # Small batch for limited dataset
  use_8bit: true         # Based on analysis results
  augment_train: true    # Enable augmentation

# Model settings
model:
  generator:
    latent_dim: 256
    ngf: 64               # Generator filters
    use_skip_connections: true
  discriminator:
    ndf: 64               # Discriminator filters
    use_multiscale: true  # Multi-scale discrimination

# Training settings
max_epochs: 1000
log_images_every_n_epochs: 25
```

## 🔧 Advanced Usage

### Custom Model Variants

For very small datasets (< 200 images):
```bash
python main.py model.active_variant=micro
```

For larger compute resources:
```bash
python main.py model.active_variant=large
```

### Experiment Tracking

```bash
# With custom experiment name
python main.py experiment_name=my_experiment

# With Weights & Biases
python main.py \
    use_wandb=true \
    wandb.project=cars-fastgan \
    wandb.entity=your_username \
    experiment_name=baseline_run
```

### Resume Training

```bash
python main.py \
    resume_from_checkpoint=experiments/checkpoints/last.ckpt
```

## 📈 Evaluation

### During Training
- FID scores computed every 50 epochs
- Sample images logged every 25 epochs
- Comprehensive metrics in tensorboard

### Post-Training Evaluation
```bash
# Comprehensive evaluation
python src/evaluation/evaluate_model.py \
    --checkpoint_path experiments/checkpoints/best.ckpt \
    --data_path /path/to/your/data \
    --output_dir outputs/evaluation
```

### Custom Evaluation
```python
from src.evaluation.metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate(real_images, fake_images)
print(f"FID Score: {results['fid_score']:.3f}")
```

## 🔬 Medical Imaging Metrics

The project includes specialized metrics for medical imaging:

- **Texture Analysis**: LBP uniformity, GLCM properties, Gabor responses
- **Morphological Analysis**: Cell density, structure regularity
- **Diversity Metrics**: Mode collapse detection
- **Clinical Relevance**: Tissue-specific quality assessment

## 📁 Project Structure

```
cars_fastgan/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── data/               # Data configurations
│   ├── model/              # Model configurations
│   ├── training/           # Training configurations
│   └── evaluation/         # Evaluation configurations
├── src/
│   ├── data/               # Data loading and processing
│   │   ├── dataset.py      # Dataset classes
│   │   └── analyze.py      # Data analysis tool
│   ├── models/             # Model architectures
│   │   └── fastgan.py      # FASTGAN implementation
│   ├── training/           # Training modules
│   │   └── fastgan_module.py # Lightning training module
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # Comprehensive metrics
│   └── visualization/      # Plotting and visualization
├── experiments/            # Training outputs
│   ├── logs/              # Tensorboard logs
│   └── checkpoints/       # Model checkpoints
├── outputs/               # Generated outputs
│   ├── images/            # Generated samples
│   ├── metrics/           # Evaluation results
│   └── reports/           # Analysis reports
├── main.py                # Main training script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎯 Tips for Success

### Data Preparation
1. Run the analysis tool first to determine optimal bit depth
2. Ensure consistent image formats and sizes
3. Consider your train/val/test split ratios

### Training Tips
1. Start with smaller models for faster iteration
2. Monitor FID scores - they should decrease over time
3. Use feature matching loss (enabled by default)
4. Be patient - GANs need many epochs on small datasets

### Common Issues
- **Mode collapse**: Increase feature matching weight
- **Training instability**: Reduce learning rates
- **Poor quality**: Try different GAN loss types
- **Memory issues**: Reduce batch size

## 📚 References

- [FASTGAN Paper](https://arxiv.org/abs/2101.04775): "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis"
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- [Hydra Configuration](https://hydra.cc/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙋‍♂️ Support

For questions about:
- **CARS microscopy**: Consult domain experts
- **Code issues**: Open GitHub issues
- **Configuration**: Check Hydra documentation
- **Training problems**: Review PyTorch Lightning guides

## 🏆 Acknowledgments

- Original FASTGAN authors for the architecture
- PyTorch Lightning team for the framework
- Medical imaging community for evaluation insights