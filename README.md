# CARS-FASTGAN: Advanced Generative Models for CARS Microscopy Image Synthesis

A PyTorch Lightning implementation of FASTGAN optimized for generating synthetic Coherent Anti-Stokes Raman Scattering (CARS) microscopy images of thyroid tissue. This project addresses the unique challenges of non-linear optical microscopy data with sparse, high-dynamic-range features.

## 🎯 Key Features

- **Optimized for Microscopy**: Specialized data processing for 16-bit microscopy images with sparse bright features
- **Hardware Auto-Optimization**: Automatic benchmarking to find optimal model size and batch size for your GPU
- **Multi-Scale Discriminator**: Captures both fine follicle details and global tissue structure
- **Comprehensive Evaluation**: Medical imaging-specific metrics including texture analysis and morphological assessment
- **Production Ready**: PyTorch Lightning integration with logging, checkpointing, and mixed precision training

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Hardware Optimization](#-hardware-optimization)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)

## 🚀 Quick Start

```bash
# 1. Prepare your data
python scripts/cars_data_pipeline.py \
    --input_path /path/to/raw/tif/files \
    --output_path data/processed \
    --task prepare \
    --percentile_low 5.0 \
    --percentile_high 95.0

# 2. Run hardware optimization and auto-launch training
python scripts/cars_training_manager.py \
    --auto_optimize \
    --data_path data/processed

# 3. Monitor training
tensorboard --logdir experiments/logs
```

## 💿 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (16GB+ recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cars_fastgan.git
cd cars_fastgan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
```

## 🔬 Data Preparation

### Understanding CARS Microscopy Data

CARS microscopy produces images with unique characteristics:
- **16-bit depth**: High dynamic range capturing subtle intensity variations
- **Sparse features**: Bright follicles on dark background (often <5% of pixels are bright)
- **Non-linear response**: Intensity scales non-linearly with molecular concentration

### Data Pipeline

The unified data pipeline handles these challenges:

```bash
# Basic preparation with optimized settings for CARS
python scripts/cars_data_pipeline.py \
    --input_path /path/to/raw/tif/files \
    --output_path data/processed \
    --task prepare \
    --percentile_low 5.0 \
    --percentile_high 95.0

# For extremely sparse data, use log transformation
python scripts/cars_data_pipeline.py \
    --input_path /path/to/raw/tif/files \
    --output_path data/processed_log \
    --task prepare \
    --use_log_transform \
    --percentile_low 5.0 \
    --percentile_high 95.0

# Fix existing processed data (if black images issue)
python scripts/cars_data_pipeline.py \
    --input_path data/processed \
    --task fix \
    --percentile_low 5.0 \
    --percentile_high 95.0
```

### Pipeline Tasks

- **`prepare`**: Full processing pipeline (normalization, enhancement, train/val/test split)
- **`enhance`**: Apply enhancement to existing images
- **`analyze`**: Generate detailed statistics and visualizations
- **`fix`**: Repair improperly processed images
- **`split`**: Create train/val/test splits from existing data

### Recommended Settings

| Image Characteristic | Percentile Low | Percentile High | Use Log Transform |
|---------------------|----------------|-----------------|-------------------|
| Normal sparsity | 5.0 | 95.0 | No |
| High sparsity | 5.0 | 95.0 | Yes |
| Dense features | 2.0 | 98.0 | No |
| Mixed density | 3.0 | 97.0 | No |

## 🏋️ Training

### Basic Training

```bash
# Train with default settings
python main.py data_path=data/processed

# Train with specific configuration
python main.py \
    data_path=data/processed \
    experiment_name=my_experiment \
    model.generator.ngf=64 \
    data.batch_size=16 \
    max_epochs=1000
```

### Using Training Manager

The training manager provides high-level orchestration with presets:

```bash
# Use improved preset (recommended)
python scripts/cars_training_manager.py \
    --preset improved \
    --data_path data/processed

# Available presets:
# - baseline: Standard FASTGAN configuration
# - improved: Enhanced stability and quality (recommended)
# - fast: Quick experiments with smaller model
# - high_quality: Large model for best quality

# Launch multiple experiments
python scripts/cars_training_manager.py \
    --experiments micro standard large \
    --data_path data/processed
```

### Model Sizes

| Size | Generator Filters | Discriminator Filters | Layers | Memory Usage | Speed |
|------|------------------|--------------------|--------|--------------|-------|
| micro | 32 | 32 | 3 | ~2GB | Fast |
| small | 48 | 48 | 3 | ~3GB | Fast |
| standard | 64 | 64 | 4 | ~4GB | Moderate |
| large | 96 | 96 | 4 | ~6GB | Slow |
| xlarge | 128 | 128 | 5 | ~8GB | Very Slow |

## 🔧 Hardware Optimization

### Automatic Optimization

The hardware optimizer benchmarks your system to find optimal settings:

```bash
# Run optimization and auto-launch with best config
python scripts/cars_training_manager.py \
    --auto_optimize \
    --data_path data/processed

# Run optimization only (saves results for future use)
python scripts/cars_training_manager.py \
    --optimize_only \
    --data_path data/processed
```

### What It Does

1. **Benchmarks all model sizes** with varying batch sizes
2. **Handles OOM gracefully** by testing until memory limits
3. **Saves recommendations** for future runs
4. **Auto-applies optimal settings** when training

### Using Optimization Results

Once optimization is run, all subsequent training commands automatically use the optimal settings:

```bash
# This will use optimized batch size for your hardware
python scripts/cars_training_manager.py \
    --preset improved \
    --data_path data/processed

# Override optimization with manual settings
python scripts/cars_training_manager.py \
    --preset improved \
    --data_path data/processed \
    --batch_size 8  # Manual override
```

## 📊 Evaluation

### During Training

Training progress is automatically logged:
- **TensorBoard**: Loss curves, generated samples, metrics
- **Checkpoints**: Best models saved based on validation loss
- **Sample Images**: Generated every 25 epochs

```bash
# Monitor training
tensorboard --logdir experiments/logs

# View latest samples
ls experiments/logs/*/version_*/samples/
```

### Post-Training Evaluation

Comprehensive evaluation with medical imaging metrics:

```bash
# Full evaluation suite
python scripts/evaluate_model.py \
    --checkpoint_path experiments/checkpoints/best.ckpt \
    --data_path data/processed \
    --output_dir outputs/evaluation

# Outputs include:
# - FID and IS scores
# - Texture analysis (LBP, GLCM)
# - Morphological metrics
# - Sample grids and comparisons
# - Detailed JSON report
```

### Model Analysis

Analyze training progress and generated images:

```bash
# Analyze model training and outputs
python scripts/cars_model_analyzer.py \
    --mode all \
    --experiment_dir experiments/logs/experiment_name \
    --output_dir outputs/analysis

# Specific analysis modes:
# --mode training: Loss curves and convergence analysis
# --mode samples: Generated image quality metrics
# --mode comparison: Real vs fake comparisons
```

## 📁 Project Structure

```
cars_fastgan/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/                  # Data configurations
│   │   └── cars_dataset.yaml  # CARS-specific settings
│   ├── model/                 # Model configurations
│   │   └── fastgan.yaml       # FASTGAN architecture
│   └── training/              # Training configurations
│       └── default.yaml       # Training hyperparameters
├── src/
│   ├── data/                  # Data handling
│   │   ├── dataset.py         # PyTorch dataset
│   │   └── analyze.py         # Data analysis tools
│   ├── models/                # Model architectures
│   │   └── fastgan.py         # FASTGAN implementation
│   ├── training/              # Training logic
│   │   └── fastgan_module.py  # Lightning module
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # Comprehensive metrics
│   └── visualization/         # Plotting utilities
├── scripts/
│   ├── cars_data_pipeline.py  # Data processing
│   ├── cars_training_manager.py # Training orchestration
│   ├── evaluate_model.py      # Model evaluation
│   ├── cars_model_analyzer.py # Analysis tools
│   └── deprecated/            # Legacy scripts
├── experiments/               # Training outputs
│   ├── logs/                 # TensorBoard logs
│   ├── checkpoints/          # Model checkpoints
│   └── configs/              # Saved configurations
├── outputs/                   # Generated outputs
├── main.py                    # Main training entry
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Black/Dark Generated Images
```bash
# Check data normalization
python scripts/cars_data_pipeline.py \
    --input_path data/processed \
    --task analyze

# Fix if needed
python scripts/cars_data_pipeline.py \
    --input_path data/processed \
    --task fix \
    --percentile_low 5.0 \
    --percentile_high 95.0
```

#### 2. CUDA Out of Memory
```bash
# Run hardware optimization to find optimal batch size
python scripts/cars_training_manager.py \
    --optimize_only \
    --data_path data/processed

# Or manually reduce batch size
python main.py \
    data_path=data/processed \
    data.batch_size=4
```

#### 3. Gradient Clipping Error
The project uses manual optimization for GANs. If you see gradient clipping errors:
```bash
# Already fixed in training manager, but for manual runs:
python main.py \
    data_path=data/processed \
    gradient_clip_val=0  # Disable gradient clipping
```

#### 4. Import Errors
```bash
# Ensure you're in the project root
cd /path/to/cars_fastgan

# Verify Python path includes project
python -c "import sys; print(sys.path)"
```

### Configuration Tips

1. **For sparse microscopy data**: Use `normalize_method: "minmax"` not `"standard"`
2. **For 16-bit images**: Ensure `use_8bit: true` in data config
3. **For stability**: Enable EMA with `+model.training.use_ema=true`

## 🚀 Advanced Usage

### Custom Training Configuration

```bash
# Full custom configuration
python main.py \
    experiment_name=custom_experiment \
    data_path=data/processed \
    data.batch_size=16 \
    data.num_workers=8 \
    model.generator.ngf=96 \
    model.generator.n_layers=4 \
    model.discriminator.ndf=96 \
    model.discriminator.n_layers=4 \
    model.loss.feature_matching_weight=20.0 \
    +model.training.use_gradient_penalty=true \
    +model.training.gradient_penalty_weight=10.0 \
    +model.training.use_ema=true \
    max_epochs=2000 \
    training.log_images_every_n_epochs=10
```

### Resume Training

```bash
# Resume from specific checkpoint
python main.py \
    data_path=data/processed \
    resume_from_checkpoint=experiments/checkpoints/last.ckpt

# Resume with modified settings
python scripts/cars_training_manager.py \
    --preset improved \
    --data_path data/processed \
    --resume_checkpoint experiments/checkpoints/last.ckpt \
    --max_epochs 3000
```

### Multi-GPU Training

```bash
# Use all available GPUs
python main.py \
    data_path=data/processed \
    devices=-1 \
    accelerator=gpu \
    strategy=ddp

# Use specific GPUs
python main.py \
    data_path=data/processed \
    devices=[0,1] \
    accelerator=gpu
```

### Weights & Biases Integration

```bash
# Enable W&B logging
python scripts/cars_training_manager.py \
    --preset improved \
    --data_path data/processed \
    --use_wandb \
    --wandb_project cars_microscopy
```

## 📚 Technical Details

### Why FASTGAN for Microscopy?

1. **Few-shot capability**: Designed for small datasets (100-1000 images)
2. **Skip connections**: Preserve fine follicle structures
3. **Multi-scale discrimination**: Captures both local and global features
4. **Training stability**: Better convergence than traditional GANs

### Key Modifications for CARS

1. **Data Handling**:
   - Percentile normalization for high dynamic range
   - Log transformation option for extreme sparsity
   - Proper uint16 to uint8 conversion

2. **Architecture**:
   - Single channel (grayscale) support
   - Tuned for 512x512 resolution
   - Multi-scale discriminator for follicle patterns

3. **Training**:
   - Feature matching loss weight increased (10.0 → 20.0)
   - Optional gradient penalty for stability
   - EMA for smoother convergence

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- FASTGAN paper: [arXiv:2101.04775](https://arxiv.org/abs/2101.04775)
- PyTorch Lightning team for the excellent framework
- The medical imaging community for domain insights

## 📞 Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This project is optimized for non-linear optical microscopy. For natural images, consider using the original FASTGAN implementation.