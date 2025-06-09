#!/bin/bash
# RunPod training script for CARS-FASTGAN with large model

# Setup environment
cd /workspace
git clone https://github.com/YOUR_USERNAME/cars_fastgan.git
cd cars_fastgan

# Install dependencies
pip install -r requirements.txt
pip install wandb

# Copy data (adjust path as needed)
# You'll need to upload your fixed data to RunPod first
cp -r /workspace/data_fixed/* data/

# Login to wandb (optional)
# wandb login YOUR_API_KEY

# Launch training with large model
python main.py \
    experiment_name=cars_fastgan_large_20250609_094346 \
    data_path=data/processed_fixed \
    model.generator.ngf=128 \
    model.generator.n_layers=5 \
    model.discriminator.ndf=128 \
    model.discriminator.n_layers=4 \
    model.discriminator.num_scales=3 \
    model.loss.feature_matching_weight=20.0 \
    model.optimizer.generator.lr=0.0001 \
    model.optimizer.discriminator.lr=0.0004 \
    model.training.use_gradient_penalty=true \
    model.training.use_ema=true \
    max_epochs=2000 \
    data.batch_size=32 \
    accelerator=gpu \
    devices=1 \
    precision=16-mixed \
    data.num_workers=4 \
    use_wandb=true \
    wandb.project=cars-fastgan-large
