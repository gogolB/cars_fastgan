# M2 Mac Optimization Report

Generated for system with 128.0 GB RAM

## Recommended Settings

### Optimal Model Configuration
- **Architecture**: Micro
- **Generator filters**: 32
- **Discriminator filters**: 32
- **Generator layers**: 3
- **Discriminator layers**: 2
- **Total parameters**: 270,133,459
- **Memory usage**: 0.01 GB
- **Performance**: 49.3 samples/s

### Optimal Batch Sizes

- **Micro**: Batch size 24 (0.02 GB, 24.5 samples/s)
- **Standard**: Batch size 8 (0.03 GB, 19.8 samples/s)
- **Large**: Batch size 4 (0.05 GB, 8.5 samples/s)

## Configuration for Hydra

```yaml
model:
  generator:
    ngf: 32
    n_layers: 3
  discriminator:
    ndf: 32
    n_layers: 2

data:
  batch_size: 24
```
