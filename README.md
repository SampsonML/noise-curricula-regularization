# Noise Curricula Regularization
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](#license)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)

This repository implements research on regularization techniques using dynamic noise injection curricula (Curriculum Learning) and compares with optimization strategies such as Sign Descent (SD) and Sharpness-Aware Minimization (SAM). The codebase is built on **JAX/Flax** for model definition and training, with **TensorFlow Datasets (TFDS)** for efficient data pipelines.

## Installation

> [!IMPORTANT]  
> This repository relies on JAX which is well maintained, but also very fast moving. Please use your favorite environment manager and create a fresh env before running this.
> [uv](https://docs.astral.sh/uv/#projects) is particularly nice 

Clone the repository and install dependencies:

```shell
git clone https://github.com/SampsonML/noise-curricula-regularization.git
cd noise-curricula-regularization
pip install -e .
# or using uv
uv sync
```

## Models
Modern ResNet18: A "Wide" variant (used for CIFAR).
Standard ResNet34/50: Standard architectures (used for ImageNet).

Optimizers Supported:

- SGD (with Nesterov momentum)
- Adam (AdamW)
- SD (Sign Descent with momentum: nice review https://arxiv.org/abs/1905.12938)
- SAM (Sharpness-Aware Minimization: https://arxiv.org/abs/2010.01412)

## Running Experiments
The experiments/ folder contains scripts for reproducing training runs on CIFAR-100 and ImageNet.

### CIFAR-100
Train a Wide ResNet18 with various noise schedules.

Example: Curriculum Noise (Rising Schedule) Injects noise that increases in intensity over the course of training.
```shell
python experiments/cifar100_experiments.py \
    --optimizer SGD \
    --lr 0.1 \
    --rise \
    --use_schedule True \
    --c 0.2 \
    --n_clean 5 \
    --n_noisy 2
```

Example: SAM Optimizer
```shell
python experiments/cifar100_experiments.py \
    --optimizer SAM \
    --lr 0.05 \
    --rho 0.05 \
    --epochs 100
```

### ImageNet
Train a standard ResNet50 on ImageNet (requires manual download of ImageNet data).
Example: Sign Descent (SD)
```shell
python experiments/imagenet_experiments.py \
    --optimizer SD \
    --init_lr 0.001 \
    --schedule cosine \
    --data_dir /path/to/imagenet_data
```

Example: Standard Baseline
```shell
python experiments/imagenet_experiments.py \
    --optimizer SGD \
    --init_lr 0.1 \
    --schedule cosine \
    --epochs 50 \
    --data_dir /path/to/imagenet_data
```

## Repository Structure
```text
noise-curricula-regularization/
├── pyproject.toml              # Dependencies and metadata (uv)
├── README.md                   # Project documentation
├── src/
│   └── models/
│       ├── __init__.py         # Exports ResNet18, ResNet34, ResNet50
│       └── resnet.py           # Unified JAX/Flax ResNet implementations
└── experiments/
    ├── cifar100_experiments.py # Training script for CIFAR-100
    └── imagenet_experiments.py # Training script for ImageNet
```
### Citation
If you use this code in your research, please contact the authors:

Matt L. Sampson (matt.sampson@princeton.edu)
