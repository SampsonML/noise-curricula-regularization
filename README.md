# Noise Curricula Regularization


This repository implements research on regularization techniques using dynamic noise injection curricula (Curriculum Learning) and compares with optimization strategies such as Sign Descent (SD) and Sharpness-Aware Minimization (SAM). The codebase is built on **JAX/Flax** for model definition and training, with **TensorFlow Datasets (TFDS)** for efficient data pipelines.

## ⚡ Installation

This project is managed with [uv](https://github.com/astral-sh/uv).

### Prerequisites
- Python 3.12
- `uv` installed (recommended):
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
Setup
Clone the repository:


Bash

git clone [https://github.com/sampsonML/noise-curricula-regularization.git](https://github.com/sampsonML/noise-curricula-regularization.git)
cd noise-curricula-regularization
Initialize the environment: Use uv to create a virtual environment and sync dependencies exactly as defined in pyproject.toml.

Bash

uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync pyproject.toml
Alternatively, you can install in editable mode:

Bash

uv pip install -e .
🧠 Models & Methods
The repository features unified ResNet implementations in src/models/resnet.py:

Modern ResNet18: A "Wide" variant using Swish activations and Kaiming initialization (often used for CIFAR).

Standard ResNet34/50: Standard architectures using ReLU and He initialization (standard for ImageNet).

Optimizers Supported:

SGD (with Nesterov momentum)

Adam (AdamW)

SD (Sign Descent with momentum)

SAM (Sharpness-Aware Minimization)

🧪 Running Experiments
The experiments/ folder contains scripts for reproducing training runs on CIFAR-100 and ImageNet.

1. CIFAR-100
Train a Wide ResNet18 with various noise schedules.

Example: Curriculum Noise (Rising Schedule) Injects noise that increases in intensity over the course of training.

Bash

python experiments/cifar100_experiments.py \
    --optimizer SGD \
    --lr 0.1 \
    --rise \
    --use_schedule True \
    --c 0.2 \
    --n_clean 5 \
    --n_noisy 2
Example: SAM Optimizer

Bash

python experiments/cifar100_experiments.py \
    --optimizer SAM \
    --lr 0.05 \
    --rho 0.05 \
    --epochs 100
2. ImageNet
Train a standard ResNet50 on ImageNet (requires manual download of ImageNet data).

Example: Sign Descent (SD)

Bash

python experiments/imagenet_experiments.py \
    --optimizer SD \
    --init_lr 0.001 \
    --schedule cosine \
    --data_dir /path/to/imagenet_data
Example: Standard Baseline

Bash

python experiments/imagenet_experiments.py \
    --optimizer SGD \
    --init_lr 0.1 \
    --schedule cosine \
    --epochs 50 \
    --data_dir /path/to/imagenet_data
📂 Repository Structure
Plaintext

noise-curricula-regularization/
├── pyproject.toml              # Dependencies and metadata (uv/hatchling)
├── README.md                   # Project documentation
├── src/
│   └── models/
│       ├── __init__.py         # Exports ResNet18, ResNet34, ResNet50
│       └── resnet.py           # Unified JAX/Flax ResNet implementations
└── experiments/
    ├── cifar100_experiments.py # Training script for CIFAR-100
    └── imagenet_experiments.py # Training script for ImageNet
📝 Citation
If you use this code in your research, please contact the authors:

Matt L. Sampson (matt.sampson@princeton.edu)
