# Generative Modeling of Handwritten Digits with a WGAN-GP

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> Academic project carried out as part of the *Generative AI* course of the Master IASD (Artificial Intelligence, Systems, Data), Université Paris Dauphine – PSL (2025–2026). <!-- TODO: confirm course name / year / authors -->

A from-scratch implementation of a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** trained to generate MNIST-like handwritten digits, paired with an advanced **Discriminator-Driven Latent Sampling (DDLS)** procedure at inference time.

---

## Overview

This project tackles image generation on MNIST in two complementary stages:

1. **Adversarial training (WGAN-GP).** A generator and a critic are trained against each other using the Wasserstein distance with a gradient-penalty regularizer, which yields more stable training and avoids the weight-clipping pathologies of the original WGAN.

2. **Discriminator-Driven Latent Sampling (DDLS) at generation time.** Rather than sampling latent vectors directly from the prior, the trained critic is reused as an energy model: latent codes are refined with **Langevin dynamics** so that the final samples land in higher-density regions, improving sample quality without retraining.

Both the generator and the critic are multilayer perceptrons (MLPs) — the goal is to study the GAN/WGAN training dynamics and energy-based sampling on a compact, fully-connected architecture rather than to rely on convolutional inductive biases.

---

## Method

### Architecture

| Component | Layout | Activations | Output |
|-----------|--------|-------------|--------|
| **Generator** | `100 → 256 → 512 → 1024 → 784` | LeakyReLU(0.2) | `Tanh` (pixels in `[-1, 1]`) |
| **Critic** | `784 → 1024 → 512 → 256 → 1` | LeakyReLU(0.2) | linear score (no sigmoid) |

The latent dimension is **100**; images are MNIST digits (`28 × 28 = 784`) flattened into vectors and normalized to `[-1, 1]`.

### Training objective (WGAN-GP)

- **Critic loss:** `E[D(fake)] − E[D(real)] + λ_gp · GP`, with gradient penalty weight `λ_gp = 10`.
- **Generator loss:** `−E[D(fake)]`.
- **Gradient penalty** computed on random interpolations between real and fake samples (Gulrajani et al., 2017).
- **Optimizer:** Adam, `lr = 1e-4`, `betas = (0.0, 0.9)`.
- **LR schedule:** `StepLR` (step size 20, γ = 0.5) for both networks.
- **Critic steps per generator step (`n_critic`):** 3.
- **Generator gradient clipping:** max norm 1.0.
- **Checkpointing:** generator and critic weights saved every 10 epochs to `checkpoints/`.

### Generation (Discriminator-Driven Latent Sampling)

At inference, samples are produced by Langevin dynamics in latent space (Che et al., 2020). Starting from `z ~ N(0, I)`, the code minimizes an energy `E(z) = ½‖z‖² − σ(D(G(z)))` over **1000 steps**, with a cosine-annealed step size (from `5e-3` down to `1e-6`) and injected Gaussian noise. The refined codes are then decoded by the generator. The script writes **10,000** individual PNG images to `samples/`.

---

## Repository structure

```
GEN-AI-Project/
├── model.py            # Generator and Critic (MLP) definitions
├── train.py            # WGAN-GP adversarial training loop
├── utils.py            # Train steps, gradient penalty, save/load helpers
├── generate.py         # DDLS / Langevin sampling → 10,000 PNGs
├── requirements.txt    # Python dependencies
├── scripts/            # Slurm submission scripts (train.sh, generate.sh)
├── checkpoints/        # Saved weights (G.pth, D.pth)
├── report.pdf          # Project report
└── slides.pdf          # Presentation slides
```

---

## Setup

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/93kazy/GEN-AI-Project.git
cd GEN-AI-Project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The code automatically selects the best available device:

- **CUDA** (NVIDIA GPUs) — recommended, with multi-GPU support via `DataParallel`
- **MPS** (Apple Silicon: M1/M2/M3/M4)
- **CPU** (functional but slow, not recommended for training)

---

## Usage

### Training

```bash
python train.py --epochs 100 --lr 1e-4 --batch_size 64
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 64 | Mini-batch size |
| `--gpus` | -1 | Number of GPUs (`-1` = all available) |

MNIST is downloaded automatically into `./data` on first run. To point at an existing dataset directory instead, set the `DATA` environment variable:

```bash
DATA=/path/to/mnist python train.py
```

Checkpoints (`G.pth`, `D.pth`) are written to `checkpoints/` every 10 epochs.

### Generation

```bash
python generate.py --batch_size 2048
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 2048 | Batch size used during latent sampling |

This loads the latest checkpoints from `checkpoints/`, runs the DDLS sampling procedure, and saves 10,000 generated digits as individual PNG files in `samples/`.

---

## Running on a Slurm cluster (optional)

The `scripts/` directory contains ready-to-use submission scripts for an HPC environment (originally the MesoNet *Juliet* cluster). They activate the virtual environment and launch training or generation as batch jobs:

```bash
chmod +x scripts/train.sh
sbatch scripts/train.sh        # or: scripts/train.sh

chmod +x scripts/generate.sh
sbatch scripts/generate.sh
```

You can adjust the learning rate, number of epochs, batch size, and dataset path (`DATA`) directly inside the scripts. If you are not on the target cluster, set `DATA` to your local dataset path (or leave it unset to let MNIST download automatically).

---

## Results

<!-- TODO: add FID / qualitative results, sample grid image, and any figures from report.pdf -->
See `report.pdf` and `slides.pdf` for the full experimental analysis and qualitative results.

---

## References

- Goodfellow et al. (2014), *Generative Adversarial Networks.*
- Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN.*
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs.*
- Che et al. (2020), *Your GAN is Secretly an Energy-Based Model and You Should Use Discriminator-Driven Latent Sampling.*

---

## Author

<!-- TODO: fill in author name(s) and contact -->
Developed by [93kazy](https://github.com/93kazy).

## License

Released under the MIT License. <!-- TODO: confirm license, add a LICENSE file if needed -->
