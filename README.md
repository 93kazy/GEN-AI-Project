# GAN Sampling Strategies on MNIST: Truncation vs. Discriminator-Driven Latent Sampling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> Academic project carried out as part of my *Generative AI* course

A reproduction and comparative study of two papers — **Wasserstein GAN** (Arjovsky et al., 2017) and **"Your GAN is Secretly an Energy-Based Model and You Should Use Discriminator-Driven Latent Sampling"** (Che et al., 2020) — evaluating how different latent-sampling strategies affect the quality and diversity of generated MNIST digits.

---

## Overview

A GAN's generator is normally fed latent vectors drawn straight from a Gaussian prior. This baseline is simple but mixes high-quality samples with poor ones coming from the low-density tails of the latent space. Several strategies try to improve sample quality by reshaping *where* in latent space we sample from.

This project trains a single **WGAN-GP** backbone on MNIST and then compares **four generation strategies** that all reuse the same trained generator (and, for DDLS, the trained critic):

1. **Classic GAN sampling** — the baseline: latent codes are drawn directly from the prior `z ~ N(0, I)` and decoded by the generator. Maximum diversity, but the tails produce visibly worse digits.
2. **Hard truncation** — the truncation trick (Brock et al., 2019) in its strict form: latent values beyond a fixed threshold are rejected/clamped, discarding the low-density tails. Improves fidelity at the cost of diversity, with a sharp cutoff.
3. **Soft truncation** — a smooth variant: instead of an abrupt cutoff, latent codes are continuously shrunk toward the mode, concentrating probability mass in higher-density regions without the discontinuity of hard truncation.
4. **Discriminator-Driven Latent Sampling (DDLS)** — the principled approach: the trained critic is reinterpreted as an energy model over latent space, and samples are refined with **Langevin dynamics** so they follow the implied higher-density distribution.

The goal is to study the **fidelity/diversity trade-off** across these strategies on a common backbone — see `report.pdf` for the full analysis.

---

## Method

### Backbone architecture

Both the generator and the critic are multilayer perceptrons (MLPs) — the focus is on training dynamics and sampling strategies rather than convolutional inductive biases.

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

### Discriminator-Driven Latent Sampling

Among the strategies above, DDLS is the most involved. Starting from `z ~ N(0, I)`, the sampler minimizes an energy `E(z) = ½‖z‖² − σ(D(G(z)))` over **1000 Langevin steps**, with a cosine-annealed step size (from `5e-3` down to `1e-6`) and injected Gaussian noise. The refined latent codes are then decoded by the generator. This treats the trained critic as a learned correction to the prior, steering samples toward regions the critic considers realistic.

---

## Repository structure

```
GEN-AI-Project/
├── model.py            # Generator and Critic (MLP) definitions
├── train.py            # WGAN-GP adversarial training loop
├── utils.py            # Train steps, gradient penalty, save/load helpers
├── generate.py         # Latent sampling → 10,000 PNGs (DDLS implementation)
├── requirements.txt    # Python dependencies
├── scripts/            # Slurm submission scripts (train.sh, generate.sh)
├── checkpoints/        # Saved weights (G.pth, D.pth)
├── report.pdf          # Project report (full comparison and results)
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

This loads the latest checkpoints from `checkpoints/` and generates 10,000 digits as individual PNG files in `samples/`. As committed, `generate.py` runs the **DDLS** sampler.

<!-- TODO: document how the other strategies (classic / hard truncation / soft truncation) are selected — separate scripts, a CLI flag, or report-only experiments? -->

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

See `report.pdf` for the full comparison of the four sampling strategies, including quantitative metrics and qualitative sample grids.

---

## References

- Goodfellow et al. (2014), *Generative Adversarial Networks.*
- Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN.*
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs.*
- Brock, Donahue & Simonyan (2019), *Large Scale GAN Training for High Fidelity Natural Image Synthesis* (truncation trick).
- Che et al. (2020), *Your GAN is Secretly an Energy-Based Model and You Should Use Discriminator-Driven Latent Sampling.*
