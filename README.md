# DiffIndEye
## Diffusion-Based Retinal OCT Inpainting and Medical Image Reconstruction

DiffIndEye is a research-oriented framework for studying diffusion-based image inpainting on retinal Optical Coherence Tomography (OCT) images. The project combines classical image reconstruction methods with modern denoising diffusion probabilistic models (DDPMs) to investigate how generative models can reconstruct missing retinal anatomy from partially observed OCT scans.

Rather than focusing on retinal disease classification, DiffIndEye reframes OCT imaging as an inverse problem where a clean retinal image is partially corrupted by a mask, producing an observed image. The objective is to reconstruct anatomically plausible retinal structure from incomplete information.

The framework includes:
- retinal OCT preprocessing pipelines,
- synthetic corruption generation,
- classical inpainting baselines,
- conditional diffusion training,
- modular experiment workflows,
- checkpoint-based evaluation,
- and reproducible reconstruction experiments.

The project currently serves as:
1. a mathematical and computational study of medical image inpainting,
2. an experimental diffusion-modeling framework,
3. and a foundation for future retinal reconstruction research.

---

# Features

## Current Capabilities

- Retinal OCT preprocessing pipeline
- Synthetic stripe corruption framework
- Conditional DDPM implementation
- Baseline reconstruction methods:
  - Zero-fill
  - Gaussian smoothing
  - PDE-style diffusion
  - Telea fast marching inpainting
- Diffusion-vs-baseline evaluation
- Diffusion-vs-diffusion checkpoint comparison
- Modular experiment scripts
- Automatic checkpointing and logging
- Notebook-based experimentation workflow

---

# Repository Structure

```text
diff-in-eye/
│
├── data/
│   ├── raw/
│
├── notebooks/
│   ├── 00_dataset_overview.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_inpainting.ipynb
│   ├── 03_pde_diffusion.ipynb
│   ├── 04_ddpm_inpainting.ipynb
│   ├── 05_evaluation_and_comparison.ipynb
│   └── 06_diffindeye_tuning.ipynb
│
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
│
├── scripts/
│   ├── download_data.py
│   ├── train_ddpm.py
│   └── evaluate_methods.py
│
├── src/
│   └── diffindeye/
│       ├── baselines.py
│       ├── config.py
│       ├── data.py
│       ├── diff_models.py
│       ├── diff_sampling.py
│       ├── diff_schedules.py
│       ├── diff_train.py
│       ├── evaluation.py
│       ├── masks.py
│       ├── metrics.py
│       └── preprocess.py
│
├── writeups/
│   ├── main.pdf
│   └── main.tex
|
├── requirements.txt
└── README.md
```

---

# Dataset

This project uses the publicly available retinal OCT dataset:

**Retinal OCT Image Classification - C8**
https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8/data

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diff-in-eye.git
cd diff-in-eye
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

After downloading the Kaggle dataset, place it inside:

```text
data/raw/retinal-oct-c8/
```

The directory structure should resemble:

```text
data/raw/retinal-oct-c8/
└── RetinalOCT_Dataset/
    └── RetinalOCT_Dataset/
        ├── train/
        ├── test/
        └── val/
```

---

# Running the Project

## Training a Diffusion Model

Run the training script:

```bash
PYTHONPATH=src python scripts/train_ddpm.py
```

Example:

```bash
PYTHONPATH=src python scripts/train_ddpm.py \
    --run-name baseline_ddpm \ 
    --n-steps 1000 \ 
    --batch-size 8 \
    --checkpoint-every 250
```

---

# Training Outputs

Training automatically creates:
- timestamped checkpoint folders,
- saved model weights,
- configuration files,
- and training logs.

Example output:

```text
results/checkpoints/
└── baseline_ddpm_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── checkpoint_step_250.pt
    ├── checkpoint_step_500.pt
    └── baseline_ddpm_final_step_1000.pt
```

---

# Evaluating Against Classical Baselines

Evaluate a diffusion checkpoint against:
- zero-fill,
- Gaussian blur,
- PDE diffusion,
- and Telea inpainting.

```bash
PYTHONPATH=src python scripts/evaluate_methods.py \
    --mode baseline-vs-diffusion \
    --checkpoints path/to/checkpoint.pt \
    --n-samples 25 \
    --sample-steps 100
```

---

# Comparing Diffusion Runs

Compare multiple diffusion checkpoints directly:

```bash
PYTHONPATH=src python scripts/evaluate_methods.py \
    --mode diffusion-vs-diffusion \
    --checkpoints checkpoint1.pt checkpoint2.pt
```

---

# Current Limitations

The current baseline DDPM is intentionally lightweight and primarily serves as an experimental foundation. Known limitations include:
- lack of U-Net architecture,
- absence of attention mechanisms,
- blurry reconstruction behavior,
- and weaker performance than Telea inpainting.

Future work will explore:
- U-Net diffusion architectures,
- improved samplers,
- multi-scale feature propagation,
- and uncertainty-aware retinal reconstruction.

---

# Research Motivation

DiffIndEye was developed as a mathematical image analysis and generative modeling project focused on:
- retinal OCT reconstruction,
- medical image inpainting,
- diffusion probabilistic modeling,
- and reproducible experimental workflows.

The broader motivation is to study how probabilistic generative models can reconstruct anatomically meaningful structure under partial observation.

---

# Citation

If referencing this repository, please cite:

```text
Jonathan Ma,
DiffIndEye: Diffusion-Based Retinal OCT Inpainting,
2026.
```

---

# License

This repository is intended for educational and research purposes.