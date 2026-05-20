# Retinal OCT Diffusion Inpainting

## Overview

This project explores diffusion-based image inpainting for retinal Optical Coherence Tomography (OCT) images. The goal is to reconstruct missing or corrupted regions of retinal scans using stochastic generative modeling while developing a strong foundation in image preprocessing, medical image analysis, and diffusion-based reconstruction methods. The project uses the publicly available retinal OCT classification dataset from Kaggle: ![Retinal OCT Image Classification - C8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8/data) 

Rather than focusing on disease classification, this project reframes the dataset as an inverse imaging problem, where given a partially observed OCT image, can a diffusion model reconstruct anatomically plausible retinal structure?