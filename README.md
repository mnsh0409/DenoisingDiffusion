# PyTorch Denoising Diffusion Model

This repository contains a PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM), based on the model architecture described in the paper "Denoising Diffusion Probabilistic Models" by Ho et al.

The model is built with a U-Net architecture incorporating ResNet blocks, positional embeddings for timesteps, and self-attention mechanisms.

## Features

-   **U-Net Architecture:** A robust backbone for image-to-image tasks.
-   **Positional Embeddings:** Transformer-style sinusoidal embeddings to encode diffusion timesteps.
-   **Self-Attention:** Attention blocks at lower resolutions to capture global dependencies.
-   **Forward & Reverse Diffusion:** Complete pipeline for both noising (training) and denoising (sampling).
-   **Training & Inference:** Scripts for training the model on a custom dataset and generating new images from a trained checkpoint.

## How to Use

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your training images in a directory. Update the `dataset` path in the `TrainingConfig` class inside `diffusion_model.py` to point to your dataset folder.

### 3. Training

To train the model, uncomment the `train()` function call at the bottom of `diffusion_model.py` and run the script:

```bash
python diffusion_model.py
```

Model checkpoints and sample images will be saved to the `models/` directory during training.

### 4. Inference

Once you have a trained model checkpoint (`.pth` file), you can generate new images.

1.  Update the `model_checkpoint_path` variable in `diffusion_model.py` to point to your trained model.
2.  Uncomment the `run_inference()` block at the bottom of the script.
3.  Run the script:

```bash
python diffusion_model.py
```

A grid of generated images will be displayed and saved in the `models/inference_samples/` directory.

## Dependencies

-   PyTorch
-   Torchvision
-   NumPy
-   Pandas
-   Matplotlib
-   Pillow
-   tqdm

