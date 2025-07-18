# PyTorch Denoising Diffusion Model

This repository contains a PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM), based on the model architecture described in the paper "Denoising Diffusion Probabilistic Models" by Ho et al.

The model is built with a U-Net architecture incorporating ResNet blocks, positional embeddings for timesteps, and self-attention mechanisms.

## Features

-   **U-Net Architecture:** A robust backbone for image-to-image tasks.
-   **Positional Embeddings:** Transformer-style sinusoidal embeddings to encode diffusion timesteps.
-   **Self-Attention:** Attention blocks at lower resolutions to capture global dependencies.
-   **Forward & Reverse Diffusion:** Complete pipeline for both noising (training) and denoising (sampling).
-   **Training & Inference:** Scripts for training the model on a custom dataset and generating new images from a trained checkpoint.

## Code Structure

The main logic is contained within `diffusion_model.py`. The script is organized into the following sections:

-   **`TrainingConfig`:** A dataclass holding all hyperparameters for training, data paths, and model saving.
-   **Data Loading:**
    -   `TrainDataset`: A custom PyTorch `Dataset` class to load images from a specified folder.
    -   `get_loader`: A function that sets up the data transformations and returns a `DataLoader`.
-   **UNet Model Components:**
    -   `TransformerPositionalEmbedding`: Creates sinusoidal embeddings for the diffusion timesteps.
    -   `ConvBlock`, `DownsampleBlock`, `UpsampleBlock`: Basic building blocks for the U-Net.
    -   `ResNetBlock`: A residual block that incorporates timestep embeddings.
    -   `SelfAttentionBlock`: A self-attention mechanism for capturing global image features.
    -   `ConvDownBlock`, `ConvUpBlock`, `AttentionDownBlock`, `AttentionUpBlock`: Higher-level blocks that combine ResNet and Attention layers for the U-Net's encoder and decoder paths.
-   **`UNet` Class:** The complete U-Net model architecture that assembles all the components.
-   **`DDPMPipeline` Class:**
    -   `forward_diffusion`: Implements the forward noising process.
    -   `sampling`: Implements the reverse denoising process (image generation).
-   **Utility Functions:** Helper functions for post-processing images (`postprocess`), creating image grids (`create_images_grid`), and generating animations (`create_sampling_animation`).
-   **`train()` function:** The main training loop that handles optimization, loss calculation, and saving checkpoints.
-   **`run_inference()` function:** A function to load a trained model and generate sample images.

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
