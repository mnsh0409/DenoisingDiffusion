# inference.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config import training_config
from model import UNet
from pipeline import DDPMPipeline
from utils import postprocess, create_images_grid

def test_model(config, pipeline, model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    noisy_sample = torch.randn(
        config.eval_batch_size,
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    images = pipeline.sampling(model, noisy_sample, device=torch.device(config.device))
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=2, cols=3)

    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device(training_config.device)
    model = UNet(image_size=training_config.image_size,
                 input_channels=training_config.image_channels).to(device)
    diffusion_pipeline = DDPMPipeline(num_timesteps=training_config.diffusion_timesteps)
    
    # Make sure to have the model file at this path
    model_path = '/kaggle/working/models/img_align_celeba/unet32_e4.pth'
    test_model(training_config, diffusion_pipeline, model, model_path)

