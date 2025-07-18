# pipeline.py
import torch
from tqdm import tqdm

def broadcast(values, broadcast_to):
    values = values.flatten()
    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)
    return values

class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = num_timesteps

    def forward_diffusion(self, images, timesteps):
        gaussian_noise = torch.randn_like(images)
        alpha_hat = self.alphas_hat.to(images.device)[timesteps]
        alpha_hat = broadcast(alpha_hat, images)
        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, device, save_all_steps=False):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling"):
            ts = torch.full((image.shape[0],), timestep, device=device, dtype=torch.long)
            predicted_noise = model(image, ts)
            
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)
            
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device) if timestep > 0 else torch.tensor(1.0, device=device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.sqrt(beta_t_hat) * torch.randn_like(image) if timestep > 0 else 0
            
            image = (1 / torch.sqrt(alpha_t)) * (image - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + variance
            
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

