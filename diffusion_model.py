# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================================
# Cell 1: Training Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    image_size = 32
    image_channels = 3
    train_batch_size = 64
    eval_batch_size = 64
    num_epochs = 5
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 2
    save_model_epochs = 2
    dataset = '/kaggle/input/img-align-celeba/img_align_celeba'  # path to the training dataset
    output_dir = 'models/celeba'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    resume = None

training_config = TrainingConfig()


# ==============================================================================
# Cell 2: Data Loading and Preprocessing
# ==============================================================================

# Note: The data path is specific to the Kaggle environment. 
# You will need to modify this path to where you store your dataset locally.
data_path = '/kaggle/input/img-align-celeba/img_align_celeba/' 
img_files = os.listdir(data_path)
img_path = [os.path.join(data_path, i) for i in img_files]

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.images_path = img_path
        self.transform = transform
    def __getitem__(self, index):
        single_img_path = self.images_path[index]
        image = Image.open(single_img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image
    def __len__(self):
        return len(self.images_path)

def get_loader(config):
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = TrainDataset(transform=preprocess)
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    return loader


# ==============================================================================
# Cell 3: UNet Model Components
# ==============================================================================

class TransformerPositionalEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        even_indices = torch.arange(0, self.dimension, 2)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        self.pe_matrix = self.pe_matrix.to(timestep.device)
        return self.pe_matrix[timestep].to(timestep.device)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, input_tensor):
        return self.conv(input_tensor)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super(UpsampleBlock, self).__init__()
        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, input_tensor):
        x = F.interpolate(input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)
        return self.conv(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_channels=None, num_groups=8):
        super(ResNetBlock, self).__init__()
        self.time_embedding_projectile = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_channels, out_channels))
            if time_emb_channels
            else None
        )
        self.block1 = ConvBlock(in_channels, out_channels, groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=num_groups)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding=None):
        input_tensor = x
        h = self.block1(x)
        if self.time_embedding_projectile:
            time_emb = self.time_embedding_projectile(time_embedding)
            time_emb = time_emb[:, :, None, None]
            x = time_emb + h
        x = self.block2(x)
        return x + self.residual_conv(input_tensor)

class SelfAttentionBlock(nn.Module):
    def __init__(self, num_heads, in_channels, num_groups=32, embedding_dim=256):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        self.d_values = embedding_dim // num_heads
        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def split_features_for_heads(self, tensor):
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.stack(tensor.split(channels_per_head, dim=-1), 1)
        return heads_splitted_tensor

    def forward(self, input_tensor):
        x = input_tensor
        batch, features, h, w = x.shape
        x = x.view(batch, features, h * w).transpose(1, 2)
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        queries = self.split_features_for_heads(queries)
        keys = self.split_features_for_heads(keys)
        values = self.split_features_for_heads(values)
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, values)
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        concatenated_heads_attention_scores = attention_scores.view(batch, h * w, self.d_model)
        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(batch, self.d_model, h, w)
        return self.norm(linear_projection + input_tensor)

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, downsample=True):
        super(ConvDownBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            block_in_channels = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_channels=block_in_channels, out_channels=out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels, 2, 1) if downsample else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, upsample=True):
        super(ConvUpBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            block_in_channels = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_channels=block_in_channels, out_channels=out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x

class AttentionDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, num_att_heads, downsample=True):
        super(AttentionDownBlock, self).__init__()
        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            block_in_channels = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_channels=block_in_channels, out_channels=out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
            attention_blocks.append(SelfAttentionBlock(num_att_heads, out_channels, num_groups, out_channels))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.downsample = DownsampleBlock(out_channels, out_channels, 2, 1) if downsample else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, num_att_heads, upsample=True):
        super(AttentionUpBlock, self).__init__()
        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            block_in_channels = in_channels if i == 0 else out_channels
            resnet_blocks.append(ResNetBlock(in_channels=block_in_channels, out_channels=out_channels, time_emb_channels=time_emb_channels, num_groups=num_groups))
            attention_blocks.append(SelfAttentionBlock(num_att_heads, out_channels, num_groups, out_channels))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.upsample = UpsampleBlock(out_channels, out_channels) if upsample else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.upsample:
            x = self.upsample(x)
        return x


# ==============================================================================
# Cell 4: UNet Model Architecture
# ==============================================================================

class UNet(nn.Module):
    def __init__(self, image_size=32, input_channels=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(128, 128, 2, 128 * 4, 32),
            ConvDownBlock(128, 128, 2, 128 * 4, 32),
            ConvDownBlock(128, 256, 2, 128 * 4, 32),
            AttentionDownBlock(256, 256, 2, 128 * 4, 32, 4),
            ConvDownBlock(256, 512, 2, 128 * 4, 32)
        ])
        self.bottleneck = AttentionDownBlock(512, 512, 2, 128 * 4, 32, 4, downsample=False)
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(512 + 512, 512, 2, 128 * 4, 32),
            AttentionUpBlock(512 + 256, 256, 2, 128 * 4, 32, 4),
            ConvUpBlock(256 + 256, 256, 2, 128 * 4, 32),
            ConvUpBlock(256 + 128, 128, 2, 128 * 4, 32),
            ConvUpBlock(128 + 128, 128, 2, 128 * 4, 32)
        ])
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)
        initial_x = self.initial_conv(input_tensor)
        states_for_skip_connections = [initial_x]
        x = initial_x
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)
        
        x = self.bottleneck(x, time_encoded)
        
        for i in range(len(self.upsample_blocks)):
            skip = states_for_skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.upsample_blocks[i](x, time_encoded)
            
        x = torch.cat([x, states_for_skip_connections.pop()], dim=1)
        return self.output_conv(x)


# ==============================================================================
# Cell 5: DDPM Pipeline (Forward & Reverse Diffusion)
# ==============================================================================

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

    def forward_diffusion(self, images, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
        gaussian_noise = torch.randn_like(images)
        alphas_hat_t = self.alphas_hat[timesteps].to(images.device)
        alphas_hat_t = broadcast(alphas_hat_t, images)
        return torch.sqrt(alphas_hat_t) * images + torch.sqrt(1 - alphas_hat_t) * gaussian_noise, gaussian_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, device, save_all_steps=False):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling"):
            ts = torch.full((image.shape[0],), timestep, dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat_t = self.alphas_hat[timestep].to(device)
            
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device) if timestep > 0 else torch.tensor(1.0, device=device)
            
            term1 = 1 / torch.sqrt(alpha_t)
            term2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
            
            image = term1 * (image - term2 * predicted_noise)
            
            if timestep > 0:
                beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat_t) * beta_t
                variance = torch.sqrt(beta_t_hat) * torch.randn_like(image)
                image += variance
                
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image.cpu()


# ==============================================================================
# Cell 6: Utility Functions for Visualization
# ==============================================================================

def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images

def create_images_grid(images, rows, cols):
    images = [Image.fromarray(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def create_sampling_animation(model, pipeline, config, interval=5, every_nth_image=1, rows=2, cols=3):
    noisy_sample = torch.randn(
        config.eval_batch_size, config.image_channels, config.image_size, config.image_size
    ).to(config.device)
    images = pipeline.sampling(model, noisy_sample, device=config.device, save_all_steps=True)
    fig = plt.figure()
    ims = []
    for i in range(0, pipeline.num_timesteps, every_nth_image):
        imgs = postprocess(images[i])
        image_grid = create_images_grid(imgs, rows=rows, cols=cols)
        im = plt.imshow(image_grid, animated=True)
        ims.append([im])
    plt.axis('off')
    animate = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=5000)
    path_to_save_animation = Path(config.output_dir, "samples", "diffusion.gif")
    path_to_save_animation.parent.mkdir(parents=True, exist_ok=True)
    animate.save(str(path_to_save_animation))


# ==============================================================================
# Cell 7: Training and Evaluation Loop
# ==============================================================================

def evaluate(config, epoch, pipeline, model):
    noisy_sample = torch.randn(
        config.eval_batch_size, config.image_channels, config.image_size, config.image_size
    ).to(config.device)
    images = pipeline.sampling(model, noisy_sample, device=config.device)
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=2, cols=3)
    grid_save_dir = Path(config.output_dir, "samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{epoch:04d}.png")

def train():
    config = TrainingConfig()
    device = torch.device(config.device)
    train_dataloader = get_loader(config)
    
    model = UNet(image_size=config.image_size, input_channels=config.image_channels).to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * config.num_epochs, eta_min=1e-9
    )
    
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    diffusion_pipeline = DDPMPipeline(num_timesteps=config.diffusion_timesteps)
    global_step = config.start_epoch * len(train_dataloader)

    for epoch in range(config.start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0.0
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            original_images = batch.to(device)
            batch_size = original_images.shape[0]
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,), device=device).long()
            
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            
            mean_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": mean_loss / (step + 1), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            model.eval()
            evaluate(config, epoch, diffusion_pipeline, model)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': config,
                'epoch': epoch
            }
            output_path = Path(config.output_dir, f"unet_{config.image_size}_e{epoch}.pth")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)

# ==============================================================================
# Cell 8: Inference Function
# ==============================================================================

def run_inference(config, model_path):
    device = torch.device(config.device)
    model = UNet(image_size=config.image_size, input_channels=config.image_channels).to(device)
    diffusion_pipeline = DDPMPipeline(num_timesteps=config.diffusion_timesteps)
    
    # Use torch.load with weights_only=True for security
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    noisy_sample = torch.randn(
        config.eval_batch_size, config.image_channels, config.image_size, config.image_size
    ).to(device)
    
    images = pipeline.sampling(model, noisy_sample, device=device)
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=2, cols=3)
    
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()
    
    # Save the generated image grid
    grid_save_dir = Path(config.output_dir, "inference_samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/generated_grid.png")
    print(f"Inference image saved to {grid_save_dir}/generated_grid.png")


# ==============================================================================
# Main execution block
# ==============================================================================

if __name__ == "__main__":
    # To train the model, uncomment the following line.
    # Make sure you have a dataset available at the specified path in TrainingConfig.
    # train()
    
    # To run inference, you need a trained model checkpoint.
    # Replace 'path/to/your/model.pth' with the actual path to your checkpoint.
    # config = TrainingConfig()
    # model_checkpoint_path = 'path/to/your/model.pth' 
    # if os.path.exists(model_checkpoint_path):
    #     run_inference(config, model_checkpoint_path)
    # else:
    #     print("Model checkpoint not found. Please train a model first or provide the correct path.")
    
    print("Script loaded. To train or run inference, modify the `if __name__ == '__main__':` block.")


