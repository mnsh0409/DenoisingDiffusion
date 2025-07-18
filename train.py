# train.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from config import training_config
from dataset import get_loader
from model import UNet
from pipeline import DDPMPipeline
from utils import postprocess, create_images_grid

def evaluate(config, epoch, pipeline, model):
    noisy_sample = torch.randn(
        config.eval_batch_size,
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    images = pipeline.sampling(model, noisy_sample, device=torch.device(config.device))
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=2, cols=3)

    grid_save_dir = Path(config.output_dir, "samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{epoch:04d}.png")

def main():
    device = torch.device(training_config.device)
    train_dataloader = get_loader(training_config)

    model = UNet(image_size=training_config.image_size,
                 input_channels=training_config.image_channels).to(device)

    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=len(train_dataloader) * training_config.num_epochs,
                                                              eta_min=1e-9)

    if training_config.resume:
        checkpoint = torch.load(training_config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        training_config.start_epoch = checkpoint['epoch'] + 1

    diffusion_pipeline = DDPMPipeline(num_timesteps=training_config.diffusion_timesteps)
    global_step = training_config.start_epoch * len(train_dataloader)

    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        mean_loss = 0.0

        model.train()
        for step, batch in enumerate(train_dataloader):
            original_images = batch.to(device)
            batch_size = original_images.shape[0]
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,), device=device).long()
            
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            mean_loss += loss.item()
            progress_bar.update(1)
            logs = {"loss": mean_loss / (step + 1), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1

        if (epoch + 1) % training_config.save_image_epochs == 0 or epoch == training_config.num_epochs - 1:
            model.eval()
            evaluate(training_config, epoch, diffusion_pipeline, model)

        if (epoch + 1) % training_config.save_model_epochs == 0 or epoch == training_config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': training_config,
                'epoch': epoch
            }
            Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, Path(training_config.output_dir, f"unet{training_config.image_size}_e{epoch}.pth"))

if __name__ == "__main__":
    main()

