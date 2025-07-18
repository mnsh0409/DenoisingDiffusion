# config.py
from dataclasses import dataclass

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
    dataset = '/kaggle/input/img-align-celeba/img_align_celeba'  # path to the training dataset (modify it according to your setting)
    output_dir = f'models/{dataset.split("/")[-1]}'
    device = "cuda"
    seed = 0  # random seed
    resume = None

training_config = TrainingConfig()

