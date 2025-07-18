# dataset.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# It's recommended to use argparse or a similar library to handle paths
# For now, we'll keep it as a global variable
data_path = '/kaggle/input/img-align-celeba/img_align_celeba/'

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.images_path = [os.path.join(data_path, i) for i in os.listdir(data_path)]
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

