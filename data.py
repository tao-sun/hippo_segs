import torch
from torch.utils.data import Dataset
from torchvision.io import read_image



class HippoDataset(Dataset):
    def __init__(self, path, subjects, frames):
        self.path = path
        self.frames = frames
        self.subjects = subjects
        return

    def __getitem__(self, index):
        start_idx = index * self.frames + self.subjects[0]
        images, labels = [], []
        for idx in range(start_idx, start_idx+48):
            img = read_image(f"{self.path}/hippo{idx}.png").squeeze()
            # img.type(torch.float)
            image = img * (1. / 255)
            images.append(image)

            label = read_image(f"{self.path}/label{idx}.png").squeeze()
            label = label * (1. / 255)
            label = label.long()
            labels.append(label)
        
        images = torch.stack(images)
        labels = torch.stack(labels)
        # print(f"images: {images.shape}, labels: {labels.dtype}")
        return images, labels

    def __len__(self):
        return len(self.subjects)