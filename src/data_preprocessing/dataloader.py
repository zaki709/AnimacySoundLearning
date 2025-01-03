from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image

class AnimacyDataset(Dataset):
    def __init__(self, csv_file_path,transform=None):
        self.data = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path = self.data.iloc[idx,0]
        label = self.data.iloc[idx,1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label,dtype=torch.long)
