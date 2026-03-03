import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label_index"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)
    


#     🧠 What This Does

# Reads CSV (train/val/test)

# Loads image using OpenCV

# Converts BGR → RGB (VERY IMPORTANT)

# Applies Albumentations transforms

# Returns tensor + label

# This is the foundation of training.