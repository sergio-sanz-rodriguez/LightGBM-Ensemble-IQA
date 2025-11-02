import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


label_mapping = {
    "clahe_1": 0,
    "clahe_2": 1,
    "pr_1": 2,
    "pr_2": 3,
    "toning_1": 4,
    "toning_2": 5,
}


class dist_cls(Dataset):
    """Salicon dataset."""

    def __init__(self, ids, stimuli_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.stimuli_dir = stimuli_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        image = image.resize((512, 384), Image.LANCZOS)
        img = np.array(image) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        gt = label_mapping[self.ids.iloc[idx, 1]]
        gt = torch.tensor(gt, dtype=torch.long)
        sample = {'image': img, 'gt': gt}

        return sample