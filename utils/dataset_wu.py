import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MyDataset_xinbo(Dataset):
    """Load dataset."""

    def __init__(self, ids, ref_dir, lab_dir, transform=None, ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.ref_dir = ref_dir
        self.labels = lab_dir


        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_path = self.ref_dir + self.ids.iloc[idx, 0] + '.png'
        ref = Image.open(ref_path).convert('RGB')
        # ref = ref.resize((512, 288), resample=Image.LANCZOS)
        ref = ref.resize((480, 270), resample=Image.LANCZOS)
        ref = np.array(ref) / 255.
        ref = np.transpose(ref, (2, 0, 1))
        ref = torch.from_numpy(ref)

        # mos = 4*float(self.ids.iloc[idx, 1])+1
        mos = float(self.labels.loc[idx, 'mos'])
        sample = {'ref': ref, 'mos': mos}

        return sample


def load_data_kfold(filepath, test_set_id=1, val_set_id=2):
    t = pd.read_csv(filepath)

    val_id = t.loc[t.folds == val_set_id]
    test_id = t.loc[t.folds == test_set_id]
    train_id = t.drop(val_id.index).drop(test_id.index)

    return train_id, val_id, test_id