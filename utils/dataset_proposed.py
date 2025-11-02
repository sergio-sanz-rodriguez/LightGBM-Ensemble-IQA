import time
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):

    """
    Custom PyTorch Dataset for loading images and their associated MOS (Mean Opinion Score) values.

    Args:
        ids (pd.DataFrame): DataFrame containing image IDs and corresponding MOS scores.
        ref_dir (Path or str): Directory path where image files are stored.
        transform (callable, optional): Optional transform to be applied on a PIL image 
            before converting to tensor. This can include resizing, normalization, augmentation, etc.

    Returns:
        'ref' - image tensor (C, H, W),
        'mos' - float MOS score.
    """

    def __init__(self, ids, ref_dir, transform=None):
        self.ids = ids
        self.ref_dir = ref_dir
        self.transform = transform

    def __len__(self):

        """
        Returns the total number of samples in the dataset.
        """

        return len(self.ids)

    def __getitem__(self, idx):
        
        """
        Retrieves the sample (image and MOS) at the given index.

        Args:
            idx (int or tensor): Index of the sample to retrieve.

        Returns:
            sample (dict): Contains:
                'ref': image tensor processed by the transform pipeline (or manually converted),
                'mos': MOS score as float.
        """

        start = time.time()

        # Handle tensor indices by converting to list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Build image file path using the image ID from the DataFrame
        ref_path = self.ref_dir / f"{self.ids.iloc[idx, 0]}.png"
        # Open image and convert to RGB mode
        ref = Image.open(ref_path).convert('RGB')        

        # Apply transformation pipeline if provided (expects PIL Image input)
        if self.transform:
            ref = self.transform(ref)
        else:
            # Manual fallback: resize, normalize, and convert to tensor
            ref = ref.resize((480, 270), resample=Image.LANCZOS)
            ref = np.array(ref) / 255.0  # Normalize to [0, 1]
            ref = np.transpose(ref, (2, 0, 1))  # Rearrange to (C, H, W)
            ref = torch.from_numpy(ref).float()

        # Retrieve MOS score from the DataFrame and convert to float
        mos = float(self.ids.iloc[idx, 1])
        fold = int(self.ids.iloc[idx, 2])
        name = str(self.ids.iloc[idx, 0])
        prep_time = time.time() - start

        return ref, torch.tensor(mos, dtype=torch.float32), torch.tensor(fold, dtype=torch.int64), name, prep_time