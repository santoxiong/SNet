import numpy as np
import torch
import logging
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self.len = data.shape[0]
        self.reflection = torch.from_numpy(data[:, :-3])
        self.sugar = torch.from_numpy(data[:, -2])
        self.hardness = torch.from_numpy(data[:, :-1])
        logging.info('All data are already!')

    def __len__(self):  # To use len(dataset)
        return self.len

    def __getitem__(self, index):  # To support dataset[index] operation
        return self.reflection, self.sugar, self.hardness
