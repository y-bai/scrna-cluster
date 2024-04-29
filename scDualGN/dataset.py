#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    
    def __init__(self, adata):
        self.x = adata.X
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx))