import sys, os
import glob
import numpy as np
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader

class MammoData(Dataset):

    def __init__(self, datadir):
        super().__init__()
        self.datadir = os.path.abspath(datadir)
        self.datafiles = glob.glob(os.path.join(self.datadir, "**/*.dcm"),
                                                recursive=True)

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, i):
        filepath = self.datafiles[i]
        ds = pydicom.dcmread(filepath)
        return ds

if __name__ == "__main__":
    pass
