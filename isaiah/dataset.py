import sys, os
from os.path import isdir, abspath
from collections import defaultdict
import glob
import numpy as np
import pandas as pd
from skimage.measure import label
import cv2
import torch
import torchvision
import torchio as tio
import pydicom
import scipy.stats as st
import yaml
from addict import Dict
import h5py
from torch.utils.data import Dataset, DataLoader

#for viewing and troubleshooting
import matplotlib.pyplot as plt

class MammoH5Data(Dataset):

    def __init__(self, datapath, metadata_path, cfgs):
        super().__init__()
        self.datapath =abspath(datapath)
        print(self.datapath)
        self.metadata_path = abspath(metadata_path)
        self.metadata = None
        self.md_fext_map = {
            "json": self._ReadJson,
            "csv": self._ReadCSV
        }
        self.aug_map = defaultdict(self._AugInvalid,
            {
                "contrast_brightness": tio.RescaleIntensity(out_min_max=(0, 1),
                                                            percentiles=(0, 99.5))
            }
        )
        self.aug = cfgs.augmentations
        self.labels = cfgs.labels
        self.ReadMetadata() # reads data into self.metadata
        self.GetDataIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        """outputs a list [tuple, torch.tensor, torch.tensor]"""
        key = self.img_ids[i]
        with h5py.File(self.datapath, "r") as f:
            ds = f.get(key)
            ds_arr = np.zeros_like(ds)
            ds.read_direct(ds_arr)
        md = self.metadata.loc[key, self.labels].to_numpy()
        ds_aug = self.Augment(np.expand_dims(ds_arr, (0, 2)))
        return key, torch.from_numpy(ds_aug), torch.from_numpy(md)

    def _ReadCSV(self):
        self.metadata = pd.read_csv(self.metadata_path)
        return

    def _ReadJson(self):
        self.metadata = pd.read_json(self.metadata_path, orient="index",
                                                 convert_axes=False, convert_dates=False)
        return

    def ReadMetadata(self):
        fext = self.metadata_path.split(".", 1)[-1]
        self.md_fext_map.get(fext, lambda: "Invalid file extension for metadata")()
        return

    def GetDataIds(self):
        self.img_ids = list(self.metadata.index.astype(str))

    def _AugInvalid(self):
        prompt = "Invalid augmentation"
        raise Exception(prompt)

    def Augment(self, im):
        aug_list = []
        for key in self.aug:
            aug_list.append(self.aug_map[key])
        transform = tio.Compose(aug_list)
        im = transform(im)
        return im

if __name__ == "__main__":
    torch.manual_seed(42)
    filepath = "/Users/isaiah/datasets/kaggle_mammograph/preprocessed/mammodata.h5"
    mdpath = "/Users/isaiah/datasets/kaggle_mammograph/preprocessed/metadata.json"
    cfilepath = "/Users/isaiah/GitHub/hana_mammograph/isaiah/dataset_config.yaml"
    cfgs = Dict(yaml.load(open(cfilepath, "r"), Loader=yaml.Loader))
    train_data = MammoH5Data(filepath, mdpath, cfgs)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    img_id, inp, gt = next(iter(train_loader))
    im = inp.numpy().squeeze()
    labels = gt.numpy().squeeze()
    print(img_id)
    print(labels)
    plt.imshow(im, cmap="gray")
    plt.show()
