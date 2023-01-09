import sys, os
import glob
import numpy as np
from skimage.measure import label
import cv2
import torch
import torchvision
import pydicom
import scipy.stats as st
from torch.utils.data import Dataset, DataLoader

#for viewing and troubleshooting
import matplotlib.pyplot as plt

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

    def getFileList(self):
        return self.datafiles

if __name__ == "__main__":
    datadir = "/Users/isaiah/datasets/kaggle_mammograph/"
    data = MammoData(datadir)
    fileList = data.getFileList()
    for i in range(len(data)):
        file = fileList[i]
        file_woext = os.path.splitext(file)[0]
        name = os.path.basename(file_woext)
        ds = data[i]
        ds_arr = ds.pixel_array
        mask = np.ones_like(ds_arr, dtype=np.uint8)
        mask[ds_arr < 5] = 0
        mask[ds_arr > 2999] = 0
        # num_labels, mask_connected_comp, stats, centroids = \
            # cv2.connectedComponentsWithStats(mask, connectivity=8,
                                             # ltype=cv2.CV_32S)
        # print(f"number labels: {num_labels}\n", f"stats:\n{stats}\n", f"centroids:\n{centroids}\n")
        # # ignore the first index of stats because it is the background
        # obj_idx = np.argmax(stats[1:], axis=0)[-1] + 1
        # x, y, w, h, area = stats[obj_idx]
        # cx, cy = centroids[obj_idx]
        # ds_cropped = ds_arr[y:y+h, x:x+w]
        # cv2.imwrite(file_woext + "_cropped.png",  mask_cropped)
        plt.imsave(file_woext + "_thres.png", mask, cmap="gray")
        # fig = plt.figure(figsize=(12, 10))
        # ax = plt.axes()
        # ax.hist(ds_arr.flatten(), bins=max_x-min_x)
        # ax.set_title(name)
        # ax.set_xlabel("bins")
        # ax.set_ylabel("counts")
        # plt.savefig(file_woext + ".png")

        ## Old connectivity method
        # mask_rm_extra = label(mask, background=0., connectivity=2)
    print("done")
