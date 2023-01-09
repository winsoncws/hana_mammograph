import sys, os
from os.path import join, abspath, dirname, basename
import glob
import numpy as np
import cv2
from scipy.ndimage import label
from skimage.measure import regionprops
import torch
import torchvision
import pydicom
import scipy.stats as st
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MammoPreprocess:

    def __init__(self, source_directory, savepath, file_extension=".png",
                 resolution=None):
        self.src = abspath(source_directory)
        self.src_len= len(self.src) + 1
        self.savepath = abspath(savepath)
        self.res = resolution
        self.file_ext = file_extension
        self.datafiles = glob.glob(join(self.src, "**/*.dcm"),
                                                recursive=True)

    def Invert(self, im):
        return im.max() - im

    def JoinCompressPad(self, im):
        h, w = im.shape
        if self.res != None:
            if np.max(self.res) > np.max(im.shape):
                print("WARNING: Input image size is smaller than output image size.")
            else:
                end_shape = (np.asarray(self.res) * (im.shape/np.max(im.shape))).astype(np.int16)[::-1]
                im = cv2.resize(im, dsize=end_shape, interpolation=cv2.INTER_CUBIC)
                w, h = end_shape
        diff = np.abs(h - w)
        if h > w:
            top, bot, left, right = [0, 0, diff // 2, diff - (diff // 2)]
        else:
            top, bot, left, right = [diff // 2, diff - (diff // 2), 0, 0]
        im_pad = cv2.copyMakeBorder(im, top, bot, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=0.)
        return im_pad

    def NormThres(self):
        return

    def MinMaxThres(self, offset=5):
        for file in self.datafiles:
            parentdir = dirname(file[self.src_len:])
            name = file[self.src_len:].split(".", 1)[0]
            os.makedirs(join(self.savepath, parentdir), exist_ok=True)
            ds = pydicom.dcmread(file)
            im = ds.pixel_array
            p = np.sum(im[im == np.max(im)])/np.prod(im.shape)
            if p > 0.75:
                im = self.Invert(im)
            min_x = np.min(im)
            mask = np.ones_like(im, dtype=np.int8)
            mask[im < min_x + offset] = 0
            mask, no_of_labels = label(mask)
            # num_labels, mask_connected_comp, stats, centroids = \
                # cv2.connectedComponentsWithStats(mask, connectivity=8,
                                                 # ltype=cv2.CV_32S)
            # # ignore the first index of stats because it is the background
            _, stats = np.unique(mask, return_counts=True)
            obj_idx = np.argmax(stats[1:]) + 1
            x1,y1,x2,y2 = regionprops(1*(mask == obj_idx))[0].bbox
            h = x2-x1
            w = y2-y1
            im_crop = im[x1:x2, y1:y2]
            im_fin = self.JoinCompressPad(im_crop)
            plt.imshow(im_fin, cmap="bone")
            plt.show()
            sys.exit()
            sfile = os.path.join(self.savepath, name + self.file_ext)
            plt.imsave(sfile, im_fin, cmap="gray")

if __name__ == "__main__":
    datadir = sys.argv[1]
    savedir = sys.argv[2]
    preprocessing = MammoPreprocess(datadir, savedir, resolution=(512, 512))
    preprocessing.MinMaxThres()
