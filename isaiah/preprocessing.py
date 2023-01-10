import sys, os
from os.path import join, abspath, dirname
import glob
import numpy as np
import cv2
from scipy.ndimage import label
from skimage.measure import regionprops
import pydicom
import matplotlib.pyplot as plt
import ast
import argparse
from addict import Dict
import h5py
import yaml

class MammoPreprocess:

    def __init__(self, source_directory, savepath, file_extension=".png",
                 resolution=None, normalize=False):
        self.src = abspath(source_directory)
        self.src_len= len(self.src) + 1
        self.savepath = abspath(savepath)
        self.res = resolution
        self.file_ext = file_extension
        self.datafiles = glob.glob(join(self.src, "**/*.dcm"),
                                                recursive=True)
        self.normit = normalize

    def ProportionInvert(self, im, alpha=0.75):
        p = np.sum(im[im == np.max(im)])/np.prod(im.shape)
        if p > alpha:
            im = im.max() - im
        return im

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

    def ThresCrop(self, im, offset=5):
        min_x = np.min(im)
        mask = np.ones_like(im, dtype=np.int8)
        mask[im < min_x + offset] = 0
        mask, _ = label(mask)
        # # ignore the first index of stats because it is the background
        _, stats = np.unique(mask, return_counts=True)
        obj_idx = np.argmax(stats[1:]) + 1
        x1,y1,x2,y2 = regionprops(1*(mask == obj_idx))[0].bbox
        h = x2-x1
        w = y2-y1
        return im[x1:x2, y1:y2]

    def ProcessDicom(self, file):
        ds = pydicom.dcmread(file)
        im = ds.pixel_array
        im = self.ProportionInvert(im)
        im = self.ThresCrop(im)
        im = self.JoinCompressPad(im)
        if self.normit:
            im= cv2.normalize(im, None, alpha=0, beta=1,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_32F)
        return im

    def save(self, filepath, im):
        if self.file_ext == ".npy":
            np.save(filepath, im)
        elif self.file_ext == ".npz":
            np.savez_compressed(filepath, im)
        else:
            plt.imsave(filepath, im, cmap="gray")

    def GenerateFolderTreeDataset(self):
        no_of_files = 0
        for file in self.datafiles:
            parentdir = dirname(file[self.src_len:])
            name = file[self.src_len:].split(".", 1)[0]
            os.makedirs(join(self.savepath, parentdir), exist_ok=True)
            im = self.ProcessDicom(file)
            sfile = os.path.join(self.savepath, name + self.file_ext)
            self.save(sfile, im)
            no_of_files += 1
        print(f"{self.savepath} created. Total of {str(no_of_files)} images.")

    def GenerateH5Dataset(self):
        hdf = h5py.File(self.savepath, "a")
        no_of_files = 0
        for file in self.datafiles:
            parentdir = dirname(file[self.src_len:])
            name = file[self.src_len:].split(".", 1)[0]
            grp = hdf.require_group(parentdir)
            im = self.ProcessDicom(file)
            grp.create_dataset(name, data=im, compression="gzip",
                               compression_opts=9)
            no_of_files += 1
        hdf.close()
        print(f"{self.savepath} created. Total of {str(no_of_files)} images.")


class ProcessPath(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        if self.dest in ["src", "dest"]:
            if (values == None) or (values == ".") or (values == "./"):
                values = os.getcwd()
            if values[-1] != "/":
                values = f"{values}/"
        elif self.dest == "fext":
            if values[0] != ".":
                values = f".{values}"
        setattr(namespace, self.dest, values)

class ConvertTuple(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        values = ast.literal_eval(values)
        if not isinstance(values, tuple):
            raise Exception("Resolution is not a tuple.")
        setattr(namespace, self.dest, values)

def main(args):
    if args.cfgs != None:
        cfgs = Dict(yaml.load(open(args.cfgs, "r"), Loader=yaml.Loader))
        preprocessing = MammoPreprocess(cfgs.src, cfgs.dest,
                                                          cfgs.fext, cfgs.res,
                                                          cfgs.norm)
        makeh5 = cfgs.makeh5
    else:
        preprocessing = MammoPreprocess(args.src, args.dest,
                                                          args.fext, args.res,
                                                          args.norm)
        makeh5 = args.makeh5
    if makeh5:
        preprocessing.GenerateH5Dataset()
    else:
        preprocessing.GenerateFolderTreeDataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Configuration yaml file."
                              "All other arguments ignored if this is passed."))
    parser.add_argument("-e", "--extension", type=str, metavar="EXT",
                        dest="fext", nargs="?", default=".png", action=ProcessPath,
                        help=("File extension for output images."))
    parser.add_argument("-r", "--resolution", metavar="INT,INT",
                        dest="res", nargs="?", default=None, action=ConvertTuple,
                        help=("Desired resolution of the output images."))
    parser.add_argument("-n", "--normalize", action="store_true",
                        dest="norm",
                        help=("Normalize the image to within 0, 1. "))
    parser.add_argument("-h5", "--h5-dataset", action="store_true",
                        dest="makeh5",
                        help=("Create a hdf5 dataset."))
    parser.add_argument("src", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Directory to search for DICOM files recursively."))
    parser.add_argument("dest", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Directory to save output images."))
    args = parser.parse_args()

    main(args)
