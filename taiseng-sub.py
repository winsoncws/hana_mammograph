import os, sys, platform, subprocess
repo = "/home/isaiah/hana_mammograph/isaiah/"
sys.path.insert(0, repo)
from os.path import join, dirname, basename, abspath
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import cv2
import torch
import tensorflow
import matplotlib.pyplot as plt
import yaml
import h5py
import json
import time
from glob import glob
# import addict
from addict import Dict
# import monai
import monai
# import torchio
import torchio as tio

from submission import Submission
from preprocessing import MammoPreprocess, MetadataPreprocess
from splitdata import SplitData

def preprocess_loop(cfgs):
    paths = cfgs.paths
    pcfgs = cfgs.preprocess_params
    data_prep = MammoPreprocess(paths.data_src, paths.data_dest,
                                                  pcfgs.resolution,
                                                  pcfgs.init_downsample_ratio,
                                                  pcfgs.normalization)

    mcfgs = cfgs.metadata_params
    mdata_prep = MetadataPreprocess(paths.metadata_src, paths.metadata_dest,
                                    mcfgs)

    mdata_prep.GenerateMetadata()
    mdata_prep.Save()

    data_prep.GenerateDataset()
    return

def splitdata_loop(cfgs):
    paths = cfgs.paths
    SplitData(paths.metadata_dest, paths.data_ids_dest,
         cfgs.preprocess_params.test_set,
         cfgs.preprocess_params.num_samples,
         cfgs.preprocess_params.test_size)

def main(cfile):
    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    # preprocess_loop(cfgs)
    # splitdata_loop(cfgs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    submit = Submission(cfgs)
    submit.Run()

if __name__ == "__main__":
    config_file = os.path.join(repo, "config/test_config.yaml")
    main(config_file)
