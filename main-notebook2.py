import os, sys, platform, subprocess
repo = "/home/isaiah/RSNABreastCancer2023/"
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
    timesheet = Dict()
    prep_init_start = time.time()
    paths = cfgs.paths
    pcfgs = cfgs.preprocess_params
    data_prep = MammoPreprocess(paths.data_src, paths.data_dest,
                                                  pcfgs.file_extension, pcfgs.resolution,
                                                  pcfgs.init_downsample_ratio,
                                                  pcfgs.normalization)
    prep_init_end = time.time()
    prep_init_time = prep_init_end - prep_init_start

    mcfgs = cfgs.metadata_params
    md_init_start = time.time()
    mdata_prep = MetadataPreprocess(paths.metadata_src, paths.metadata_dest,
                                    mcfgs)
    md_init_end = time.time()
    md_init_time = md_init_end - md_init_start

    md_proc_start = time.time()
    mdata_prep.GenerateMetadata()
    mdata_prep.Save()
    md_proc_end = time.time()
    md_proc_time = md_proc_end - md_proc_start

    prep_proc_start = time.time()
    data_prep.GenerateDataset()
    prep_proc_end = time.time()
    prep_proc_time = prep_proc_end - prep_proc_start

    timesheet.metadata.initialization = md_init_time
    timesheet.metadata.process = md_proc_time
    timesheet.preprocessing.initialization = prep_init_time
    timesheet.preprocessing.process = prep_proc_time

    with open(paths.timesheet_dest, "w") as f:
        json.dump(timesheet, f, indent=4)
    print(f"Timesheet created in {paths.timesheet_dest}.")
    return

def splitdata_loop(cfgs):
    paths = cfgs.paths
    SplitData(paths.metadata_dest, paths.data_ids_dest,
         cfgs.preprocess_params.test_set,
         cfgs.preprocess_params.num_samples,
         cfgs.preprocess_params.test_size)

def main(cfile):
    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    preprocess_loop(cfgs)
    splitdata_loop(cfgs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    submit = Submission(cfgs)
    submit.Run()
    submit.ExportSubmissionCSV()

if __name__ == "__main__":
    config_file = os.path.join(repo, "config/test_config.yaml")
    main(config_file)
