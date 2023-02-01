import os, sys, platform, subprocess
repo = "/home/isaiah/RSNABreastCancer2023/"
sys.path.insert(0, repo)
import traceback, logging
logging.basicConfig(level=logging.DEBUG)
handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(handler)
from os.path import join, dirname, basename, abspath
import time
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
from glob import glob
# import addict
from addict import Dict
# import monai
import monai
# import torchio
import torchio as tio
# import custom classes

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
    return

def randresult(md_path):
    test_md = pd.read_csv(md_path)
    pats = test_md["patient_id"].tolist()
    lats = test_md["laterality"].tolist()
    pat_ids = ["_".join(item) for item in zip(map(str, pats), lats)]
    rand_cancer = np.random.rand(len(pat_ids))
    raw_results = pd.DataFrame({"prediction_id": pat_ids, "cancer": rand_cancer})
    results = raw_results.groupby("prediction_id")["cancer"].mean()
    return results

def main(cfile, random_submission_path, test_metadata_path):
    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    continue_test = True
    try:
        preprocess_loop(cfgs)
        splitdata_loop(cfgs)
    except Exception as err:
        logger.exception(err)
        continue_test = False
        res = randresult(test_metadata_path)
        res.to_csv(random_submission_path, index=True)
    if continue_test:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            submit = Submission(cfgs)
            start = time.time()
            submit.Run()
            submit.ExportSubmissionCSV()
            end = time.time()
            print("\n")
            start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
            end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
            time_taken = end - start
            if time_taken > 3600.:
                divisor = 3600.
                suffix = "hr"
            else:
                divisor = 60.
                suffix = "min"
            print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(time_taken)/divisor:.3f} {suffix}")
        except Exception as err:
            logger.exception(err)
            res = randresult(test_md_path)
            res.to_csv(save_path, index=True)
    return

if __name__ == "__main__":
    config_file = os.path.join(repo, "config/test_config.yaml")
    save_path = "/home/isaiah/submission.csv"
    test_md_path = "/home/dataset/kaggle/input/test.csv"
    main(config_file, save_path, test_md_path)
