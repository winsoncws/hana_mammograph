import sys, os
from os.path import abspath, dirname, join, basename, isdir
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from train import Train
import argparse
import yaml
import shutil
from addict import Dict
import time

def main(args):
    torch.manual_seed(42)
    cfgs = Dict(yaml.load(open(abspath(args.cfgs), "r"), Loader=yaml.Loader))
    train_cfile = args.cfgs
    results_path = dirname(abspath(cfgs.train_params.model_path))
    model_id = cfgs.train_params.model_path.split(".", 1)[0][-2:]
    filename = basename(args.cfgs).split(".", 1)[0] + "_" + model_id + ".yaml"
    cp_path = join(results_path, filename)

    if not isdir(results_path):
        os.makedirs(results_path)
    shutil.copy(args.cfgs, cp_path)

    train = Train(cfgs)
    start = time.time()
    train.TrainDenseNet()
    end = time.time()
    train.SaveTrainingReport()
    start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(end-start)/3600.:.3f} hrs")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Training configuration yaml file."))
    args = parser.parse_args()
    main(args)
