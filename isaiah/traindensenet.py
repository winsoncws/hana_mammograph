import sys, os
from os.path import abspath, dirname, join, basename, isdir
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from train import Train
import argparse
import yaml
import shutil
from addict import Dict

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
    train.TrainDenseNet()
    train.SaveTrainingReport()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Training configuration yaml file."))
    args = parser.parse_args()
    main(args)
