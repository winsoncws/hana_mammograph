import sys, os
from os.path import abspath, dirname, join, basename, isdir
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from train_ddp import Train
import argparse
import yaml
import shutil
from addict import Dict
import time

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):
    cfgs = Dict(yaml.load(open(abspath(args.cfgs), "r"), Loader=yaml.Loader))
    results_path = dirname(abspath(cfgs.train_params.model_path))
    model_id = cfgs.train_params.model_path.split(".", 1)[0][-2:]
    filename = basename(args.cfgs).split(".", 1)[0] + "_" + model_id + ".yaml"
    cp_path = join(results_path, filename)

    if not isdir(results_path):
        os.makedirs(results_path)
    shutil.copy(args.cfgs, cp_path)

    torch.manual_seed(42)
    ddp_setup(rank, world_size)
    train = Train(rank, cfgs)
    start = time.time()
    train.TrainDenseNet()
    end = time.time()
    train.SaveTrainingReport()
    start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(end-start)/3600.:.3f} hrs")
    destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Training configuration yaml file."))
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
