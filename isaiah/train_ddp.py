import os, sys
from os.path import isdir, abspath, dirname
from collections import defaultdict
import numpy as np
import json
import csv
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torchmetrics.functional.classification import binary_f1_score
import timm
from torchlars import LARS
from dataset import MammoH5Data, GroupDistSampler, DoubleBalancedGroupDistSampler
from models import DenseNet
from utils import printProgressBarRatio

# ViT transfer learning model? Inception net model?

class NullScheduler:

    def __init__(self, lr):
        self.lr = lr

    def get_last_lr(self):
        return [self.lr]

class Train:

    def __init__(self, cfgs):
        assert torch.cuda.is_available()
        self.no_gpus = torch.cuda.device_count()

        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.scheduler_cfgs = cfgs.scheduler_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.run_params

        if self.paths.model_load_src != None:
            self.model_state = torch.load(self.paths.model_load_src)["model"]
            self.optimizer_state = torch.load(self.paths.model_load_src)["optimizer"]
        else:
            self.model_state = None
            self.optimizer_state = None

        self.ckpts_path = abspath(self.paths.model_ckpts_dest)
        self.model_best_path = abspath(self.paths.model_best_dest)
        self.train_report_path = abspath(self.paths.train_report_path)
        self.eval_report_path = abspath(self.paths.eval_report_path)
        self.data_path = abspath(self.paths.data_dest)
        self.metadata_path = abspath(self.paths.metadata_dest)
        self.data_ids_path = abspath(self.paths.data_ids_dest)

        self.device = "cuda"

        self.train = self.train_cfgs.train
        self.epochs = self.train_cfgs.epochs
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.track_freq = self.train_cfgs.tracking_frequency
        self.block_size = self.track_freq * self.batch_size
        self.classes = self.train_cfgs.classes
        self.selected_model = self.train_cfgs.model
        self.sel_optim = self.train_cfgs.optimizer
        self.sel_scheduler = self.train_cfgs.scheduler

        self.model_dict = defaultdict(timm.create_model, {
            "custom_densenet": DenseNet,
        })

        self.optim_dict = defaultdict(Adam, {
            "adam": Adam,
            "sgd": SGD,
        })
        self.scheduler_dict = defaultdict(None, {
            "exponential": ExponentialLR,
            "cyclic": CyclicLR,
        })

        self.met_name = "PFbeta"
        self.labels = self.data_cfgs.labels
        self.ratio = self.data_cfgs.sample_ratio
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]

        with open(self.data_ids_path, "r") as f:
            self.data_ids = Dict(json.load(f))

        self.e = 1e-6

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)): os.makedirs(dirname(filepath))

    def _RemovePath(self, filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    def _SaveBestModel(self, state, value):
        self._CheckMakeDirs(self.model_best_path)
        torch.save(state, self.model_best_path)
        print(f"New best model with {self.met_name} = {value} saved to {self.model_best_path}.")

    def _SaveCkptsModel(self, state, value):
        self._CheckMakeDirs(self.ckpts_path)
        torch.save(state, self.ckpts_path)
        print(f"Checkpoint with {self.met_name} = {value} saved to {self.ckpts_path}.")

    def _TrainModelDDP(self, gpu_id):
        self._SetupDDP(gpu_id, self.no_gpus)
        self.data = MammoH5Data(gpu_id, self.data_path, self.metadata_path,
                                self.data_cfgs)
        self.train_sampler = DoubleBalancedGroupDistSampler(self.data_ids.train[self.classes[0]],
                                                    self.data_ids.train[self.classes[1]],
                                                    shuffle=True)
        self.val_sampler = GroupDistSampler(self.data_ids.val[self.classes[0]] + self.data_ids.val[self.classes[1]],
                                                  shuffle=True)
        self.trainloader = DataLoader(self.data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.data, self.val_size, sampler=self.val_sampler)
        self.total_val_size = self.val_sampler.num_samples
        model = self.model_dict[self.selected_model](**self.model_cfgs).to(gpu_id)
        if self.model_state != None:
            model.load_state_dict(self.model_state)
            print(f"gpu_id: {gpu_id} - model loaded.")
        self.model = DDP(model, device_ids=[gpu_id])
        self.optimizer = self.optim_dict[self.sel_optim](self.model.parameters(),
                                                         **self.optimizer_cfgs)
        if self.optimizer_state != None:
            self.optimizer.load_state_dict(self.optimizer_state)
        if self.train_cfgs.apply_lars:
            self.optimizer = LARS(self.optimizer)
        if self.sel_scheduler == None:
            self.scheduler = NullScheduler()
        else:
            self.scheduler = self.scheduler_dict[self.sel_scheduler](self.optimizer,
                                                                      **self.scheduler_cfgs)
        # a = torch.from_numpy(np.array([[self.loss_weight_map[key] for key in self.labels]],
                                               # dtype=np.float32)).to(self.device)
        best_score = 0.
        sco = 0.
        block = 1
        preds = []
        truths = []
        if gpu_id == 0:
            self._RemovePath(self.train_report_path)
            self._RemovePath(self.eval_report_path)
            train_log = open(self.train_report_path, "a")
            eval_log = open(self.eval_report_path, "a")
            train_writer = csv.writer(train_log)
            eval_writer = csv.writer(eval_log)
            train_writer.writerow(["epoch", "block", "learning_rate",
                             "predictions", "truths", "loss", "f1_score"])
            eval_writer.writerow(["epoch", "samples", "predictions", "truths",
                                  "f1_score"])
        for epoch in range(1, self.epochs + 1):
            self.train_sampler.set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)
            if self.train:

                # Training loop
                self.model.train()
                for batch, (img_id, inp, gt) in enumerate(self.trainloader):

                    last_lr = self.scheduler.get_last_lr()[0]
                    self.optimizer.zero_grad()
                    out = self.model(inp)
                    # c1 = F.binary_cross_entropy_with_logits(out[:, :4], gt[:, :4], reduction="none")
                    # c2 = F.l1_loss(out[:, 4:], gt[:, 4:], reduction="none")
                    # loss = torch.sum(a[:, :4]*c1) + torch.sum(a[:, 4:]*c2)
                    p = torch.sigmoid(out)
                    preds.append(p.detach())
                    truths.append(gt.detach())
                    loss = F.binary_cross_entropy(p, gt)
                    loss.backward()
                    self.optimizer.step()
                    if (gpu_id == 0) and ((batch + 1) % self.track_freq == 0):
                        preds = torch.cat(preds).squeeze()
                        truths = torch.cat(truths).squeeze()
                        bf1 = binary_f1_score(preds, truths)
                        train_writer.writerow([epoch, block, last_lr, preds.cpu().tolist(),
                                         truths.cpu().tolist(), loss.detach().to("cpu").item(),
                                         bf1.item()])
                        print((f"epoch {epoch}/{self.epochs} | block: {block}, block_size: {self.block_size} | "
                               f"loss: {loss.item():.5f}, block_f1: {bf1:.3f}, "
                               f"{self.met_name}: {sco:.3f}, best: {best_score:.3f}"))
                        block += 1
                        preds = []
                        truths = []

            # Validation loop;  every epoch
            self.scheduler.step()
            self.model.eval()
            samples = []
            probs = []
            labels = []
            for vbatch, (vimg_id, vi, vt) in enumerate(self.validloader):
                samples.append(vimg_id.detach())
                probs.append(torch.sigmoid(self.model(vi)).detach())
                labels.append(vt.detach())
            samples = torch.cat(samples)
            probs = torch.cat(probs)
            labels = torch.cat(labels)
            sam_gather = [torch.zeros((self.total_val_size, 1),
                                       dtype=torch.int64).to(probs.device) for _ in range(self.no_gpus)]
            probs_gather = [torch.zeros((self.total_val_size, 1),
                                        dtype=torch.float32).to(probs.device) for _ in range(self.no_gpus)]
            labels_gather = [torch.zeros((self.total_val_size, 1),
                                         dtype=torch.float32).to(probs.device) for _ in range(self.no_gpus)]
            dist.all_gather(sam_gather, samples)
            dist.all_gather(probs_gather, probs)
            dist.all_gather(labels_gather, labels)
            all_samples = torch.cat(sam_gather).squeeze()
            all_probs = torch.cat(probs_gather).squeeze()
            all_labels = torch.cat(labels_gather).squeeze()
            if gpu_id == 0:
                sco = binary_f1_score(all_probs, all_labels)
                eval_writer.writerow([epoch, all_samples.cpu().tolist(), all_probs.cpu().tolist(),
                                      all_labels.cpu().tolist(), sco.item()])
                state = {
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                self._SaveCkptsModel(state, sco)
                if sco > best_score:
                    best_score = sco
                    self._SaveBestModel(state, best_score)
        if gpu_id == 0:
            train_log.close()
            eval_log.close()
        self._ShutdownDDP()
        return

    def _ConfusionMatrix(self, p, gt):
        tp = np.sum((p == 1) & (gt == 1)).astype(int)
        fp = np.sum(p).astype(int) - tp
        tn = np.sum((p == 0) & (gt == 0)).astype(int)
        fn = np.sum(gt).astype(int) - tp
        return tp, fp, tn ,fn

    def _SetupDDP(self, rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return

    def _ShutdownDDP(self):
        destroy_process_group()
        return

    def RunDDP(self):
        mp.spawn(self._TrainModelDDP, nprocs=self.no_gpus)

if __name__ == "__main__":
    pass
