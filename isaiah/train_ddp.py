import os, sys
from os.path import isdir, abspath, dirname
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import MammoH5Data, GroupDistSampler
from models import DenseNet
from metrics import PFbeta
import json
from addict import Dict
from utils import printProgressBarRatio

# ViT transfer learning model? Inception net model?

class Train:

    def __init__(self, gpu_id, cfgs):
        assert torch.cuda.is_available()
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.train_params

        self.device = "cuda"
        self.gpu_id = gpu_id

        self.epochs = self.train_cfgs.epochs
        self.model_path = abspath(self.train_cfgs.model_path)
        self.train_report_path = abspath(self.train_cfgs.report_path)

        self.train_report = Dict({
            "epoch": [],
            "batch": [],
            "samples": [],
            "loss": [],
            "score": []
        })

        self.labels = self.data_cfgs.labels
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]
        self.best_weights = None

        self.data = MammoH5Data(self.device, self.data_cfgs)
        self.traintest_path = abspath(self.data_cfgs.traintest_path)
        with open(self.traintest_path, "r") as f:
            self.traintestsplit = Dict(json.load(f))
        self.train_sampler = GroupDistSampler(self.traintestsplit.train, shuffle=True)
        self.val_sampler = GroupDistSampler(self.traintestsplit.val, shuffle=True)
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.trainloader = DataLoader(self.data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.data, self.val_size, sampler=self.val_sampler)

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)):
            os.makedirs(dirname(filepath))

    def _SaveBestModel(self):
        if self.best_weights != None:
            self._CheckMakeDirs(self.model_path)
            torch.save(self.best_weights, self.model_path)
        print(f"Best model saved to {self.model_path}.")

    def SaveTrainingReport(self):
        self._CheckMakeDirs(self.train_report_path)
        with open(self.train_report_path, "w") as f:
            json.dump(self.train_report, f)
        print(f"Training report saved to {self.train_report_path}.")

    def GetTrainingReport(self):
        return self.train_report

    def TrainDenseNet(self):
        self.best_weights = None
        self.model = DenseNet(**self.model_cfgs)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), **self.optimizer_cfgs)
        met_name = "PFbeta"
        a = torch.from_numpy(np.array([[self.loss_weight_map[key] for key in self.labels]],
                                               dtype=np.float32)).to(self.device)
        loss = torch.tensor(0.).to(self.device)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        best_score = np.nan
        batches_per_epoch = int(np.ceil(len(self.trainloader) / float(self.batch_size)))
        cancer_p = np.nan
        for epoch in range(1, self.epochs + 1):
            self.train_report.epoch.append(epoch)

            # Training loop
            self.model.train()
            for batch, (img_id, inp, gt) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                self.train_report.batch.append(batch + 1)
                samples = list(img_id)
                self.train_report.samples.append(samples)
                out = self.model(inp)
                c1 = F.binary_cross_entropy_with_logits(out[:, :4], gt[:, :4], reduce=False)
                c2 = F.l1_loss(out[:, 4:], gt[:, 4:], reduce=False)
                loss = torch.sum(a[:, :4]*c1) + torch.sum(a[:, 4:]*c2)
                loss.backward()
                self.optimizer.step()
                self.train_report.loss.append(loss.detach().to("cpu").numpy())
                pre = f"epoch {epoch}/{self.epochs} | batch"
                suf = f"size: {self.batch_size}, loss: {loss.item():.5f}, pred: {cancer_p:.3f}, {met_name}: {best_score:.3f}"
                printProgressBarRatio(batch, batches_per_epoch, prefix=pre, suffix=suf, length=50)
            # Validation loop;  every epoch
            self.model.eval()
            img_id, vi, vt = next(iter(self.validloader))
            preds = self.model(vi)
            cancer_p = torch.mean(preds[:, 3]).detach().to("cpu").item()
            sco = PFbeta(preds[:, 3], vt[:, 3], beta=0.5)
            self.train_report.score.append(sco.item())

            if sco > best_score:
                self.best_weights = self.model.module.state_dict()
                best_score = sco.item()
                self._SaveBestModel()
        return

if __name__ == "__main__":
    pass