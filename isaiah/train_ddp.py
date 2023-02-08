import os, sys
from os.path import isdir, abspath, dirname
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from dataset import MammoH5Data, DoubleBalancedGroupDistSampler
from models import DenseNet
from torchmetrics.functional.classification import binary_accuracy
from metrics import PFbeta
import json
import csv
from addict import Dict
from utils import printProgressBarRatio

# ViT transfer learning model? Inception net model?


class Train:

    def __init__(self, gpu_id, cfgs):
        assert torch.cuda.is_available()
        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.scheduler_cfgs = cfgs.scheduler_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.run_params

        if self.paths.model_load_src != None:
            self.model_weights = torch.load(self.paths.model_load_src)
            print(f"gpu_id: {gpu_id} - model weights loaded.")
        else:
            self.model_weights = None

        self.ckpts_path = abspath(self.paths.model_ckpts_dest)
        self.model_best_path = abspath(self.paths.model_best_dest)
        self.model_final_path = abspath(self.paths.model_final_dest)
        self.train_report_path = abspath(self.paths.train_report_path)
        self.data_path = abspath(self.paths.data_dest)
        self.metadata_path = abspath(self.paths.metadata_dest)
        self.data_ids_path = abspath(self.paths.data_ids_dest)

        self.device = "cuda"
        self.gpu_id = gpu_id

        self.train = self.train_cfgs.train
        self.epochs = self.train_cfgs.epochs

        self.met_name = "PFbeta"
        self.labels = self.data_cfgs.labels
        self.ratio = self.data_cfgs.sample_ratio
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]

        self.data = MammoH5Data(self.device, self.data_path, self.metadata_path,
                                self.data_cfgs)
        with open(self.data_ids_path, "r") as f:
            self.data_ids = Dict(json.load(f))
        self.train_sampler = DoubleBalancedGroupDistSampler(self.data_ids.train.healthy,
                                                    self.data_ids.train.cancer,
                                                    shuffle=True)
        self.val_sampler = DoubleBalancedGroupDistSampler(self.data_ids.val.healthy,
                                                  self.data_ids.val.cancer,
                                                  shuffle=True)
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.trainloader = DataLoader(self.data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.data, self.val_size, sampler=self.val_sampler)

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)):
            os.makedirs(dirname(filepath))

    def _SaveBestModel(self, value):
        if self.model_weights != None:
            self._CheckMakeDirs(self.model_best_path)
            torch.save(self.model_weights, self.model_best_path)
        print(f"New best model with {self.met_name} = {value} saved to {self.model_best_path}.")

    def _SaveFinalModel(self):
        self._CheckMakeDirs(self.model_final_path)
        torch.save(self.model_weights, self.model_final_path)
        print(f"Final model saved to {self.model_final_path}.")

    def SaveTrainingReport(self):
        self._CheckMakeDirs(self.train_report_path)
        with open(self.train_report_path, "w") as f:
            json.dump(self.train_report, f)
        print(f"Training report saved to {self.train_report_path}.")

    def GetTrainingReport(self):
        return self.train_report

    def TrainDenseNet(self):
        self.model = DenseNet(**self.model_cfgs)
        if self.model_weights != None:
            self.model.load_state_dict(self.model_weights)
        self.model_weights = None
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), **self.optimizer_cfgs)
        self.scheduler = CyclicLR(self.optimizer, **self.scheduler_cfgs)
        a = torch.from_numpy(np.array([[self.loss_weight_map[key] for key in self.labels]],
                                               dtype=np.float32)).to(self.device)
        loss = torch.tensor(0.).to(self.device)
        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)
        bacc = 0.
        eacc = 0.
        best_score = 0.
        sco = 0.
        if self.gpu_id == 0:
            if os.path.exists(self.train_report_path):
                os.remove(self.train_report_path)
            log = open(self.train_report_path, "a")
            writer = csv.writer(log)
            writer.writerow(["epoch", "batch", "learning_rate", "samples", "loss", "batch_accuracy",
                             "epoch_accuracy", "f1_score", "best_score"])
        for epoch in range(1, self.epochs + 1):
            if self.train:
                # if self.gpu_id == 0:
                    # self.train_report.epoch.append(epoch)

                # Training loop
                self.model.train()
                for batch, (img_id, inp, gt) in enumerate(self.trainloader):
                    last_lr = self.scheduler.get_last_lr()[0]
                    self.optimizer.zero_grad()
                    samples = list(img_id)
                    out = self.model(inp)
                    # c1 = F.binary_cross_entropy_with_logits(out[:, :4], gt[:, :4], reduction="none")
                    # c2 = F.l1_loss(out[:, 4:], gt[:, 4:], reduction="none")
                    # loss = torch.sum(a[:, :4]*c1) + torch.sum(a[:, 4:]*c2)
                    loss = F.binary_cross_entropy_with_logits(out, gt)
                    bacc = binary_accuracy(out, gt).detach().to("cpu").item()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # self.train_report.batch.append(batch + 1)
                    # self.train_report.samples.append(samples)
                    # self.train_report.loss.append(loss.detach().to("cpu").tolist())
                    if self.gpu_id == 0:
                        writer.writerow([epoch, batch + 1, last_lr, samples, loss.detach().to("cpu").item(),
                                         bacc, eacc, sco, best_score])
                        print((f"epoch {epoch}/{self.epochs} | batch_size: {self.batch_size} | "
                               f"loss: {loss.item():.5f}, batch_acc: {bacc:.3f}, epoch_acc: {eacc:.3f}, "
                               f"{self.met_name}: {sco:.3f}, best: {best_score:.3f}"))

            # Validation loop;  every epoch
            self.model.eval()
            preds = []
            labels = []
            for vbatch, (vimg_id, vi, vt) in enumerate(self.validloader):
                preds.extend(torch.sigmoid(self.model(vi)).detach().to("cpu").numpy().flatten().tolist())
                labels.extend(vt.detach().to("cpu").numpy().flatten().tolist())
            eacc = np.sum(np.array(preds).round() == np.array(labels))/float(len(preds))
            sco = float(PFbeta(labels, preds, beta=0.5))
            # self.train_report.score.append(sco)
            if self.gpu_id == 0:
                self._CheckMakeDirs(self.ckpts_path)
                torch.save(self.model.module.state_dict(), self.ckpts_path)
            if sco > best_score:
                best_score = sco
                self.model_weights = self.model.module.state_dict()
                if self.gpu_id == 0:
                    self._SaveBestModel(best_score)
        self.model_weights = self.model.module.state_dict()
        if self.gpu_id == 0:
            self._SaveFinalModel()
            log.close()
        return

if __name__ == "__main__":
    pass
