import os, sys
from os.path import isdir, abspath, dirname
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchio.data import dataset
from build import MammoH5Data
from build import DenseNet
from build import PFbeta
import yaml
from addict import Dict

# ViT transfer learning model? Inception net model?

class Train:

    def __init__(self, cfgs):
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.train_params
        self.labels = self.data_cfgs.label
        self.epochs = self.train_cfgs.epochs
        self.savepath = abspath(self.train_cfgs.savepath)
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]
        self.best_weights = None

        self.data = MammoH5Data(self.data_cfgs)
        self.trainloader = DataLoader(self.data, 10, shuffle=True)
        self.validloader = DataLoader(self.data)

    def _save(self):
        if self.best_weights != None:
            if not isdir(dirname(self.savepath)):
                os.makedirs(dirname(self.savepath))
            torch.save(self.best_weights, self.savepath)

    def TrainDenseNet(self):
        self.model = DenseNet(self.model_cfgs)
        self.optimizer = Adam(self.model.parameters, cfgs.optim_params)
        met_name = "PFbeta"
        a = torch.from_numpy(np.array([self.loss_weight_map[key] for key in self.labels],
                                               dtype=np.float32))
        c1 = nn.BCELoss(a[:4])
        c2 = nn.L1Loss(a[4:])
        val_count = 0
        loss = torch.tensor(0.)
        best_score = 0.
        for epoch in range(1, self.epochs + 1):
            img_id, inp, gt = next(iter(self.trainloader))
            out = self.model(inp)
            loss = c1(out[:4], gt[:4]) + c2(out[4:], gt[4:])
            loss.backward()
            print(f"epoch {epoch}, loss: {loss.item()}")
            if epoch % 10 == 0:
                val_count += 1
                self.model.eval()
                img_id, vi, vt = next(iter(self.validloader))
                preds = self.model(vi)
                sco = PFbeta(preds[3].numpy(), vt[3])
                print(f"Validation {val_count}, {met_name}: {sco}")
                if sco > best_score:
                    self.best_weights = self.model.state_dict()
                    best_score = sco
                    self._save()
                    print("Better model found and saved")
        print(f"total epochs: {self.epochs}, final_loss: {loss.item()}, best_score: {best_score}.")
        return

if __name__ == "__main__":

    torch.manual_seed(42)
    config_file: str = "/Users/isaiah/GitHub/hana_mammograph/isaiah/train_config.yaml"
    cfgs = Dict(yaml.load(open(config_file, "r"), Loader=yaml.Loader))
    train = Train(cfgs)
