import os, sys
from os.path import isdir, abspath, dirname
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from dataset import MammoH5Data, GroupSampler
from models import DenseNet
from metrics import PFbeta
import json
from addict import Dict
from utils import printProgressBarRatio

# ViT transfer learning model? Inception net model?

class Submission:

    def __init__(self, cfgs):
        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.data_cfgs = cfgs.dataset_params
        self.test_cfgs = cfgs.run_params

        self.model_weights_path = self.paths.model_load_src
        self.submission_path = abspath(self.paths.submission_path)
        self.other_result_path = abspath(self.paths.other_result_dest)
        self.data_path = abspath(self.paths.data_dest)
        self.metadata_path = abspath(self.paths.metadata_dest)
        self.data_ids_path = abspath(self.paths.data_ids_dest)

        if not self.test_cfgs.no_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model_weights = torch.load(self.model_weights_path)
        elif not self.test_cfgs.no_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model_weights = torch.load(self.model_weights_path, map_location=self.device)
        else:
            self.device = torch.device('cpu')
            self.model_weights = torch.load(self.model_weights_path, map_location=self.device)

        self.for_submission = self.test_cfgs.submission
        self.labels = self.data_cfgs.labels
        self.default_value = self.test_cfgs.default_value
        self.lmap = defaultdict(lambda: self.default_value, self.test_cfgs.laterality_map)
        self.results = None
        self.other_res = None

        self.data = MammoH5Data(self.device, self.data_path, self.metadata_path,
                                self.data_cfgs)
        with open(self.data_ids_path, "r") as f:
            self.test_ids = Dict(json.load(f))
        self.test_sampler = GroupSampler(self.test_ids.test, shuffle=True)
        self.batch_size = self.test_cfgs.batch_size
        self.testloader = DataLoader(self.data, self.batch_size, sampler=self.test_sampler)

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)):
            os.makedirs(dirname(filepath))

    def _ExportSubmissionCSV(self):
        self._CheckMakeDirs(self.submission_path)
        self.results.to_csv(self.submission_path, index=True, index_label="prediction_id")

    def _ExportOtherCSV(self):
        self._CheckMakeDirs(self.other_res)
        self.other_res.to_csv(self.other_result_path, index=False)

    def _RunSub(self):
        self.model = DenseNet(**self.model_cfgs)
        if self.model_weights != None:
            self.model.load_state_dict(self.model_weights)
        self.model.to(self.device)
        self.model.eval()
        pats = []
        lats = []
        preds = []
        for vbatch, (vimg_id, vi, vt) in enumerate(self.testloader):
            pats.extend(vt.int()[:, 0].detach().tolist())
            lats.extend(vt.int()[:, 1].detach().tolist())
            preds.extend(torch.sigmoid(self.model(vi))[:, 3].detach().tolist())
            printProgressBarRatio(vbatch + 1, len(self.testloader), prefix="Samples")
        rlats = [self.lmap[val] for val in lats]
        pred_ids = ["_".join(item) for item in zip(map(str, pats), rlats)]
        df = pd.DataFrame(preds, index=pred_ids, columns=["cancer"])
        self.results = df.groupby(df.index)["cancer"].mean()
        return

    def _RunOther(self):
        self.model = DenseNet(**self.model_cfgs)
        self.model.load_state_dict(self.model_weights)
        self.model.to(self.device)
        self.model.eval()
        img_ids = []
        truths = []
        preds = []
        for vbatch, (vimg_id, vi, vt) in enumerate(self.testloader):
            img_ids.extend(list(vimg_id))
            truths.extend(vt.detach().tolist())
            preds.extend(torch.sigmoid(self.model(vi)).detach().tolist())
            printProgressBarRatio(vbatch + 1, len(self.testloader), prefix="Samples")
        pats = list(np.asarray(truths).astype(np.int)[0])
        btruths = np.asarray(truths).astype(np.int)[1, :5]
        dtruths = np.asarray(truths)[:, 5:]
        bpreds = np.asarray(preds).astype(np.int)[1, :5]
        dpreds = np.asarray(preds)[:, 5:]
        acc = np.sum(bpreds == btruths, axis=0)/bpreds.shape[0]
        lats = list(bpreds[:, 0])
        rlats = [self.lmap[val] for val in lats]
        pred_ids = ["_".join(item) for item in zip(map(str, pats), rlats)]
        df = pd.DataFrame(np.asarray(preds)[:, 3], index=pred_ids, columns=["cancer"])
        self.results = df.groupby(df.index)["cancer"].mean()
        self.other_res = pd.DataFrame(acc, columns=self.labels[1:])

    def Run(self):
        if self.for_submission:
            self._RunSub()
            self._ExportSubmissionCSV()
        else:
            self._RunOther()
            self._ExportOtherCSV()
        return

if __name__ == "__main__":
    pass
