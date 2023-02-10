import sys, os
from os.path import abspath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import json
import h5py
import random
import yaml
from addict import Dict

from dataset import MammoH5Data, BalancedGroupSampler


def PFbeta(labels, predictions, beta, eps=1e-5):
    # eps is a small error term added for numerical stability
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = (ctp + eps) / (ctp + cfp + eps)
    c_recall = (ctp + eps) / (y_true_count + eps)
    print(f"Precision: {c_precision}")
    print(f"Recall: {c_recall}")
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.

def PlotLoss(loss, ma):
    minloss = np.min(loss.to_numpy())
    maxloss = np.max(loss.to_numpy())
    print(f"minimum: {minloss}, maximum: {maxloss}")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(loss.index, loss["loss"], s=0.3, c="red")
    ax.plot(loss.index, ma, "-b")
    plt.show()

def MovingAvg(v, size):
    avg = np.empty_like(v)
    avg[:size] = np.mean(v[:2*size])
    avg[len(v)-2*size+1:len(v)] = np.mean(v[len(v)-2*size+1:len(v)])
    for i in range(size, len(v) - size):
        avg[i] = np.mean(v[i-size:i+size])
    return avg

def ExploreLoss(reportpath, window_size):
    # results = pd.read_json(reportpath, orient="columns")
    with open(reportpath, "r") as f:
        results = json.load(f)
    for key in results.keys():
        print(f"{key}: {len(results[key])}")
    loss = pd.DataFrame({"loss": results["loss"]})
    ma = MovingAvg(loss.loss, int(window_size))
    PlotLoss(loss, ma)

def Dashboard(report: pd.DataFrame, window_size: int=100):
    loss_ma = MovingAvg(report.loss, window_size)
    bacc_ma = MovingAvg(report.batch_accuracy, window_size)
    lr_ma = MovingAvg(report.learning_rate, window_size)
    eacc = report.groupby("epoch").mean()[["batch_accuracy", "epoch_accuracy"]]
    f1_scores = report.groupby("epoch").mean(numeric_only=True)["f1_score"]
    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    fig.suptitle("Training Log")
    # plot loss
    axs[0, 0].set_title("loss")
    axs[0, 0].set_xlabel("batch")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].scatter(report.index, report.loss, s=0.3, c="red", label="value")
    axs[0, 0].plot(report.index, loss_ma, "-b", label="moving avg")
    axs[0, 0].legend()
    # plot batch acc
    axs[0, 1].set_title("batch accuracy")
    axs[0, 1].set_xlabel("batch")
    axs[0, 1].set_ylabel("accuracy")
    axs[0, 1].scatter(report.index, report.batch_accuracy, s=0.3, c="red")
    axs[0, 1].plot(report.index, bacc_ma, "-b")
    # plot epoch acc vs mean batch_acc
    axs[0, 2].set_title("epoch accuracy")
    axs[0, 2].set_xlabel("epoch")
    axs[0, 2].set_ylabel("accuracy")
    axs[0, 2].plot(eacc.index, eacc.epoch_accuracy, "-r",
                   linewidth=0.3, label="eval")
    axs[0, 2].plot(eacc.index, eacc.batch_accuracy, "-g",
                   linewidth=0.3, label="train")
    axs[0, 2].legend()
    # plot epoch PFbeta
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("accuracy")
    axs[1, 0].plot(f1_scores.index, f1_scores, "-r",
                   linewidth=0.3)
    # plot learning rate
    axs[1, 1].set_title("Learning Rate")
    axs[1, 1].set_xlabel("batch")
    axs[1, 1].set_ylabel("accuracy")
    axs[1, 1].scatter(report.index, report.learning_rate, s=0.3, c="red")
    axs[1, 1].plot(report.index, lr_ma, "-b")
    plt.tight_layout()
    plt.show()

def GetF1Score(metadatapath, submissionpath, savepath):
    md = pd.read_json(metadatapath, orient="index", convert_axes=False, convert_dates=False)
    lmap = {0: "L", 1: "R"}
    pats = md.patient_id.to_list()
    lats = md.laterality.to_list()
    rlats = [lmap[val] for val in lats]
    md["prediction_id"] = ["_".join(item) for item in zip(map(str, pats), rlats)]
    all_labels = md.loc[:, ["prediction_id", "cancer"]].copy()
    all_labels.drop_duplicates(inplace=True)
    results = pd.read_csv(submissionpath)
    gt = [int(all_labels[all_labels.prediction_id == i].cancer) for i in results.prediction_id]
    results["gt"] = gt
    score = PFbeta(results["gt"], results["cancer"], 1)
    print(f"F1: {score}")
    results.to_csv(savepath, index=False)


if __name__ == "__main__":
    rp = sys.argv[1]
    ws = int(sys.argv[2])
    if ws == None:
        ws = 100
    df = pd.read_csv(rp)
    Dashboard(df, ws)
