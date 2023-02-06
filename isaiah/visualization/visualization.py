import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import h5py
import random

def ExploreLoss(reportpath, window_size):
    # results = pd.read_json(reportpath, orient="columns")
    with open(reportpath, "r") as f:
        results = json.load(f)
    for key in results.keys():
        print(f"{key}: {len(results[key])}")
    loss = pd.DataFrame({"loss": results["loss"]})
    ma = MovingAvg(loss.loss, window_size)
    PlotLoss(loss, ma)

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

def GetTestSet(reportpath, datasetpath, savepath):
    with open(reportpath, "r") as f:
        results = json.load(f)
    train_set = set()
    all_samples = set()
    for batch in results["samples"]:
        train_set.update(batch)
    with h5py.File(datasetpath, "r") as ds:
        ds.visit(lambda n: all_samples.add(n))
    test_set = all_samples - train_set
    print(len(test_set))
    test_dict = {}
    test_dict["test"] = list(test_set)
    with open(savepath, "w") as sp:
        json.dump(test_dict, sp)

def ExploreTestSet(testsetpath, metadatapath):
    with open(testsetpath, "r") as f:
        test_dict = json.load(f)
    test_img_ids = set([int(item) for item in test_dict["test"]])
    md = pd.read_json(metadatapath, orient="index")
    missing_ids = set()
    for val in test_img_ids:
        if val in md.index:
            continue
        else:
            missing_ids.add(val)
    final_ids = test_img_ids - missing_ids
    test_md = md.loc[list(final_ids), :]
    print(f"Proportion of cancer images in data: {md[md.cancer == 1].count()/float(md.size)}")
    print(f"Proportion of cancer images in test_set: {test_md[test_md.cancer == 1].count()/float(test_md.size)}")

def CreateDummyTestSet(metadatapath, savepath):
    md = pd.read_json(metadatapath, orient="index", convert_axes=False, convert_dates=False)
    cancer_ids = set(md.loc[md.cancer == 1].index)
    non_cancer_ids = set(md.index) - cancer_ids
    bal_cancer_set = set(random.sample(list(cancer_ids), int(len(cancer_ids)/2)) + random.sample(non_cancer_ids, int(len(cancer_ids)/2)))
    test_set_size = len(bal_cancer_set)
    split_size = int(0.01*test_set_size)
    low_cancer_set = set(random.sample(cancer_ids, split_size) + random.sample(non_cancer_ids, test_set_size - split_size))
    high_cancer_set = set(random.sample(non_cancer_ids, split_size) + random.sample(cancer_ids, test_set_size - split_size))
    test_dict = {
        "balanced_cancer": list(bal_cancer_set),
        "low_cancer": list(low_cancer_set),
        "high_cancer": list(high_cancer_set)
    }
    with open(savepath, "w") as f:
        json.dump(test_dict, f)

if __name__ == "__main__":
    fp = sys.argv[1]
    sp = sys.argv[2]
    CreateDummyTestSet(fp, sp)
