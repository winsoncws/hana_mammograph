import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import h5py

def ExploreLoss(filepath, window_size):
    # results = pd.read_json(filepath, orient="columns")
    with open(filepath, "r") as f:
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

def ExploreTestSet(filepath, datasetpath, savepath):
    with open(filepath, "r") as f:
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

if __name__ == "__main__":
    fp = sys.argv[1]
    fp2 = sys.argv[2]
    sp = sys.argv[3]
    ExploreTestSet(fp, fp2, sp)
