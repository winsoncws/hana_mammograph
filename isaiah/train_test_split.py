import sys, os
from glob import glob
import json
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    output = {}
    dataDir = os.path.abspath(sys.argv[1])
    saveFile = sys.argv[2]
    dataFiles = sorted(glob(os.path.join(dataDir, "*.npz"), recursive=True))
    output["Train"], output["Test"] = train_test_split(dataFiles, test_size=0.2)
    with open(saveFile, "w") as f:
        json.dump(output, f)

