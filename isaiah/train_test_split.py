import sys, os
from glob import glob
import json
import yaml
import numpy as np
import pandas as pd
from addict import Dict
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    output = {}
    config_file = os.path.abspath(sys.argv[1])
    cfgs = Dict(yaml.load(open(config_file, "r"), Loader=yaml.Loader))
    md = pd.read_json(cfgs.metadata_dest, orient="index", convert_axes=False, convert_dates=False)
    output["Train"], output["Test"] = train_test_split(md.index, test_size=cfgs.test_size)
    with open(cfgs.traintest_dest, "w") as f:
        json.dump(output, f)

