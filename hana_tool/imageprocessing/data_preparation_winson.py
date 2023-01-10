import os, sys, time, platform
from os.path import join, dirname, basename, exists, splitext
from glob import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import cv2
import monai
import torch
import sklearn
import skimage
import torchio as tio
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from skimage.measure import regionprops
from scipy.ndimage import label as sclabel

def preprocessing(dcm_path, crop_size=(256,256), wh_ratio=0.7):
    # dcm_path : dicom filepath
    # crop_size: the final cropped ROI scaling to be saved as training data
    # wh_ratio : to prevent ROI overly stretched. bigger number to retain orignal image aspect-ratio.
    size = (512,512)
    dcm  = pydicom.read_file(dcm_path)
    arr  = dcm.pixel_array
    img  = resize2d(arr, size ).astype(float) # to make sclabel faster
    img  = img.max() - img if dcm.PhotometricInterpretation == "MONOCHROME1" else img - img.min()
    img  = 255. * img / img.max()

    regions, nums = sclabel( 1*(img > 1) )
    nums, areas   = np.unique( regions, return_counts=True)
    num_maxarea   = np.argmax( areas[1:] ) + 1
    breast_region = (regions == num_maxarea).astype('uint8')
    img[breast_region == 0] = 0               # to remove unwanted tags/labels/comments
    x1,y1,x2,y2 = regionprops(breast_region)[0].bbox
    h, w = x2-x1, y2-y1
    if w/h < wh_ratio:
        new_w = int( h * wh_ratio ) 
        if y2 == size[1]:
            y1 = size[1] - new_w
        else:
            y2 = new_w
    crop         = img[x1:x2,y1:y2]
    crop_resized = resize2d(crop, crop_size).clip(0,255).astype(float)
    img = img.clip(0,255)
    return crop_resized


if __name__ == "__main__":
    root      = "/home/nni/code/kaggle_rsna"
    working   = join(root,"working")
    data      = "/home/dataset/kaggle/input"
    train_csv = pd.read_csv(join(data,"train.csv"))
    dense_map = {"A":1,"B":2,"C":3,"D":4}
    
    crop_size = (256,256)
    wh_ratio  = 0.7
    print("prepare for data preprocessing and saving dataset into", join(data, "preprocessed") )
    
    for enum, case in enumerate(cases[:]):
        img_id  = splitext(basename(case))[0]
        print(enum, img_id)
        info = train_csv[train_csv["image_id"] == int(img_id)]
        if info.view.values[0] not in ["CC","MLO"]:
            continue
        i = {i[0]: info.get(i).values[0] for i in ["patient_id","laterality","view","cancer","age", "implant","density","machine_id"]}
        i["diff"] = info.get("difficult_negative_case").values[0]
        i["d"]    = dense_map.get(i['d'],0)  ## "N/A":0,"A":1,"B":2,"C":3,"D":4
        i["v"]    = "0" if i["v"] == "CC" else "1"
        img = preprocessing(case, crop_size, wh_ratio)
        if i['l'] == "R":
            img = img[:,::-1] ## default all image Left-Side

        dest_name = f"c{i['c']}_v{i['v'].ljust(1)}_diff{i['diff']*1}_dense{i['d']}_im{i['i']}_age{int(i['a'])}_{i['l']}_mac{i['m']}_pat{i['p']}_img{img_id}"

        np.save( join(data, "preprocessed", dest_name ), img )

