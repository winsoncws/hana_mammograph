import os, sys
import numpy as np
import scipy
import skimage
import sklearn
import SimpleITK as sitk
import nibabel
import pandas
import cv2
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from os.path import join, dirname, exists, basename, altsep, isdir
from time import time
import math
from PIL import Image
import pickle, json, yaml
from skimage.morphology import erosion, dilation
from PIL import Image
import torch

def read_pickle(file):
    with open(file, "rb") as p:
        return pickle.load(p)

def imshow(x, s=(5,5), t=None):
    """
    x: array
    s: image size
    t: title
    """
    fig = plt.figure(figsize=s)
    plt.imshow(x)
    plt.axis("off")
    if t:
        plt.title(t)
    plt.show()

def show_flatten(data1, data2=None, Vmin=None, Vmax=None, size=(6,6),
                 alpha=0.5, cmap='gist_heat', border_spacing=0):
    data1 = npy(data1)
    data2 = npy(data2)
    plt.figure(figsize=size)
    plt.imshow(flatten3d(data1,border_spacing), cmap='gray', vmin=Vmin, vmax=Vmax)
    if data2 is not None:
        if data1.shape != data2.shape: print('Warning: data1 and data2 shape not match')
        data2 = flatten3d(data2,border_spacing)
        if isinstance(alpha, np.ndarray):
            if len( alpha.shape ) > len( data2.shape ):
                alpha = flatten3d(alpha,border_spacing)
        plt.imshow( data2, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.show()

def flatten3d(data, border_spacing=0):
    assert isinstance(data, np.ndarray), 'expecting ndarray input'
    if len(data.shape) == 2: return data
    assert len(data.shape) == 3, 'expecting either 2D or 3D input images'
    f,h,w = data.shape
    b = border_spacing
    sqrt,wide = np.sqrt(f), int(np.sqrt(f))
    if (sqrt == wide) or (f % wide == 0):
        tile = np.ones( ( (h+b)*wide, (w+b)* int(f/wide)    )) * 0
    else:
        tile = np.ones( ( (h+b)*wide, (w+b)*(int(f/wide)+1 ))) * 0
    for i in range(f):
        x,y = np.mod(i,wide), int(i/wide)
        tile[x*(h+b):x*(h+b)+h,y*(w+b):y*(w+b)+w] = data[i]
    return tile


def get_mask_boundary(mask, inner=4, outer=6):
    """ To create boundary for segmented mask. The boundary thickness is outer ring substracted by inner ring.
        inner: inner ring layers compare to actual boundary. Can be both pos/neg integer
        outer: outer ring layers compare to actual boundary. Can be only positive integer

        NOTE: Recommend to flatten 3Dmask into 2D before getting boundary.
    """

    assert isinstance(mask, np.ndarray), 'expect a numpy array'
    #check_types(mask[0], float) # grayscale mask

    inner_method = dilation
    if inner < 0: inner=-inner; inner_method=erosion
    outer_mask, inner_mask = np.copy(mask), np.copy(mask)
    for _ in range(outer): outer_mask = dilation(    outer_mask)
    for _ in range(inner): inner_mask = inner_method(inner_mask)
    return outer_mask - inner_mask

color_map = {'red'   : [255,0  ,0  ], 'green' : [0  ,255,0  ],
             'blue'  : [0  ,0  ,255], 'violet': [129,112,249],
             'grey'  : [126,126,126], 'pink'  : [255,0  ,255],
             'yellow': [255,255,0  ], 'orange': [255,100,0  ],
             'cyan'  : [0  ,255,255], 'brown' : [205,133,63 ],
             'dblue' : [44, 52, 144], 'lblue' : [0  ,255,255],
             'darkgreen':[36,129,57],
             }

random_color_map = [
    [180, 111, 87], [241, 237, 205], [231, 25, 220], [235, 72, 222], [180, 150, 35], [222, 130, 37], [56, 212, 223], [33, 84, 78], [64, 241, 116], [237, 219, 27], [237, 88, 155], [224, 167, 188], [247, 130, 174], [92, 3, 160], [3, 170, 178], [200, 188, 195], [115, 176, 161], [238, 107, 163], [47, 169, 159], [153, 189, 138], [23, 201, 222], [44, 209, 66], [160, 42, 30], [41, 60, 14], [82, 136, 48], [209, 160, 180], [42, 58, 227], [248, 44, 95], [36, 81, 238], [254, 144, 90], [27, 218, 247], [116, 99, 203], [78, 245, 96], [9, 23, 176], [93, 62, 46], [172, 68, 48], [203, 217, 58], [1, 83, 159], [186, 31, 41], [206, 254, 179], [16, 167, 138], [150, 204, 90], [194, 218, 104], [72, 232, 156], [46, 27, 197], [192, 239, 46], [244, 153, 36], [229, 215, 180], [182, 35, 170], [202, 166, 233], [109, 242, 220], [207, 63, 185], [15, 241, 44], [32, 97, 220], [202, 110, 37], [51, 112, 188], [148, 117, 87], [76, 18, 91], [19, 49, 109], [183, 138, 24], [237, 78, 93], [253, 32, 87], [79, 103, 24], [113, 177, 248]
]

def overlay_image_multimask(image, masks, colors=['pink','blue'],
                            opacity=0.8,  only_border=True,
                            inner=0, outer=5, cmap=color_map ):
    """ Overlay image and multi-labels-mask together, to view/save via PIL or plt.imshow.

        Args:
            image:       2D-image, grayscale
            masks:       either a list of 2D masks, or single 2D-masks of multi-labels
            colors:      list of colors found in cmap. Must be same number with mask-labels
            opacity:     value between 0 to 1
            only_border: True=show only border of mask, False=show opaque masks
            inner:       inner ring layers compare to actual boundary. Can be both pos/neg integer.
            outer:       outer ring layers compare to actual boundary. Can be only positive integer.
            cmap:        a dictionary of color_map: key=colour_str, value=[r,g,b]

        Returns:
            overlay 2D-image in RGB format
    """
    ### checking input masks and image
    assert isinstance(image,np.ndarray), 'input image should be ndarray, please apply flatten3d'
    assert len(image.shape) == 2, 'dimension of image should be 2'
    if isinstance(masks,np.ndarray):
        mask_lbl = np.unique(masks)
        mask = []
        for i in mask_lbl[1:]:  mask.append( (masks == i ) * 1 )
        masks = mask
    else:
        assert isinstance(masks,list), 'input masks should be a list of masks or ndarray'
    len_mask = len(masks)
    for mask in masks:
        assert len(mask.shape) == 2, 'dimension of each mask should be 2'
    ### checking cmap and colours
    if cmap is not None:
        assert isinstance(cmap,dict), "cmap should be a dictionary of key=colour_str, value=[r,g,b]"
    assert 0. <= opacity <= 1., 'opacity value should be between 0~1'
    if colors is None:
        if cmap is None: cmap = {}
        if len(cmap) < len_mask:
            cmap = {str(idx):i for idx,i in enumerate(random_color_map)}
        if len(cmap) < len_mask:
            print('-- {}-cmap-colors not enough for {}-channel-mask, use random-color instead --'.format(len(cmap),len_mask))
            cmap = {str(i):list(np.random.randint(0,255,3)) for i in range(len_mask)}
            #print('current rgb color', list(cmap.values()))
        colors = list(cmap.keys())[:len_mask]
    else:
        assert isinstance(colors,list),   'input should be a list of colour_str, or None'
        assert len(masks) <= len(colors), 'masks_list and colors_list length mismatch'
    ### start mapping colour to its RGB
    colour_str = ', '.join( list(cmap.keys()) )
    try:    rgbs = [ np.array(cmap[i]) for i in colors[:len(masks)] ]
    except: raise Exception('current available colours are {}'.format(colour_str))
    ### normalize image and convert grayscale to rgb
    img_ref = ( image - image.min() )
    img_ref = np.uint8( (img_ref * 255. / img_ref.max() ).clip(0,255))
    im_r ,im_g ,im_b  = np.copy(img_ref),np.copy(img_ref),np.copy(img_ref)
    ### overlaying multichannel masks
    if masks: # is mask is empty, just print image
        for mask,rgb in zip(masks,rgbs):
            mask  = (mask > 0.) * 1
            if only_border: mask = get_mask_boundary(mask, inner=inner, outer=outer)
            roi_r = im_r * mask * (1-opacity)
            roi_g = im_g * mask * (1-opacity)
            roi_b = im_b * mask * (1-opacity)
            bg_r  = im_r * (1-mask)
            bg_g  = im_g * (1-mask)
            bg_b  = im_b * (1-mask)
            im_r  = np.uint8(((mask * rgb[0] * opacity + roi_r) + bg_r).clip(0,255))
            im_g  = np.uint8(((mask * rgb[1] * opacity + roi_g) + bg_g).clip(0,255))
            im_b  = np.uint8(((mask * rgb[2] * opacity + roi_b) + bg_b).clip(0,255))
            img   = np.asarray(np.dstack((im_r,im_g,im_b)), dtype=np.uint8)
    else:
        img   = np.asarray(np.dstack((im_r,im_g,im_b)), dtype=np.uint8)
    return img

def opentif(x):
    im = Image.open(x)
    return np.array(im)

def opennrrd(x):
    im = Image.open(x)
    return np.array(im)

def show(x, size=(5,5)):
    fig = plt.figure(figsize=size)
    plt.imshow(x)
    plt.axis('off'); plt.show()

def save_as_png(img, dest):
    Image.fromarray(img).save(dest)

def save_grayscale_as_png(img, dest):
    np.asarray(np.dstack([img/img.max()*255]*3), dtype = np.uint8)
    Image.fromarray(img).save(dest)

    
########### TORCH.UTILS ###########

def compare(pred, actual, size=(6,3)):
    ## show figure side by side predicted and actual ground-truth
    p = npy(pred).squeeze()
    i = npy(img).squeeze()
    show_flatten( np.concatenate((i,p), -1) , size=size)

def npy(x):
    ## convert tensor.gpu back to numpy
    x = x.cpu().detach().numpy() if type(x) == torch.Tensor else x
    return x

def value(x):
    ## convert tensor.gpy back to np.float
    return np.round(npy(x),3 )

