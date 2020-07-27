import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import torch.utils.data
import torchvision.transforms as transforms
import copy
import lightgbm as lgb
from PIL import Image
import warnings
import itertools
warnings.filterwarnings("ignore")

def ordinal(i):
    return str(i)+({1:"st",2:"nd",3:"rd"}.get(i if 14>i>10 else i % 10) or "th") 


def cuda_to_numpy(x):
    return x.detach().to("cpu").numpy()


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data


def min_max(x, axis=None, mean0=False, get_param=False):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    if get_param:
        return result, min, max
    return result


def image_from_output(output):
    image_list = []
    output = output.detach().to("cpu").numpy()
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list


def weights_init(m):
    # 重みの初期化
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('linear') != -1:        # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('batchnorm') != -1:     # バッチノーマライゼーションの場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)