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
import shutil
import copy
from mpl_toolkits.mplot3d import Axes3D
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
        
def label_standardization(data, mean=0, scale=1):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    new_data = (data - m) / s
    new_data = (new_data + mean) * scale / np.sqrt(data.shape[1])
    return new_data
        
        
def save_gif(data_list, gif_path, relational_label, moving_label, route, save_dir = "contempolary_images/", fig_size=(15.5, 9.2), font_title=24, duration=100, classes=tuple(range(8))):
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    each = len(data_list) // len(classes)
    for i in range(len(data_list)):
        fig = plt.figure(figsize=fig_size)
        plt.cla()
        ax = fig.add_subplot(1,2,1)
        ax.imshow(data_list[i])
        if i >= each * (len(classes)-1):
            ax.set_title(f"({label_discription[route[-1]]}) \n → ({label_discription[route[0]]})", fontsize=20)
        else:
            ax.set_title(f"({label_discription[route[i//each]]}) \n → ({label_discription[route[i//each+1]]})", fontsize=20)

        ax = fig.add_subplot(1,2,2, projection="3d")
        for lbl in classes:
            m = relational_label[lbl]
            x_ = m[0:1] 
            y_ = m[1:2]
            z_ = m[2:]
            ax.scatter3D(x_, y_, z_, label=f"{lbl}", s=150)
        a = moving_label[i:i+1,:]
        ax.set_title("relation label \n and current label", fontsize=16)
        ax.scatter3D(a[:,0:1], a[:,1:2], a[:,2:3], label=f"current label", s=100)
        ax.view_init(elev=30., azim=210)
        ax.legend()

        save_path = save_dir + f"{str(i).zfill(3)}"
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.savefig(save_path, dpi = 64, facecolor = "lightgray", tight_layout=True)
        plt.close()
        
    files = sorted(glob.glob(save_dir + '*.png'))
    images = list(map(lambda file: Image.open(file), files))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    
    
def get_moving_label(label, classes, delta=0.1):
    points = []
    for j in range(len(classes)-1):
        for i in range(int(1/delta)+1):
            alpha = delta * i
            point = (1-alpha)*label[classes[j]] + alpha*label[classes[j+1]]
            points.append(point)
    for i in range(int(1/delta)+1):
        alpha = delta * i
        point = (1-alpha)*label[classes[-1]] + alpha*label[classes[0]]
        points.append(point)
    points = np.array(points)
    return points

def get_moving_path(target_label, delta=0.1, route=np.array([])):
    if len(route)==0:
        route = two_opt(target_label, 0)
    moving_label = get_moving_label(target_label, route, delta)
    return moving_label, route

############### https://codeday.me/jp/qa/20190604/929507.html #######################################
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
    return route # When the route is no longer improving substantially, stop searching and return the route.