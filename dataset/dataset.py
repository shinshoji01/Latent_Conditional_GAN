import torch
import torch.nn as nn
import numpy as np
import glob
import torch.utils.data
from PIL import Image
import warnings
import itertools
warnings.filterwarnings("ignore")

import sys
sys.path.append("../")
from util.util import *

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, label_root, transform, dataset_label, classes, data_type="train", do_min_max=True, train_num=2000, val_num=500, test_num=500):
        
        self.transform = transform
        self.do_min_max = do_min_max
        self.images = []
        self.labels = []

        def make_path(name):
            name = name.split(".")[0]
            return root + name + ".png"
        
        def get_class_label(n_class_type):
            test = []
            for com in list(itertools.combinations_with_replacement([-1,1], n_class_type)):
                for per in list(itertools.permutations(com)):
                    test.append(per)
            cl = list(set(test))
            cl.sort(reverse=True)
            return cl
        
        def combine2bool_and(a, b):
            return np.array(np.array(a,np.int) * np.array(b,np.int),bool)
        
        # Get class label
        cl = get_class_label(len(dataset_label["class"]))
        
        # 画像のpathとラベルを生成する
        images_dir = {}
        labels = []
        
        for i in range(len(classes)):
            images_dir[i] = []
            for label_path in glob.glob(label_root + "*"):
                info = pickle_load(label_path)
                if len(dataset_label["delete"]) == 0:
                    index_deleted = np.array(np.ones(info.shape[0]), bool)
                else:
                    index_deleted = np.sum(1-np.array(info[:, np.array(dataset_label["delete"])]=="-1", np.int), axis=1)==0
                if len(dataset_label["existed"]) == 0:
                    index_existed = np.array(np.ones(info.shape[0]), bool)
                else:
                    index_existed = np.sum(1-np.array(info[:, np.array(dataset_label["existed"])]=="1", np.int), axis=1)==0
                index_conditioned = np.array(np.array(index_deleted,np.int) * np.array(index_existed,np.int),bool)
                info_con = info[combine2bool_and(index_deleted, index_existed)]
                for j in range(len(dataset_label["class"])):
                    if j==0:
                        bool_list = info_con[:, dataset_label["class"][j]]==str(cl[i][j])
                    else:
                        bool_list = combine2bool_and(bool_list, info_con[:, dataset_label["class"][j]]==str(cl[i][j]))
                path_list = list(map(make_path, info_con[bool_list,0]))
                path_list.sort()
                images_dir[i] += path_list
                
            images_dir[i].sort()
            new_train_num = min(train_num, len(images_dir[i])-val_num-test_num)
            if data_type=="train":
                images_dir[i] = images_dir[i][:new_train_num]
            if data_type=="val":
                images_dir[i] = images_dir[i][new_train_num:new_train_num+val_num]
            if data_type=="test":
                images_dir[i] = images_dir[i][new_train_num+val_num:new_train_num+val_num+test_num]
            labels.append(np.array([i] * len(images_dir[i])))
                
        # 一つのリストにまとめる
        for label in classes:
            for image, label in zip(images_dir[label], labels[label]):
                self.images.append(image)
                self.labels.append(label)


    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)
            
        if self.do_min_max:    
            image = torch.Tensor(min_max(image.numpy(), mean0=True))
        
        return image, label
    
    def __len__(self):
        
        return len(self.images)