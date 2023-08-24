#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus April 18 17:18:50 2023

@author: zhsc
"""
#coding:utf-8
import os
import random

import imageio
import torchvision
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from funcs.utils import *
import torch
from PIL import Image
import scipy.io as scio

  
def loadSubjectData(path):
    
    data_imgs = scio.loadmat(path) 
    
    img_flair = data_imgs['data_img']['flair'][0][0].astype(np.float32)
    img_t1    = data_imgs['data_img']['t1'][0][0].astype(np.float32)
    img_t1ce  = data_imgs['data_img']['t1ce'][0][0].astype(np.float32)
    img_t2    = data_imgs['data_img']['t2'][0][0].astype(np.float32)
    
    # crop 160*180 images
    # img_t1    = img_t1[40:200,20:200,:]
    # img_t1ce  = img_t1ce[40:200,20:200,:]
    # img_t2    = img_t2[40:200,20:200,:]
    # img_flair = img_flair[40:200,20:200,:]

    img_t1 = img_t1[40:200, 20:200]
    img_t1ce = img_t1ce[40:200, 20:200]
    img_t2 = img_t2[40:200, 20:200]
    img_flair = img_flair[40:200, 20:200]
    return img_t1,img_t1ce,img_t2,img_flair


def load_gansim_data(path):
    data = np.load(path)
    z, y, x = data.shape
    for i in range(z):
        image_i = data[i]
        i_min = np.min(image_i)
        i_max = np.max(image_i)
        image_0255 = 255*((image_i-i_min)/(i_max-i_min))
        imageio.imwrite('data/gansim_data/' + str(i) + '.png', image_0255)
    print(data.shape)


class MUltiConditionData_load(data.Dataset):

    def __init__(self, opt, transforms=None, train=True, test=False, data_path='data/', txt_path='data/dataset2.txt'):
    # def __init__(self, opt, transforms=None, train=True, test=False, data_path='data/case2d/', txt_path='data/case2d/dataset.txt'):
        self.opt = opt
        self.test = test
        self.train = train

        data_paths = []
        data_name = open(txt_path, 'r')
        for line in data_name:
            line = line.strip('\n')
            temps = line.split()
            data_paths.append((data_path+'images/'+temps[0], data_path+'images2/'+temps[0]))
        self.data_paths = np.array(data_paths)
        # random.shuffle(data_paths)
        # if train:
        #     self.data_paths = np.array(data_paths)[:150]
        # else:
        #     self.data_paths = np.array(data_paths)

    def __getitem__(self, index):
        # path
        cur_path = self.data_paths[index]

        image1 = Image.open(cur_path[0]).convert('L')
        image2 = Image.open(cur_path[1]).convert('L')

        image1 = torchvision.transforms.ToTensor()(image1)
        image2 = torchvision.transforms.ToTensor()(image2)
        # print(image2.shape)

        return image1, image2, image1, image1

    def __len__(self):
        return len(self.data_paths)


def load_reference_model(path):
    file = open(path)
    value_list = []
    scale = ''
    cnt = 0
    for line in file:
        if cnt == 0:
            scale = line
        elif cnt >= 3:
            line = line.replace('\n', '')
            value_list.append(float(line))
        cnt+=1
    scale_list = scale.split(' ')
    x = int(scale_list[0])
    y = int(scale_list[1])
    z = int(scale_list[2])
    ti = np.reshape(value_list, [z, y, x])
    for k in range(0, z):
        for i in range(0, y):
            for j in range(0, x):
                ti[k, i, j] = float(ti[k, i, j])
    return ti


class MUltiConditionData3D_load(data.Dataset):
    def __init__(self, opt, transforms=None, train=True, test=False, data_path='data/case_study/100/', txt_path='data/case_study/100/dataset.txt'):
    # def __init__(self, opt, transforms=None, train=True, test=False, data_path='data/large_deltaic/EXP large Deltaic/', txt_path='data/large_deltaic/EXP large Deltaic/dataset.txt'):
        self.opt = opt
        self.test = test
        self.train = train

        data_paths = []
        data_name = open(txt_path, 'r')
        for line in data_name:
            line = line.strip('\n')
            temps = line.split()
            data_paths.append((data_path+'images/'+temps[0], data_path+'images2/'+temps[0]))
        self.data_paths = np.array(data_paths)[:]
        # random.shuffle(data_paths)
        # if train:
        #     self.data_paths = np.array(data_paths)[:200]
        # else:
        #     self.data_paths = np.array(data_paths)[200:]

    def __getitem__(self, index):
        # path
        cur_path = self.data_paths[index]
        image1 = load_reference_model(cur_path[0])
        # image1 = image1[:72, :, :]
        image1 = torch.from_numpy(image1).float()
        image1 = torch.unsqueeze(image1, 0)

        image2 = load_reference_model(cur_path[1])
        # image2 = image2[:72, :, :]
        image2 = torch.from_numpy(image2).float()
        image2 = torch.unsqueeze(image2, 0)

        return image1, image2, image1, image1

    def __len__(self):
        return len(self.data_paths)


class MUltiConditionData3DLarge_load(data.Dataset):

    def __init__(self, opt, transforms=None, train=True, test=False, data_path='data/large_deltaic/EXP large Deltaic/', txt_path='data/large_deltaic/EXP large Deltaic/dataset2.txt'):

        self.opt = opt
        self.test = test
        self.train = train

        data_paths = []
        data_name = open(txt_path, 'r')
        for line in data_name:
            line = line.strip('\n')
            temps = line.split()
            data_paths.append((data_path+'images/'+temps[0], data_path+'images2/'+temps[0]))
        self.data_paths = np.array(data_paths)[:200]
        # random.shuffle(data_paths)
        # if train:
        #     self.data_paths = np.array(data_paths)[:200]
        # else:
        #     self.data_paths = np.array(data_paths)[200:]

    def __getitem__(self, index):
        # path
        cur_path = self.data_paths[index]
        image1 = load_reference_model(cur_path[0])
        image1 = image1[:72, :, :]
        image1 = torch.from_numpy(image1).float()
        image1 = torch.unsqueeze(image1, 0)

        image2 = load_reference_model(cur_path[1])
        image2 = image2[:72, :, :]
        image2 = torch.from_numpy(image2).float()
        image2 = torch.unsqueeze(image2, 0)

        return image1, image2, image1, image1

    def __len__(self):
        return len(self.data_paths)
# if __name__ == '__main__':
    # load_gansim_data('data/gansim.npy')
    # load_gansim_data('data/stanfordp1.npy')