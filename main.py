#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/03/21

@author: Zhesi Cui
"""

#from SurvivalPredictionModel import SurvivalPredictionModel
# from HiNet_SynthModel import LatentSynthModel
from MCF_SynthModel import LatentSynthModel
from config import opt
import fire

    
def train(**kwargs):
    
    opt.parse(kwargs)
    
    SynModel = LatentSynthModel(opt=opt)
    SynModel.train() 
    

def test(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.test(200, "result_case2d/")


def test_rotate(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    rotate_degree = 90
    SynModel.test_rotate_soft_data(300, rotate_degree, "result_2d_rotate_test/")


def test_single(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.test_single(300, "result_pic_single2d/", 100)


def visualize_fmap(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.test_visualize(355, "visualize_f/")

   
if __name__ == '__main__':
    # https://github.com/taozh2017/HiNet
    # python main.py train --batch_size=128 --task_id=2 --gpu_id=[0,1]
    # python main.py test --batch_size=1 --task_id=1
    # python main.py test_rotate --batch_size=1 --task_id=1
    fire.Fire()
    # SynModel = LatentSynthModel(opt=opt)
    # SynModel.train()

