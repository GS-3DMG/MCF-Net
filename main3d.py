#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/04/20

@author: Zhesi Cui
"""

# from SurvivalPredictionModel import SurvivalPredictionModel
# from HiNet_SynthModel import LatentSynthModel
from MCF3D_SynthModel import LatentSynthModel
from config import opt
import fire


def train(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.train()


def test(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.test(120, "result_case")


def test_rotate(**kwargs):
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    rotate_degree = 90
    SynModel.test_rotate_soft_data(25, rotate_degree, "result_3d_rotate_test")


if __name__ == '__main__':
    # https://github.com/taozh2017/HiNet
    # python main3d.py train --batch_size=2 --task_id=3d
    # python main3d.py test_rotate --batch_size=1 --task_id=1
    fire.Fire()
    # SynModel = LatentSynthModel(opt=opt)
    # SynModel.train()

