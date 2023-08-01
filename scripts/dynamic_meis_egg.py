"""
Run diffusion MEIs on a set of units.
"""
import gc
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
#import wandb
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from egg.diffusion import EGG
from egg.models import models
import scripts.vgg as vgg
import matplotlib.pyplot as plt
import os
import shutil
import glob
import cv2
from dynamic.models.helper_functions import get_model_and_dataloader, get_model_temp_reach, get_model_and_dataloader_for_nm

#from dynamic.utils.global_functions import *
# from IPython.display import Video
from dynamic.models.helper_functions import get_model_and_dataloader
# get_model_temp_reach, get_model_and_dataloader_for_nm
# from dynamic.evaluations.single_cell_performance import get_performance_for_single_cell
# from dynamic.meis.visualizer import get_model_activations
# from Python.display import Video
# from dynamic.datasets.stas import get_cell_sta, show_sta, get_sta
# import pickle




data_type = 'marmoset'
directory = "dynamic/dynamic_models"
filename = "lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_16_tr_10_ik_27x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_48.0000_gt_0.0740_l1_0.0230_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90"
device = 'cuda'
model_fn = 'models.FactorizedEncoder.build_trained'
seed = 8

#build dynamic model


dynamic_model = get_model_and_dataloader_for_nm(
            directory,
            filename,
            model_fn=model_fn,
            device=device,
            data_dir=None, # if data_dir is None, root of the project is considered
            test=False,
            seed=seed,
            data_type=data_type,
        )


print("yeaahh")