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

# experiment settings
num_timesteps = 100
energy_scale = 5  # 20
#energy_scale2 = 1
seeds = [0]#np.arange(1)
unit_seed=27#42
norm_constraint = 25  # 25
model_type = "task_driven"  #'task_driven' #or 'v4_multihead_attention'
energyfunction = "MSE" #"MSE" "VGG" "None"
number_units = 3
number_frames = np.arange(4)
create_vgg = True
fps = 20
unit_ids = None #None [id]
#for vgg
vgg_gray=True
escale2 = [0, 1, 2, 3, 4, 5]

