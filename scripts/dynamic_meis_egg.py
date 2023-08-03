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
from dynamic.utils.global_functions import get_cell_names

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
from dynamic.meis.visualizer import get_model_activations
# from Python.display import Video
# from dynamic.datasets.stas import get_cell_sta, show_sta, get_sta
# import pickle


num_timesteps = 100



retina_index = 1
data_type = 'marmoset'
directory = "src/dynamic/dynamic_models"
filename = "dynamic_model"
device = 'cuda'
model_fn = 'dynamic.models.FactorizedEncoder.build_trained'
seed = 8

#build dynamic model

dataloader, dynamic_model, config = get_model_and_dataloader_for_nm(
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
tier = 'train'
inputs, targets = next(iter(dataloader[tier][f'0{retina_index+1}']))

print(inputs.shape) 

# the targets shape is (batch_size, num_of_neurons, time_chunk)
print(targets.shape)

print("HHHHHHHHHHHHHHHHHHHHHHHHHHH")
print(dynamic_model.config_dict["img_h"])
print(dynamic_model.config_dict["img_w"])

tensor_shape = (1, 1, 40, 80, 90)
tensor = torch.zeros(tensor_shape).to(device).double()
print("##################################")

cell_names = get_cell_names(retina_index=1, explained_variance_threshold=0.15, config=dynamic_model.config_dict['config'])
cell_indices = list(range(len(cell_names)))
print(cell_names)
print(cell_indices)

print("##########################")

for key in dataloader.keys():
    print(key)
    print(dynamic_model.config_dict["n_neurons_dict"]["02"])
print("##########################")
print(get_model_temp_reach(dynamic_model.config_dict))
#activation = get_model_activations(dynamic_model, tensor)
#print(activation.shape)
#print(activation)
print("##########################")

output = dynamic_model(tensor)
print(output)

### diffusion model


if os.path.exists("output"):
    shutil.rmtree("output")
    os.makedirs("output")

output_dir = f"output"
model = EGG(num_steps=num_timesteps)



def tmpt_func(x):
    return 0


samples = model.sample(
        energy_fn=None,
        energy_scale=0,
        num_samples=10
    )

for i, sample in enumerate(samples):
    plt.imshow(np.transpose(sample["sample"].cpu().detach().squeeze(), (1,2,0)))
    plt.savefig(f"{output_dir}/{i}.png")
    plt.close()
                        