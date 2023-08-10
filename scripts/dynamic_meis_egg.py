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


num_timesteps = 10


def get_gpu_memory(device=0):
    properties = torch.cuda.get_device_properties(device)
    total_memory = properties.total_memory
    available_memory = total_memory - torch.cuda.memory_allocated(device)
    print(f"Total GPU memory: {total_memory / (1024**3):.2f} GiB")
    print(f"Used GPU memory: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GiB")
    print(f"Available GPU memory: {available_memory / (1024**3):.2f} GiB")

# Call the function with device=0 (assuming you have one GPU)
get_gpu_memory()



torch.backends.cudnn.enabled = False
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








if os.path.exists("output"):
    shutil.rmtree("output")
    os.makedirs("output")

output_dir = f"output"
model = EGG(num_steps=num_timesteps)



def dynamic_function(x):
    x = x.permute(1, 0, 2, 3).unsqueeze(0)
    print(f"SHAPE OF DYNAMIC INPUT: {x.shape}")
    x = x.mean(dim=1, keepdim=True)
    print(f"SHAPE OF DYNAMIC INPUT: {x.shape}")
    print(x.dtype)
    output = dynamic_model(x)
    #output = x
    print(output.shape)
    #output = x
    if output.requires_grad:
        print("Tensor 'out' is part of the computation graph.")
    else:
        print("Tensor 'pit' is not part of the computation graph.")
    loss = output.mean()
    return loss





# #tensor_shape = (1, 1, 40, 80, 90)
# tensor_shape = (39, 3, 256, 256)

# # tensor_shape = (1, 64, 2, 36, 46)
# # tensor_shape = (1, 8, 40, 80, 90)
# tensor = torch.zeros(tensor_shape).to(device).double().requires_grad_()
# tensor = tensor.contiguous()

# output = dynamic_function(tensor)
# #output= output.mean()
# norm_grad = torch.autograd.grad(outputs=output, inputs=tensor)
# print(norm_grad)
# print(norm_grad[0].shape)







def tmp(x):
    return x[0][0][0][0]


outputs = model.sample_video(
        energy_fn=dynamic_function,
        energy_scale=5,
        num_samples=39
    )
print("hee")
for i, samples in enumerate(outputs):
    pass
for j, sample in enumerate(samples["sample"]):
    print(sample.shape)
    #samples_dir = f"{output_dir}/output_{j}"
    #os.makedirs(samples_dir, exist_ok=True)
    plt.imshow(np.transpose(sample.cpu().detach(), (1,2,0)))
    plt.axis("off")
    plt.savefig(f"{output_dir}/new_image_{j}.png")
    plt.close()

                        