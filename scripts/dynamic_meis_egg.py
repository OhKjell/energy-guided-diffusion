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
    energy = -output[0][0]
    return energy





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







def MSE_sum(x):
    print("MMSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    print(x.shape)
    mse = 0
    next_image = torch.mean(x[0], dim=0, keepdim=True)
    first_image = torch.mean(x[0], dim=0, keepdim=True)
    print(next_image.shape)
    for i in range(x.shape[0] - 1):
        image = next_image
        next_image = torch.mean(x[i + 1], dim=0, keepdim=True)
        # mse += (torch.mean((image - next_image) ** 2)) ** 2
        mse += (torch.mean((image - first_image) ** 2)) ** 2
    print(f"MSEEEEEEEEEEEEEEEEEEEEEEEEEEEe: {mse}")
    return mse / (x.shape[0] - 1)



def mse(x, y):
    x = torch.mean(x[0], dim=0, keepdim=True)
    y = torch.mean(y[0], dim=0, keepdim=True)
    mse = torch.mean((x - y) ** 2)
    
    return mse




















if os.path.exists("output"):
    shutil.rmtree("output")
    os.makedirs("output")
# output_dir = f"output"
# one_dir = f"{output_dir}/one"
# if os.path.exists(one_dir):
#     shutil.rmtree(one_dir)
#     os.makedirs(one_dir)
# two_dir = f"{output_dir}/two"
# if os.path.exists(two_dir):
#     shutil.rmtree(two_dir)
#     os.makedirs(two_dir)
one_dir = f"output/one_dir"
os.makedirs(one_dir, exist_ok=True)
two_dir = f"output/two_dir"
os.makedirs(two_dir, exist_ok=True)
#output_dir = f"output"
model = EGG(num_steps=num_timesteps)





outputs = model.sample_video(
        energy_fn=dynamic_function,
        energy_fn2=MSE_sum,
        energy_scale=0,
        energy_scale2=10,
        num_samples=39,
        iterative = False,
        iterations=10
    )




print("hee")

test1 = []

for i, samples in enumerate(outputs):
    pass

for j, sample in enumerate(samples["sample"]):
    print(sample.shape)
    #samples_dir = f"{output_dir}/output_{j}"
    #os.makedirs(samples_dir, exist_ok=True)

    test1.append(sample)
    plt.imshow(np.transpose(sample.cpu().detach(), (1,2,0)))
    plt.axis("off")
    plt.savefig(f"{one_dir}/image_{j}.png")
    plt.close()


test1 = torch.stack(test1, dim=0)
print(dynamic_function(test1))



# outputs = model.sample_video(
#         energy_fn=dynamic_function,
#         energy_scale=10,
#         num_samples=39
#     )
# print("hee")
# test2 = []
# for i, samples in enumerate(outputs):
#     pass


# for j, sample in enumerate(samples["sample"]):
#     print(sample.shape)
#     test2.append(sample)
#     #samples_dir = f"{output_dir}/output_{j}"
#     #os.makedirs(samples_dir, exist_ok=True)
#     plt.imshow(np.transpose(sample.cpu().detach(), (1,2,0)))
#     plt.axis("off")
#     plt.savefig(f"{two_dir}/tt_image_{j}.png")
#     plt.close()
# test1 = torch.stack(test1, dim=0)
# test2 = torch.stack(test2, dim=0)
# print(dynamic_function(test1))
# print(dynamic_function(test2))