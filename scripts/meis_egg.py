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
energy_scale = 10  # 20
energy_scale2 = 10
seeds = np.arange(1)
unit_seed=0
norm_constraint = 25  # 25
model_type = "task_driven"  #'task_driven' #or 'v4_multihead_attention'
energyfunction = "VGG" #"MSE" "VGG" "None"
number_units = 2
number_frames = np.arange(5)
create_vgg = True


def do_run(model, energy_fn, energy_fn2, desc="progress", grayscale=False, seed=None, run=1):
    if seed is not None:
        torch.manual_seed(seed)

    cur_t = num_timesteps - 1

    samples = model.sample_frame(
        energy_fn=energy_fn,
        energy_fn2= energy_fn2,
        energy_scale=energy_scale,
        energy_scale2=energy_scale2
    )

    for j, sample in enumerate(samples):
        cur_t -= 1
        if (j % 10 == 0) or cur_t == -1:

            energy = energy_fn(sample["pred_xstart"])

            for k, image in enumerate(sample["pred_xstart"]):
                filename = f"output/{str(run)}{desc}_{j:05}.png"
                if grayscale:
                    image = image.mean(0, keepdim=True)

                # normalize
                tar = image / torch.norm(image) * norm_constraint * 256 / 100

                tqdm.write(
                    f'step {j} | train energy: {energy["train"]:.4g} | val energy: {energy["val"]:.4g} | cross-val energy: {energy["cross-val"]:.4g}'
                )

                import matplotlib.pyplot as plt

                plt.imshow(tar.cpu().detach().squeeze(), cmap="gray", vmin=-1.7, vmax=1.7)
                plt.axis("off")
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                #plt.savefig(
                #    filename, transparent=True, bbox_inches="tight", pad_inches=0
                #)

    return energy, sample["sample"]


if __name__ == "__main__":

    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output")
    
    if os.path.exists("frames"):
        shutil.rmtree("frames")
    os.makedirs("frames")

    data_driven_corrs = np.load("./data/data_driven_corr.npy")
    units = np.load("./data/pretrained_resnet_unit_correlations.npy")
    available_units = (data_driven_corrs > 0.5) * (units > 0.5)

    np.random.seed(unit_seed)
    units = np.random.choice(np.arange(len(available_units))[available_units], number_units)


        # Initialize the video writer
    # video_name = "output/video.avi"
    # fps = 10
    # frame_width, frame_height = 640, 480
    # video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def energy_fn(x, unit_idx=403, models=None):
        tar = F.interpolate(
            x.clone(), size=(100, 100), mode="bilinear", align_corners=False
        ).mean(1, keepdim=True)
        # normalize
        tar = tar / torch.norm(tar) * norm_constraint  # 60

        train_energy = -models["train"](tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        val_energy = -models["val"](tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        cross_val_energy = -models["cross-val"](
            tar, data_key="all_sessions", multiplex=False
        )[0, unit_idx]

        return {
            "train": train_energy,
            "val": val_energy,
            "cross-val": cross_val_energy,
        }

    model = EGG(num_steps=num_timesteps)

    energy_fn2 = None
    
    #vgg model
    if energyfunction == "VGG":
        vgg_model = vgg.create_model(create_vgg)
        energy_fn2 = partial(vgg.compare_images, model = vgg_model)

    #MSE
    if energyfunction == "MSE":
         energy_fn2=vgg.image_similarity_energy


    train_scores = []
    val_scores = []
    cross_val_scores = []
    image_gray_list = []

    for seed in seeds:
        
        for unit_idx in units:
            image = None

            frame_dir = f"frames/diffMEI_{unit_idx}_seed_{seed}"
            os.makedirs(frame_dir, exist_ok=True)
            frame_idx = 0

            for frame in number_frames:
                if frame == 0:
                    energy_fn_2 = None
                else:
                    if energyfunction != "None":
                        energy_fn_2=partial(energy_fn2, image2=image)
                    else: energy_fn_2=None
                    
                start = time.time()
                score, image = do_run(
                    model=model,
                    energy_fn=partial(energy_fn, unit_idx=unit_idx, models=models[model_type]),
                    energy_fn2=energy_fn_2,
                    desc=f"diffMEI_{unit_idx}",
                    grayscale=True,
                    seed=seed,
                    run=frame,
                )
                end = time.time()

                #plt.imshow(np.transpose(image.cpu().detach().squeeze(), (1,2,0)))
                #plt.savefig(f"output/{str(unit_idx)}_{str(frame)}.png")

                image_gray = np.mean(image.cpu().detach().squeeze().numpy(), axis=0)   # Convert tensor to numpy array

                
                image_gray_uint8 = (image_gray * 255).astype(np.uint8)
                cv2.imwrite(f"{frame_dir}/{frame_idx:05}.png", image_gray_uint8)
                frame_idx += 1
                # Convert the grayscale image to uint8 format
                #image_gray_uint8 = (image_gray * 255).astype(np.uint8)
                
                # Write the image to the video file
                #video_writer.write(image_gray)
                #image_gray_uint8 = (image_gray * 255).astype(np.uint8)
                #image_gray_list.append(image_gray_uint8)


                # Plot and save the grayscale image
                #plt.imshow(image_gray, cmap='gray')  # Use 'gray' colormap for grayscale
                #plt.savefig(f"output/unit={str(unit_idx)}_seed={str(seed)}_frame={str(frame)}.png")
                #plt.close()


                train_scores.append(score["train"].item())
                val_scores.append(score["val"].item())
                cross_val_scores.append(score["cross-val"].item())



        folder_path = frame_dir
        # Output video path and filename
        output_path = f"{frame_dir}/{unit_idx}.avi"

        # Frame rate of the output video
        fps = 20

        # Get the list of image files in the folder
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        # Load the first image to get the frame size
        first_image_path = os.path.join(folder_path, image_files[0])
        first_image = cv2.imread(first_image_path)
        print(type(first_image))
        frame_height, frame_width, _ = first_image.shape

        # Initialize the video writer
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))

        # Write each image to the video writer
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)

        # Release the video writer and close the video file
        video_writer.release()

        print("Video created successfully.")




    print("Train:", train_scores)
    print("Val:", val_scores)
    print("Cross-val:", cross_val_scores)

