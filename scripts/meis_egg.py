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
energy_scale2 = 0
seeds = [0]#np.arange(1)
unit_seed=27#42
norm_constraint = 25  # 25
model_type = "task_driven"  #'task_driven' #or 'v4_multihead_attention'
energyfunction = "MSE" #"MSE" "VGG" "None"
number_units = 1
number_frames = np.arange(1)
create_vgg = True
fps = 20
unit_ids = None #None [id]
#for vgg
vgg_gray=True
escale2 = [0]

def do_run(model, energy_fn, energy_fn2, desc="progress", grayscale=False, seed=None, run=1):
    #move out
    

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
    
    if os.path.exists("output/frames"):
        shutil.rmtree("output/frames")
    os.makedirs("output/frames")

    if os.path.exists("output/video"):
        shutil.rmtree("output/video")
    os.makedirs("output/video")

    data_driven_corrs = np.load("./data/data_driven_corr.npy")
    units = np.load("./data/pretrained_resnet_unit_correlations.npy")
    available_units = (data_driven_corrs > 0.5) * (units > 0.5)

    np.random.seed(unit_seed)
    units = np.random.choice(np.arange(len(available_units))[available_units], number_units)
    
    
    if unit_ids != None:
        units = unit_ids


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
         energy_fn2= partial(vgg.image_similarity_energy, grayscale = vgg_gray)


    train_scores = []
    val_scores = []
    cross_val_scores = []
    image_gray_list = []

    image = None

    lambdas = []
    energies = []

    for model_idx in range(1):

        if model_idx == 0:
            model_dir = f"output/MSE"
        else:
            model_dir = f"output/VGG"
        
        os.makedirs(model_dir, exist_ok=True)

        for seed in seeds:
            
            
            for unit_idx in units:

                unit_dir = f"{model_dir}/diffMEI_{unit_idx}_seed_{seed}_gray"
                os.makedirs(unit_dir, exist_ok=True)

                for energy_scale2 in escale2:

                    energy_dir = f"{unit_dir}/energy_scale_{energy_scale2}"
                    os.makedirs(energy_dir, exist_ok=True)

                    frame_dir = f"{energy_dir}/frames_gray"
                    os.makedirs(frame_dir, exist_ok=True)

                    frame_dir_color = f"{energy_dir}/frames_color"
                    os.makedirs(frame_dir_color, exist_ok=True)

                    

                    video_dir = f"{energy_dir}/videos"
                    os.makedirs(video_dir, exist_ok=True)

                    frame_idx = 0

                    if seed is not None:
                        torch.manual_seed(seed)

                    #FRAME LOOP
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
                        lambdas.append(energy_scale)
                        print(type(image))
                        energies.append(energy_fn(image)["train"])

                        #SAVE IMAGES

                        plt.imshow(np.transpose(image.cpu().detach().squeeze(), (1,2,0)))
                        plt.savefig(f"{frame_dir_color}/{frame_idx:05}.png")
                        plt.close()
                        
                        # Plot and save the grayscale image
                        image_gray = np.mean(image.cpu().detach().squeeze().numpy(), axis=0)   # Convert tensor to numpy array
                        plt.imshow(image_gray, cmap='gray')  # Use 'gray' colormap for grayscale
                        plt.axis('off')
                        plt.savefig(f"{frame_dir}/{frame_idx:05}.png")
                        plt.close()
                        frame_idx += 1

                        train_scores.append(score["train"].item())
                        val_scores.append(score["val"].item())
                        cross_val_scores.append(score["cross-val"].item())



                    #MAKE VIDEOS

                    folder_path = frame_dir
                    # Output video path and filename
                    output_path = f"{video_dir}/{unit_idx}_gray.avi"

                    # Frame rate of the output video

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


                    folder_path = frame_dir_color
                    # Output video path and filename
                    output_path = f"{video_dir}/{unit_idx}_color.avi"

                    # Frame rate of the output video

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


        #SET ENERGy FUNCTION TO VGG

        print("Train:", train_scores)
        print("Val:", val_scores)
        print("Cross-val:", cross_val_scores)
        vgg_model = vgg.create_model(create_vgg)
        energy_fn2 = partial(vgg.compare_images, model = vgg_model)

    plt.plot(lambdas, energies)
    plt.savefig(f"{model_dir}/plot.png")
    plt.close()

