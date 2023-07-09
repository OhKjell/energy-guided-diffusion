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

# experiment settings
num_timesteps = 100
energy_scale = 5  # 20
energy_scale2 = 5
seeds = [0]
unit_seed=5
norm_constraint = 25  # 25
model_type = "task_driven"  #'task_driven' #or 'v4_multihead_attention'
energyfunction = "MSE" #VGG
number_units = 1


def do_run(model, energy_fn, energy_fn2, desc="progress", grayscale=False, seed=None, run=1):
    if seed is not None:
        torch.manual_seed(seed)

    cur_t = num_timesteps - 1

    samples = model.sample_frame(
        energy_fn=energy_fn,
        energy_fn2= energy_fn2,
        energy_scale=energy_scale,
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
    data_driven_corrs = np.load("./data/data_driven_corr.npy")
    units = np.load("./data/pretrained_resnet_unit_correlations.npy")
    available_units = (data_driven_corrs > 0.5) * (units > 0.5)

    np.random.seed(unit_seed)
    units = np.random.choice(np.arange(len(available_units))[available_units], number_units)

    # wandb.init(project="egg", entity="sinzlab", name=f"diffmeis_{time.time()}")
    # wandb.config.update(
    #     dict(
    #         energy_scale=energy_scale,
    #         norm_constraint=norm_constraint,
    #         model_type=model_type,
    #         units=units,
    #     )
    # )

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
        vgg_model = vgg.create_model()
        energy_fn2 = partial(vgg.compare_images, model = vgg_model)

    #MSE
    if energyfunction == "MSE":
         energy_fn2=vgg.image_similarity_energy


    train_scores = []
    val_scores = []
    cross_val_scores = []
    image = None

    for seed in seeds:
        
        for unit_idx in units:
            image = None
            for frame in range(5):
                start = time.time()
                score, image = do_run(
                    model=model,
                    energy_fn=partial(energy_fn, unit_idx=unit_idx, models=models[model_type]),
                    energy_fn2=partial(energy_fn2, image2=image,),
                    desc=f"diffMEI_{unit_idx}",
                    grayscale=True,
                    seed=seed,
                    run=frame,
                )
                end = time.time()

                #plt.imshow(np.transpose(image.cpu().detach().squeeze(), (1,2,0)))
                #plt.savefig(f"output/{str(unit_idx)}_{str(frame)}.png")

                image_gray = np.mean(image.cpu().detach().squeeze().numpy(), axis=0)   # Convert tensor to numpy array

                # Plot and save the grayscale image
                plt.imshow(image_gray, cmap='gray')  # Use 'gray' colormap for grayscale
                plt.savefig(f"output/{str(unit_idx)}_{str(frame)}.png")
                plt.close()


                train_scores.append(score["train"].item())
                val_scores.append(score["val"].item())
                cross_val_scores.append(score["cross-val"].item())

    print("Train:", train_scores)
    print("Val:", val_scores)
    print("Cross-val:", cross_val_scores)
