import argparse
import os

import numpy as np
import torch

import models
import wandb
from meis.helper_functions import get_seed_model_versions
from meis.postprocessing import ChangeStdAndClip, PNormConstraintAndClip
from meis.tracking import GradientObjective, LoggingTracker
from meis.visualizer import get_logged_array, save_mei_video
from models.helper_functions import get_model_and_dataloader
from utils.global_functions import cuda, get_cell_names, home, mei_seed


def optimize(
    model,
    input_shape,
    optimizer,
    log_dir,
    lr=0.1,
    cell_index=0,
    postprocessing=None,
    init_range=0.01,
):
    mei = torch.FloatTensor(*input_shape).uniform_(-init_range, init_range)
    vmin = -1
    vmax = 1
    model.double()
    if cuda:
        mei = mei.to("cuda").double()
    mei.requires_grad_(True)
    optimizer = optimizer(params=[mei], lr=lr)

    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    prev_activation = torch.tensor([-1])
    for i in range(5000):
        optimizer.zero_grad()
        activation = model(mei)[:, cell_index]
        print(activation.shape)
        wandb.log({"activation": activation.item()}, i)
        if postprocessing is None:
            vmin = None
            vmax = None
        save_mei_video(
            i,
            current_state=np.array(mei.clone().detach().cpu()),
            log_dir=log_dir,
            colormap="gray",
            vmin=vmin,
            vmax=vmax,
            prefix="mei",
        )
        #
        #
        wandb.log(
            {
                "video": wandb.Video(
                    os.path.join(log_dir, "videos", f"mei_e{i}.mp4"), format="mp4"
                )
            },
            step=i,
        )

        print(f"activation at {i}: {activation.item()}")
        wandb.log({"max_mei_value": torch.max(mei.clone().detach())}, step=i)
        print(f"max mei value: {torch.max(mei.clone().detach())}")
        (-activation).backward()
        save_mei_video(
            i,
            current_state=np.array(mei.grad.clone().detach().cpu()),
            log_dir=log_dir,
            colormap="gray",
            vmin=None,
            vmax=None,
            prefix="raw_grad",
        )
        wandb.log(
            {
                "raw_grad": wandb.Video(
                    os.path.join(log_dir, "videos", f"raw_grad_e{i}.mp4"), format="mp4"
                )
            },
            step=i,
        )
        wandb.log({"max_raw_grad_value": torch.max(mei.grad.clone().detach())}, step=i)
        print(f"max raw grad value: {torch.max(mei.grad)}")

        optimizer.step()
        if postprocessing is not None:
            mei.data = postprocessing(mei.data, i)
        if (
            np.abs(prev_activation.item() - activation.item())
            < (prev_activation * 0.001)
        ) and (i > 1000):
            break
        print(
            f"epoch {i} difference: {np.abs(prev_activation.item() - activation.item())}",
            f"threshold: {prev_activation.item()*0.001}",
        )
        prev_activation = activation


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=10.0, type=float, help="learning rate")
parser.add_argument("--p", default=2, type=int)
parser.add_argument("--norm_value", default=10, type=float, help="learning rate")
parser.add_argument("--std", default=1.0, type=float, help="learning rate")
parser.add_argument("--postprocessing", default="pnorm_clip", type=str)
parser.add_argument("--init_range", default=0.01)
parser.add_argument("--optimizer", default="SGD", type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    cell_index = 58
    retina_index = 0
    epoch = 5000
    suppress = False
    sigma = None
    sigma_temp = None
    lr = args.lr
    seeds = [8]
    pprocessing = args.postprocessing
    optimizer = args.optimizer
    init_range = args.init_range
    if pprocessing == "stdclip":
        postprocessing = ChangeStdAndClip(args.std)
    elif pprocessing == "pnorm_clip":
        postprocessing = PNormConstraintAndClip(norm_value=args.norm_value, p=args.p)
    else:
        raise ValueError("Postprocessing code invalid")
    precondition = None
    if optimizer == "SGD":
        optimizer = torch.optim.SGD
    else:
        optimizer = torch.optim.Adam
    # [8], [128],
    #      [512], [64], [256], [8, 128], [1024], [2048],
    # [8, 128, 512],
    #             [1024, 2048, 8], [256, 64, 1024, 2048], [512, 1024, 8, 2048]]
    directory = f"{home}/models/basic_ev_0.15_cnn/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    mei_dir = f"{home}/meis/data/retina{retina_index + 1}"
    filename = "lr_0.0100_l_1_ch_16_t_15_bs_16_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1"
    # mei = get_logged_array(os.path.join(mei_dir, filename, f'cell_{cell_index}'), epoch=epoch)
    cell_names = get_cell_names(
        retina_index=retina_index,
        correlation_threshold=0,
        explained_variance_threshold=0.15,
    )
    for seed in seeds:
        if isinstance(seed, list) or (seed is None):
            model, seed = get_seed_model_versions(
                model_name=filename,
                model_dir=directory,
                model_fn="models.BasicEncoder.build_trained",
                device="cuda",
                seeds=seed,
            )
        else:
            _, model, _ = get_model_and_dataloader(
                directory=directory,
                filename=filename,
                model_fn="models.BasicEncoder.build_trained",
                device="cuda",
                seed=seed,
            )
        for cell in range(len(cell_names)):
            cell_name = cell_names[cell]
            wandb.init(
                config={
                    "model_name": model.config_dict["model_name"],
                    "cell": cell_name,
                    "cell_index": cell,
                    "lr": lr,
                    "std": args.std,
                    "p": args.p,
                    "norm_value": args.norm_value,
                    "seed": seed,
                    "optimizer": optimizer,
                    "mei_seed": mei_seed,
                    "manual": True,
                    "postprocessing": postprocessing,
                    "arg_pp": args.postprocessing,
                    "precondition": precondition,
                    "init_range": init_range,
                },
                project="eMEI-1layer-contrast-and-clip-all-cells",
                entity="retinal-circuit-modeling",
            )

            optimize(
                model=model,
                optimizer=optimizer,
                lr=lr,
                input_shape=(
                    1,
                    1,
                    1
                    + model.config_dict["layers"]
                    * (model.config_dict["num_of_frames"] - 1),
                    model.config_dict["img_h"],
                    model.config_dict["img_w"],
                ),
                log_dir=os.path.join(
                    home,
                    "meis",
                    "data",
                    f"retina{model.config_dict['retina_index'] + 1}",
                    model.config_dict["model_name"],
                    f"cell_{cell_name}",
                ),
                cell_index=cell,
                postprocessing=postprocessing,
                init_range=init_range,
            )
            wandb.finish()
