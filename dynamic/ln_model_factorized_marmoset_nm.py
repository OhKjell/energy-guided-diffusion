import argparse
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import colors
from nnfabrik import builder
from torch import nn, optim
from tqdm import tqdm

import wandb
from datasets.stas import (create_cell_file_from_config_version, get_cell_sta,
                           get_cell_svd, separate_time_space_sta)
from models.ln_model import FactorizedModel
from training.measures import correlation, variance_of_predictions
from training.trainers import model_step, save_checkpoint, train_step, train_ln
from utils.global_functions import dataset_seed, global_config, home

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"
divnorm = colors.TwoSlopeNorm(vcenter=0.0)


def get_receptive_field_slice(file, cell_index, time_frame_index):
    receptive_fields = np.load(file)
    receptive_field = receptive_fields[cell_index, time_frame_index]
    return receptive_field


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=50, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_channels", default=25, type=int)
parser.add_argument(
    "--data_path",
    default="/user/vystrcilova/",
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--rf_size", default=20, type=int, help="")
parser.add_argument("--num_of_trials", default=10, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--l1", default=0.0, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--crop", default=0, type=int, nargs=4)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--time_chunk", default=1, type=int)
parser.add_argument("--cell_index", default="None", type=str)
parser.add_argument("--retina_index", default=1, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--gamma", default=100.0, type=float)
parser.add_argument("--stopper_patience", default=7, type=int)
parser.add_argument("--norm", default=0, type=int)
parser.add_argument("--sta", default=0, type=int)
parser.add_argument("--svd", default=0, type=int)
parser.add_argument("--do_nonlin", default=1, type=int)
parser.add_argument("--fancy_nonlin", default=0, type=int)
parser.add_argument("--wandb", default=1, type=int)
parser.add_argument("--model_seed", default=8, type=int)
parser.add_argument('--performance', default='validation', type=str)
parser.add_argument(
    "--config_file",
    default="config_s4",
    type=str,
    help="config file with dataset paths, sta stuff etc",
)
parser.add_argument(
    "--stimulus_seed",
    default=2022,
    type=int,
    help="stimulus seed from which the data was generated, only valid for natural movie stimulus",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.data_path is None:
        basepath = os.path.dirname(os.path.abspath(__file__))
    else:
        basepath = args.data_path
    with open(
        f"{basepath}/data/marmoset_data/responses/{args.config_file}.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    # crops = {'01': (20, 40, 50, 50), '02': (40, 35, 65, 65), '05': (35, 50, 65, 65)}
    # cell_numbers = {'01': 78, '02': 60, '05': 58}
    lr = args.learning_rate
    stimulus_seed = args.stimulus_seed

    stopper_patience = args.stopper_patience
    performance = args.performance
    num_of_channels = args.num_of_channels
    rf_size = args.rf_size
    num_of_trials = args.num_of_trials
    batch_size = args.batch_size
    img_w = args.image_width
    img_h = args.image_height
    data_type = config_dict["data_type"]
    time_chunk = args.time_chunk

    filter_type = "ete"
    do_sta = False
    do_svd = False
    sta = None
    svd = None
    learn_filter = True
    if args.sta == 1:
        filter_type = "sta"
        do_sta = True
        learn_filter = False

    do_svd = False
    if args.svd == 1:
        do_svd = True
        do_sta = True
        learn_filter = False
        filter_type = "svd"

    do_nonlin = True
    if args.do_nonlin == 0:
        do_nonlin = False

    fancy_nonlin = False
    if args.fancy_nonlin == 1:
        fancy_nonlin = True

    epochs = args.epochs
    if (not learn_filter) and (not fancy_nonlin):
        epochs = 1

    l1 = args.l1
    l2 = args.l2
    gamma = args.gamma
    retina_index = args.retina_index

    normalize_response = False
    if args.norm == 1:
        normalize_response = True

    wandb_log = True
    if args.wandb == 0:
        wandb_log = False

    if args.cell_index != "None":
        cells = [int(args.cell_index)]
    else:
        cells = None

    max_coordinate = None

    crop = args.crop
    subsample = args.subsample
    if args.log_dir == "None":
        log_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        log_dir = args.log_dir

    os.listdir(f"{basepath}/data")
    model_seed = args.model_seed

    neuronal_data_path = os.path.join(basepath, config_dict["response_path"])
    img_dir = os.path.join(basepath, config_dict["image_path"])
    config_dict["stas_path"] = os.path.join(basepath, config_dict["stas_path"])
    print("image path", img_dir)

    if cells is None:
        cells = [
            x
            for x in range(
                0, config_dict["cell_numbers"][str(retina_index + 1).zfill(2)]
            )
            if x not in config_dict["exclude_cells"][str(retina_index + 1).zfill(2)]
        ]

    for cell in cells:
        dataset_fn = "datasets.frame_movie_loader"
        dataset_config = dict(
            config=config_dict,
            basepath=basepath,
            img_dir_name=img_dir,
            neuronal_data_dir=neuronal_data_path,
            all_image_path=img_dir,
            batch_size=batch_size,
            seed=None,
            train_frac=0.8,
            subsample=subsample,
            crop=0,
            num_of_trials_to_use=num_of_trials,
            num_of_frames=num_of_channels,
            cell_index=cell,
            retina_index=retina_index,
            device=device,
            time_chunk_size=time_chunk,
            num_of_layers=1,
            cell_indices_out_of_range=True,
            oracle_correlation_threshold=None,
            normalize_responses=False,
            full_img_h=300,
            full_img_w=350,
            padding=50,
            retina_specific_crops=False,
            stimulus_seed=stimulus_seed,
        )

        """dataloaders = builder.get_data(dataset_fn, dataset_config)
        print(dataloaders)

        first_session_ID = list((dataloaders["train"].keys()))[0]
        print(first_session_ID)
        a_dataloader = dataloaders["train"][first_session_ID]

        img_h = int(
            (img_h - a_dataloader.dataset.crop[0] - a_dataloader.dataset.crop[1])
            / subsample
        )
        img_w = int(
            (img_w - a_dataloader.dataset.crop[2] - a_dataloader.dataset.crop[3])
            / subsample
        )
        inputs, targets = next(iter(a_dataloader))"""

        model_dir = f"models/ln_models/ln_models_factorized_4_{filter_type}_nm/{data_type}/retina{retina_index + 1}/cell_{cell}/lr_{lr}_whole_rf_{rf_size}_ch_{num_of_channels}_l1_{l1}_l2_{l2}_g_{gamma}_bs_{batch_size}_tr_{num_of_trials}_s_{subsample}_c_{crop}_n_{args.norm}_fn_{args.fancy_nonlin}_do_n_{args.do_nonlin}/"
        if rf_size is not None or sta:
            cell_file = config_dict["sta_file"]
            cell_file = create_cell_file_from_config_version(
                original_file=cell_file, cell_index=cell, retina_index=retina_index
            )
            sta_dir = config_dict["stas_path"]
            half_rf_size = int(rf_size / 2)
            receptive_field, max_coordinate = get_cell_sta(
                cell_file=cell_file,
                data_dir=sta_dir,
                h=img_h,
                w=img_w,
                rf_size=(num_of_channels, rf_size, rf_size),
            )
            #receptive_field = a_dataloader.dataset.transform(receptive_field)
            sta = receptive_field
            if do_svd:
                svd = get_cell_svd(sta)
            else:
                sta = separate_time_space_sta(sta)
            size = sta[0].shape

        else:
            size = (img_h, img_w)

        input_shape = size
        model_config = dict(
            input_shape=input_shape,
            num_of_neurons=1,
            num_of_channels=num_of_channels,
            l1=l1,
            gamma=gamma,
            fancy_nonlin=fancy_nonlin,
            do_sta=do_sta,
            do_svd=do_svd,
            do_nonlin=do_nonlin,
            learn_filter=learn_filter,
            sta=sta,
            svd=svd,
            seed=model_seed,
        )

        model = FactorizedModel(**model_config)

        if cuda:
            model = model.to(device)
        model = model.double()

        loss_function = nn.PoissonNLLLoss
        loss_function_config = dict(
            log_input=False, reduction="sum"
        )

        optimizer = optim.Adam
        optimizer_config = dict(lr=lr, weight_decay=args.l2)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau
        scheduler_config = dict(
            mode="max",
            patience=10,
            threshold=0.0001,
            verbose=True,
        )


        config = dict(
            epochs=epochs,
            base_path=basepath,
            img_w=img_w,
            img_h=img_h,
            rf_size=rf_size,
            model_dir=model_dir,
            log_dir=log_dir,
            size=size,
            model_config=model_config,
            dataloader_config=dataset_config,
            config_file=args.config_file,
            loss_function_config = loss_function_config,
            loss_function = loss_function,
            optimizer = optimizer,
            optimizer_config = optimizer_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            regularizer_start=0

        )

        train_ln(dataset_fn=dataset_fn,
                 dataloader_config=dataset_config,
                 model_config=model_config,
                 dataset_config_dict=config_dict,
                 model_dir=model_dir,
                 device=device,
                 max_coordinate=max_coordinate,
                 img_h=img_h,
                 img_w=img_w,
                 wandb_log=wandb_log,
                 config=config,
                 filter_type=filter_type,
                 cuda=device=='cuda',
                 performance=performance,
                 nm=True)

