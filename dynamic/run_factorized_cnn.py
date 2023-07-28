import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from nnfabrik import builder
from torch import nn, optim
from tqdm import tqdm

import wandb
from training.measures import correlation, variance_of_predictions
from training.trainers import (model_step, save_checkpoint, train, train_cnn,
                               train_step)
from utils.global_functions import (dataset_seed, global_config, home,
                                    model_seed)

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=8e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_frames", default=10, type=int)
# parser.add_argument("--data_path", default='/local/eckerlab/white_noise_salamander_gollisch/', type=str,
parser.add_argument(
    "--data_path",
    default="/user/vystrcilova/",
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--layers", default=3, type=int, help="")
parser.add_argument("--num_of_trials", default=250, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--gamma", default=0.33, type=float)
parser.add_argument("--gamma_temp", default=0.33, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--l1", default=1, type=float)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--cell_index", default="all", type=str)
parser.add_argument("--retina_index", default=0, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--time_chunk", default=20, type=int)
parser.add_argument("--trial_portion", default=0.1, type=float)
parser.add_argument("--spatial_hidden_kernel_size", default=7, type=int)
parser.add_argument("--temporal_hidden_kernel_size", default=5, type=int)
parser.add_argument("--spatial_input_kernel_size", default=19, type=int)
parser.add_argument("--padding", default=0, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--x_shift", default=0.0, type=float)
parser.add_argument("--y_shift", default=0.0, type=float)
parser.add_argument("--hidden_channels", default=[2, 2, 2], nargs='+')
parser.add_argument("--readout", default="isotropic", type=str)
parser.add_argument("--readout_nonlin", default="softplus", type=str)
parser.add_argument("--final_nonlinearity", default=1, type=int)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--bias", default=1, type=int)
parser.add_argument("--readout_bias", default=0, type=int)
parser.add_argument("--init_mu_range", default=0.3, type=float)
parser.add_argument("--init_sigma", default=0.15, type=float)
parser.add_argument("--batch_norm", default=1, type=int)
parser.add_argument("--gmp", default=0, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--spatial_dilation", default=1, type=int)
parser.add_argument("--temporal_dilation", default=1, type=int)
parser.add_argument("--hidden_spatial_dilation", default=1, nargs='+', type=int)
parser.add_argument("--hidden_temporal_dilation", default=1, nargs='+', type=int)
parser.add_argument("--init_source_grid", default=1, type=int)
parser.add_argument("--regularizer_start", default=0, type=int)
parser.add_argument("--directory_prefix", default="factorized")
parser.add_argument("--correlation_threshold", default=0.0, type=float)
parser.add_argument("--explainable_variance_threshold", default=0.0, type=float)
parser.add_argument("--normalize_responses", default=0, type=int)
parser.add_argument("--model_seed", default=-1, type=int)
parser.add_argument("--dataset", default="marmoset_data", type=str)
parser.add_argument("--wandb", default=0, type=int)
parser.add_argument("--config_file", default="config_05", type=str)
parser.add_argument('--increasing_dilations', default=0, type=int)



if __name__ == "__main__":
    args = parser.parse_args()
    # with open(f'{home}/data/salamander_data/responses/config.yaml', 'rb') as config_file:
    with open(
        f"{home}/data/{args.dataset}/responses/{args.config_file}.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    lr = args.lr
    epochs = args.epochs
    wandb_log = args.wandb == 1
    num_of_frames = args.num_of_frames
    num_of_trials = args.num_of_trials
    batch_size = args.batch_size
    layers = args.layers
    img_w = args.image_width
    img_h = args.image_height
    retina_index = args.retina_index
    increasing_dilations = args.increasing_dilations == 1
    correlation_threshold = args.correlation_threshold
    explainable_variance_threshold = args.explainable_variance_threshold
    x_shift = args.x_shift
    y_shift = args.y_shift
    crop = global_config["big_crops"][str(retina_index + 1).zfill(2)]
    padding = args.padding
    bias = args.bias
    readout_bias = args.readout_bias
    core_nonlin = args.core_nonlin
    readout_nonlin = args.readout_nonlin
    if (device == "cuda") and (args.data_path is None):
        basepath = f"/local/eckerlab/{args.dataset}/"
    else:
        basepath = args.data_path
        print("basepath:", {basepath})
    log_dir = args.log_dir
    final_nonlinearity = args.final_nonlinearity

    neuronal_data_path = os.path.join(basepath, config_dict["neuronal_data_path"])
    training_img_dir = os.path.join(basepath, config_dict["training_img_dir"])
    test_img_dir = os.path.join(basepath, config_dict["test_img_dir"])

    batch_norm = args.batch_norm
    if retina_index not in range(0, 5):
        retina_index = None
        print("loading data for all retinas")
    cell_index = args.cell_index
    subsample = args.subsample
    m_seed = args.model_seed
    if m_seed == -1:
        m_seed = model_seed

    initialize_source_grid = True
    if args.init_source_grid == 0:
        print(
            "Attention! Not initializing source grid, probably no STA data available. If you think STA data should "
            "be available, set to True"
        )
        initialize_source_grid = False

    # regularization parameters
    l1 = args.l1
    l2 = args.l2
    gamma = args.gamma
    gamma_temp = args.gamma_temp
    input_regularizer = "LaplaceL2norm"

    # full gaussian readout parameters
    readout = args.readout
    init_mu_range = args.init_mu_range
    init_sigma = args.init_sigma
    gmp = args.gmp

    spatial_input_kernel = (
        args.spatial_input_kernel_size,
        args.spatial_input_kernel_size,
    )
    temporal_input_kernel = num_of_frames
    spatial_hidden_kernel = (
        args.spatial_hidden_kernel_size,
        args.spatial_hidden_kernel_size,
    )
    temporal_hidden_kernel = args.temporal_hidden_kernel_size
    stride = args.stride
    spatial_dilation = args.spatial_dilation
    temporal_dilation = args.temporal_dilation
    hidden_temporal_dilation = args.hidden_temporal_dilation
    hidden_spatial_dilation = args.hidden_spatial_dilation

    if isinstance(hidden_spatial_dilation, (list, tuple)):
        if len(hidden_spatial_dilation) == 1:
            hidden_spatial_dilation = int(hidden_spatial_dilation[0])

    if isinstance(hidden_temporal_dilation, (list, tuple)):
        if len(hidden_temporal_dilation) == 1:
            hidden_temporal_dilation = int(hidden_temporal_dilation[0])
    print(f'hidden spatial_dilation: {hidden_spatial_dilation}')

    if isinstance(hidden_spatial_dilation, int):
        hidden_spatial_dilation = (hidden_spatial_dilation,) * (layers - 1)
        if increasing_dilations:
            hidden_spatial_dilation = [x * (i + 1) for i, x in enumerate(hidden_spatial_dilation)]
            print(f'hidden spatial dilations are: {hidden_spatial_dilation}')

    if isinstance(hidden_temporal_dilation, int):
        hidden_temporal_dilation = (hidden_temporal_dilation,) * (layers - 1)

    time_chunk = args.time_chunk

    print(f'hidden spatial_dilation: {hidden_spatial_dilation}')
    print(f'hidden temporal_dilation: {hidden_temporal_dilation}')

    directory_prefix = args.directory_prefix
    directory_prefix = args.directory_prefix
    normalize_responses = args.normalize_responses

    stopper_patience = args.stopper_patience
    regularizer_start = args.regularizer_start

    hidden_channels = args.hidden_channels[0]
    if isinstance(hidden_channels, str):
        hidden_channels = [int(x) for x in hidden_channels.split(' ')]
    if isinstance(hidden_channels, (list, tuple)):
        hidden_channels = [int(x) for x in hidden_channels]
        if len(hidden_channels) > layers:
            hidden_channels = hidden_channels[:layers]
        elif len(hidden_channels) < layers:
            hidden_channels = [x for x in hidden_channels] + [hidden_channels[-1]]

    if args.cell_index != "all":
        cell = int(args.cell_index)
    else:
        cell = None

    if args.data_path is None:
        basepath = os.path.dirname(os.path.abspath(__file__))
    else:
        basepath = args.data_path

    if args.log_dir == "None":
        log_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        log_dir = args.log_dir

    data_type = config_dict["data_type"]

    os.listdir(f"{basepath}/data")

    dataloader_fn = "datasets.white_noise_loader"
    dataloader_config = dict(
        neuronal_data_dir=neuronal_data_path,
        train_image_path=training_img_dir,
        test_image_path=test_img_dir,
        batch_size=batch_size,
        crop=None,
        temporal_dilation=temporal_dilation,
        hidden_temporal_dilation=hidden_temporal_dilation,
        subsample=subsample,
        seed=dataset_seed,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=False,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=temporal_hidden_kernel,
        cell_index=None if cell_index == "all" else int(cell_index),
        retina_index=retina_index,
        conv3d=True,
        time_chunk_size=time_chunk,
        overlapping=False,
        num_of_layers=1 if "input_temporal" in directory_prefix else layers,
        explainable_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold > 0
        else None,
        oracle_correlation_threshold=correlation_threshold
        if correlation_threshold > 0
        else None,
        normalize_responses=normalize_responses == 1,
        config=config_dict,
    )

    model_fn = "models.FactorizedEncoder.build_initial"
    model_config = {
        "base_path": basepath,
        "hidden_channels": hidden_channels,
        "spatial_input_kern": spatial_input_kernel,
        "temporal_input_kern": temporal_input_kernel,
        "spatial_hidden_kern": spatial_hidden_kernel,
        "temporal_hidden_kern": temporal_hidden_kernel,
        "core_nonlinearity": core_nonlin,
        "stride": stride,
        "spatial_dilation": spatial_dilation,
        "temporal_dilation": temporal_dilation,
        "hidden_spatial_dilation": hidden_spatial_dilation,
        "hidden_temporal_dilation": hidden_temporal_dilation,
        "bias": bias == 1,
        "independent_bn_bias": True,
        "laplace_padding": None,
        "input_regularizer": input_regularizer,
        "layers": layers,
        "gamma_input": gamma,
        "gamma_temporal": gamma_temp,
        "l1": l1,
        "l2": l2,
        "subsample": subsample,
        "elu_xshift": 0.0,
        "elu_yshift": 0.0,
        "padding": padding == 1,
        "batch_norm": batch_norm == 1,
        "readout_type": readout,
        "final_nonlinearity": final_nonlinearity == 1,
        "readout_nonlinearity": readout_nonlin,
        "init_mu_range": init_mu_range,
        "init_sigma": init_sigma,
        "readout_bias": readout_bias == 1,
        "retina_index": retina_index,
        "data_dir": basepath,
        "use_grid_mean_predictor": gmp,
        "initialize_source_grid": initialize_source_grid,
        "explainable_variance_threshold": explainable_variance_threshold
        if explainable_variance_threshold > 0
        else None,
        "oracle_correlation_threshold": correlation_threshold
        if correlation_threshold > 0
        else None,
        "config": config_dict,
        "img_h": img_h,
        "img_w": img_w,
        "cell_index": None if cell_index == "all" else cell_index,
    }
    config = dict(
        dataloader_config=dataloader_config,
        model_config=model_config,
        directory_prefix=directory_prefix,
        lr=lr,
        epochs=epochs,
        model_fn=model_fn,
        gmp=gmp,
        readout=readout,
        regularizer_start=regularizer_start,
        stopper_patience=stopper_patience,
        basepath=basepath,
        log_dir=log_dir,
        device=device,
        seed=m_seed,
        wandb_log=wandb_log,
        config=config_dict,
    )
    if wandb_log:
        wandb.init(
            config=config,
            project=f"factorized-{data_type}-cnn",
            entity="retinal-circuit-modeling",
            dir=log_dir,
        )
    train_cnn(**config)
