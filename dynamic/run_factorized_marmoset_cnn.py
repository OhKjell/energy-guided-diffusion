import argparse
import os

import torch
import yaml

from training.trainers import train_cnn
from utils.global_functions import (dataset_seed, global_config, home,
                                    model_seed)

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"
import wandb

# wandb agent retinal-circuit-modeling/retinal_circuit_modeling/06nurevn
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=9e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=1000, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_frames", default=25, type=int)
# parser.add_argument("--data_path", default='/local/eckerlab/natural_movie_marmoset/', type=str,
parser.add_argument(
    "--data_path",
    default="/usr/users/vystrcilova/retinal_circuit_modeling/",
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--layers", default=3, type=int, help="")
parser.add_argument("--num_of_trials", default=10, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--gamma", default=0, type=float)
parser.add_argument("--gamma_temp", default=0.0, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--l1", default=0.0, type=float)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--cell_index", default="all", type=str)
parser.add_argument("--retina_index", default=1, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--time_chunk", default=30, type=int)
parser.add_argument("--spatial_hidden_kernel_size", default=7, type=int)
parser.add_argument("--temporal_hidden_kernel_size", default=5, type=int)
parser.add_argument("--spatial_input_kernel_size", default=10, type=int)
parser.add_argument("--padding", default=0, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--hidden_channels", default=[2, 2, 2], nargs='+')
parser.add_argument("--readout", default="isotropic", type=str)
parser.add_argument("--readout_nonlin", default="softplus", type=str)
parser.add_argument("--final_nonlinearity", default=1, type=int)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--bias", default=1, type=int)
parser.add_argument("--readout_bias", default=0, type=int)
parser.add_argument("--init_mu_range", default=0.3, type=float)
parser.add_argument("--init_sigma", default=0.25, type=float)
parser.add_argument("--batch_norm", default=1, type=int)
parser.add_argument("--gmp", default=0, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--spatial_dilation", default=1, type=int)
parser.add_argument("--hidden_spatial_dilation", default=1, nargs='+', type=int)
parser.add_argument("--hidden_temporal_dilation", default=1, nargs='+', type=int)
parser.add_argument("--temporal_dilation", default=1, type=int)
parser.add_argument("--init_source_grid", default=1, type=int)
parser.add_argument("--directory_prefix", default="factorized")
parser.add_argument("--correlation_threshold", default=0.0, type=float)
parser.add_argument("--explainable_variance_threshold", default=0.15, type=float)
parser.add_argument("--regularizer_start", default=0, type=int)
parser.add_argument("--normalize_responses", default=0, type=int)
parser.add_argument("--wandb_project_name", default="test-project", type=str)
parser.add_argument("--model_seed", default=-1, type=int)
parser.add_argument("--dataset", default="marmoset_data", type=str)
parser.add_argument("--wandb", default=1, type=int)
parser.add_argument("--config_file", default="config_05", type=str)
parser.add_argument("--stimulus_seed", default=2022, type=int)
parser.add_argument('--increasing_dilations', default=0, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    subsample_str = "_4"
    conf_file = args.config_file
    with open(
        f"{home}/data/marmoset_data/responses/{conf_file}.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)

    # img_dir_name = f'stimuli_padded{subsample_str}'
    stimulus_seed = args.stimulus_seed
    directory_prefix = args.directory_prefix + subsample_str
    lr = args.lr
    epochs = args.epochs
    num_of_frames = args.num_of_frames
    num_of_trials = args.num_of_trials
    batch_size = args.batch_size
    batch_norm = args.batch_norm
    correlation_threshold = args.correlation_threshold
    explainable_variance_threshold = args.explainable_variance_threshold
    layers = args.layers
    img_w = args.image_width
    img_h = args.image_height
    stride = args.stride
    retina_index = args.retina_index
    increasing_dilations = args.increasing_dilations == 1
    padding = args.padding
    bias = args.bias
    spatial_dilation = args.spatial_dilation
    temporal_dilation = args.temporal_dilation
    hidden_temporal_dilation = args.hidden_temporal_dilation
    hidden_spatial_dilation = args.hidden_spatial_dilation
    print(f'hidden temporal_dilation: {hidden_temporal_dilation}')

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
            hidden_spatial_dilation = [x*(i+1) for i, x in enumerate(hidden_spatial_dilation)]
            print(f'hidden spatial dilations are: {hidden_spatial_dilation}')

    if isinstance(hidden_temporal_dilation, int):
        hidden_temporal_dilation = (hidden_temporal_dilation,) * (layers - 1)

    print(f'hidden spatial_dilation: {hidden_spatial_dilation}')
    print(f'hidden temporal_dilation: {hidden_temporal_dilation}')


    readout_bias = args.readout_bias
    core_nonlin = args.core_nonlin
    readout_nonlin = args.readout_nonlin
    stopper_patience = args.stopper_patience

    subsample = args.subsample
    l1 = args.l1
    l2 = args.l2
    gamma = args.gamma
    gamma_temp = args.gamma_temp
    input_regularizer = "LaplaceL2norm"
    readout = args.readout
    gmp = args.gmp
    normalize_responses = args.normalize_responses
    wandb_log = True
    if args.wandb == 0:
        wandb_log = False

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

    init_mu_range = args.init_mu_range
    init_sigma = args.init_sigma
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
    time_chunk = args.time_chunk

    cell_index = args.cell_index
    if (device == "cuda") and (args.data_path is None):
        basepath = f"/local/eckerlab/{args.dataset}/"
    else:
        basepath = args.data_path
    log_dir = args.log_dir
    final_nonlinearity = args.final_nonlinearity

    neuronal_data_path = os.path.join(basepath, config_dict["response_path"])
    img_dir = os.path.join(basepath, config_dict["image_path"])

    dataloader_fn = "datasets.frame_movie_loader"
    dataloader_config = dict(
        config=config_dict,
        basepath=basepath,
        img_dir_name=img_dir,
        neuronal_data_dir=neuronal_data_path,
        all_image_path=img_dir,
        batch_size=batch_size,
        seed=None,
        train_frac=0.8,
        subsample=subsample,
        temporal_dilation=temporal_dilation,
        hidden_temporal_dilation=hidden_temporal_dilation,
        crop=0,
        num_of_trials_to_use=num_of_trials,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=temporal_hidden_kernel,
        cell_index=None if cell_index == "all" else int(cell_index),
        retina_index=retina_index,
        device=device,
        time_chunk_size=time_chunk,
        num_of_layers=layers,
        cell_indices_out_of_range=True,
        explainable_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold > 0
        else None,
        oracle_correlation_threshold=None,
        normalize_responses=normalize_responses,
        full_img_h=300,
        full_img_w=350,
        padding=50,
        stimulus_seed=stimulus_seed,
        # retina_specific_crops=False
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
        "temporal_dilation": temporal_dilation,
        "spatial_dilation": spatial_dilation,
        "hidden_spatial_dilation": hidden_spatial_dilation,
        'hidden_temporal_dilation': hidden_temporal_dilation,
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
        "final_nonlinearity": final_nonlinearity,
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
        dataloader_fn=dataloader_fn,
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
            project="factorized-marmoset-2-cnn",
            entity="retinal-circuit-modeling",
        )
    train_cnn(
        **config,
    )
