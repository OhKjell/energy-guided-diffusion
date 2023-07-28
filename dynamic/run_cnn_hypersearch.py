import argparse

import torch

from training.trainers import train_cnn

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"
import wandb

# wandb agent retinal-circuit-modeling/retinal_circuit_modeling/06nurevn
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_frames", default=3, type=int)
parser.add_argument(
    "--data_path",
    default=None,
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--layers", default=1, type=int, help="")
parser.add_argument("--num_of_trials", default=150, type=int)
parser.add_argument("--batch_size", default=12, type=int)
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
parser.add_argument("--time_chunk", default=10, type=int)
parser.add_argument("--hidden_kernel_size", default=3, type=int)
parser.add_argument("--input_kernel_size", default=15, type=int)
parser.add_argument("--padding", default=0, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--hidden_channels", default="3", type=str)
parser.add_argument("--readout", default="isotropic", type=str)
parser.add_argument("--readout_nonlin", default="softplus", type=str)
parser.add_argument("--final_nonlinearity", default=1, type=int)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--bias", default=1, type=int)
parser.add_argument("--readout_bias", default=0, type=int)
parser.add_argument("--init_mu_range", default=0.3, type=float)
parser.add_argument("--init_sigma", default=0.35, type=float)
parser.add_argument("--batch_norm", default=1, type=int)
parser.add_argument("--gmp", default=0, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--directory_prefix", default="basic")
parser.add_argument("--correlation_threshold", default=0.0, type=float)
parser.add_argument("--explainable_variance_threshold", default=0.15, type=float)
parser.add_argument("--regularizer_start", default=0, type=int)
parser.add_argument("--normalize_responses", default=0, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    directory_prefix = args.directory_prefix
    lr = args.lr
    epochs = args.epochs
    num_of_frames = args.num_of_frames
    time_chunk = args.time_chunk
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
    padding = args.padding
    bias = args.bias
    readout_bias = args.readout_bias
    core_nonlin = args.core_nonlin
    readout_nonlin = args.readout_nonlin

    subsample = args.subsample
    l1 = args.l1
    l2 = args.l2
    gamma = args.gamma
    gamma_temp = args.gamma_temp
    input_regularizer = "LaplaceL2norm"
    readout = args.readout
    gmp = args.gmp
    normalize_responses = args.normalize_responses

    init_mu_range = args.init_mu_range
    init_sigma = args.init_sigma
    regularizer_start = args.regularizer_start
    input_kernel_size = args.input_kernel_size
    hidden_kernel_size = args.hidden_kernel_size
    hidden_channels = args.hidden_channels
    stopper_patience = args.stopper_patience
    cell_index = args.cell_index

    basepath = "/local/eckerlab/white_noise_salamander/"
    log_dir = args.log_dir
    final_nonlinearity = args.final_nonlinearity

    config = dict(
        directory_prefix=directory_prefix,
        lr=lr,
        epochs=epochs,
        num_of_frames=num_of_frames,
        num_of_trials=num_of_trials,
        batch_size=batch_size,
        correlation_threshold=correlation_threshold,
        explainable_variance_threshold=explainable_variance_threshold,
        layers=layers,
        img_h=img_h,
        img_w=img_w,
        retina_index=retina_index,
        crop=None,
        padding_int=padding,
        bias_int=bias,
        readout_bias_int=readout_bias,
        batch_norm_int=batch_norm,
        gmp=gmp,
        normalize_responses_int=normalize_responses,
        subsample=subsample,
        l1=l1,
        l2=l2,
        gamma=gamma,
        gamma_temp=gamma_temp,
        readout=readout,
        input_regularizer=input_regularizer,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma,
        regularizer_start=regularizer_start,
        hidden_channels=hidden_channels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        stopper_patience=stopper_patience,
        cell_index=cell_index,
        basepath=basepath,
        log_dir=log_dir,
        conv3d=True,
        overlapping=False,
        time_chunk=time_chunk,
        core_nonlin=core_nonlin,
        final_nonlinearity=True if final_nonlinearity == 1 else False,
        stride=stride,
        readout_nonlin=readout_nonlin,
        device=device,
        search_index={"x": []},
    )

    # wandb.init(config=config, project="sweep-3-layer", entity="retinal-circuit-modeling")
    train_cnn(**config, hyper_search=True)
