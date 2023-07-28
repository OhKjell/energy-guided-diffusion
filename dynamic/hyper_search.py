import argparse

import torch

from training.trainers import train_cnn
from utils.global_functions import home

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--basepath",
    default=None,
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--retina_index", default=0, type=int)
parser.add_argument(
    "--epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_frames", default=15, type=int)
parser.add_argument("--layers", default=1, type=int, help="")
parser.add_argument("--num_of_trials", default=250, type=int)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--time_chunk", default=10, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--readout", default="isotropic", type=str)
parser.add_argument("--readout_nonlin", default="softplus", type=str)
parser.add_argument("--explainable_variance_threshold", default=0.15, type=float)
# parser.add_argument('--search_index', default=0, type=int)

parser.add_argument("--total_trials", default=100, type=int)
parser.add_argument("--arms_per_trial", default=1, type=int)


def train_evaluate(auto_params):
    corr = train_cnn(**auto_params, **fixed_parms)
    return corr


def run_bayesian_optimization(auto_params, total_trials, arms_per_trial):
    evaluation_function = train_evaluate
    objective_name = "validation_correlation"
    best_parameters, values, experiment, model = optimize(
        parameters=auto_params,
        evaluation_function=evaluation_function,
        objective_name=objective_name,
        total_trials=total_trials,
        arms_per_trial=arms_per_trial,
    )
    return best_parameters, values, experiment, model


if __name__ == "__main__":
    args = parser.parse_args()
    auto_params = [
        {"name": "gamma", "type": "range", "bounds": [1e-3, 5e2], "log_scale": True},
        {
            "name": "gamma_temp",
            "type": "range",
            "bounds": [1e-3, 1e2],
            "log_scale": True,
        },
        {
            "name": "input_kernel_size",
            "type": "range",
            "value_type": "int",
            "bounds": [7, 21],
            "log_scale": False,
        },
        {"name": "lr", "type": "range", "bounds": [1e-4, 3e-2], "log_scale": True},
        {"name": "l1", "type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
        {
            "name": "hidden_channels",
            "type": "choice",
            "values": ["8", "16"],
            "log_scale": True,
        },
    ]
    search_index = args.search_index
    fixed_parms = dict(
        epochs=args.epochs,
        num_of_trials=args.num_of_trials,
        layers=args.layers,
        time_chunk=args.time_chunk,
        num_of_frames=args.num_of_frames,
        batch_size=args.batch_size,
        device=device,
        retina_index=args.retina_index,
    )
    search_index = {search_index: auto_params}
    best_parameters, values, experiments, model = run_bayesian_optimization(
        auto_params, total_trials=args.total_trials, arms_per_trial=args.arms_per_trial
    )
    for k, v in best_parameters.items():
        print(f"{k}: {v}")
