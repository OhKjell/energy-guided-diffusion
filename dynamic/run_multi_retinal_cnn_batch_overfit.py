import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from neuralpredictors.training import LongCycler
from nnfabrik import builder
from torch import nn, optim
from tqdm import tqdm

from training.measures import correlation, variance_of_predictions
from training.trainers import model_step, save_checkpoint, train_step
from utils.global_functions import (big_crops, cell_numbers, crops,
                                    dataset_seed, model_seed)

cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_frames", default=8, type=int)
parser.add_argument(
    "--data_path",
    default=None,
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--layers", default=2, type=int, help="")
parser.add_argument("--num_of_trials", default=250, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--gamma", default=0.33, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--l1", default=1, type=float)
parser.add_argument("--crop", default=50, type=int, nargs=4)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--cell_index", default="all", type=str)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--trial_portion", default=0.1, type=float)
parser.add_argument("--hidden_kernel_size", default=3, type=int)
parser.add_argument("--input_kernel_size", default=12, type=int)
parser.add_argument("--padding", default=1, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--hidden_channels", default=1, type=int)
parser.add_argument("--readout", default="full_gaussian", type=str)
parser.add_argument("--readout_nonlin", default="softplus", type=str)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--bias", default=1, type=int)
parser.add_argument("--readout_bias", default=1, type=int)
parser.add_argument("--init_mu_range", default=0.3, type=float)
parser.add_argument("--init_sigma", default=0.35, type=float)
parser.add_argument("--batch_norm", default=1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()

    lr = args.learning_rate
    epochs = args.epochs
    num_of_frames = args.num_of_frames
    num_of_trials = args.num_of_trials
    batch_size = args.batch_size
    layers = args.layers
    img_w = args.image_width
    img_h = args.image_height
    retina_index = None
    # crop = crops[str(retina_index+1).zfill(2)]
    crop = "big_crop"
    if args.padding == 0:
        padding = False
    else:
        padding = True

    if args.bias == 0:
        bias = False
    else:
        bias = True

    readout_bias = True
    if args.readout_bias == 0:
        readout_bias = False

    batch_norm = True
    if args.batch_norm == 0:
        batch_norm = False

    retina_index = None
    # crop = 50, 75, 110, 60
    # crop = 0
    subsample = args.subsample
    l1 = args.l1
    gamma = args.gamma
    init_mu_range = args.init_mu_range
    init_sigma = args.init_sigma
    hidden_channels = args.hidden_channels
    input_kernel = (num_of_frames, args.input_kernel_size, args.input_kernel_size)
    hidden_kernel = (num_of_frames, args.hidden_kernel_size, args.hidden_kernel_size)
    # input_kernel = (num_of_frames, 20, 20)
    # hidden_kernel = (num_of_frames, 20, 20)
    input_regularizer = "LaplaceL2norm"
    conv3d = True
    readout = args.readout
    overlapping = True
    stopper_patience = args.stopper_patience

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

    os.listdir(f"{basepath}/data")

    neuronal_data_path = os.path.join(basepath, "data/responses/")
    # neuronal_data_path = os.path.join(basepath, 'data/dummy_data/')
    training_img_dir = os.path.join(basepath, "data/non_repeating_stimuli/")
    test_img_dir = os.path.join(basepath, "data/repeating_stimuli/")

    dataset_fn = "datasets.white_noise_loader"
    dataset_config = dict(
        neuronal_data_dir=neuronal_data_path,
        train_image_path=training_img_dir,
        test_image_path=test_img_dir,
        batch_size=batch_size,
        crop=crop,
        subsample=subsample,
        seed=dataset_seed,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=False,
        num_of_frames=num_of_frames,
        cell_index=cell,
        retina_index=retina_index,
        conv3d=conv3d,
        time_chunk_size=args.trial_portion,
        overlapping=overlapping,
        num_of_layers=layers,
    )

    dataloaders = builder.get_data(dataset_fn, dataset_config)
    print(dataloaders)

    first_session_ID = list((dataloaders["train"].keys()))[0]
    print(first_session_ID)
    a_dataloader = dataloaders["train"][first_session_ID]
    debug_output = a_dataloader.dataset.__getitem__(99)
    inputs, targets = next(iter(a_dataloader))

    max_coordinate = None

    img_h = inputs.shape[-2]
    img_w = inputs.shape[-1]
    if conv3d:
        size = inputs.shape[2], img_h, img_w
    else:
        size = num_of_frames, img_h, img_w

    print("input shape:", size)
    model_dir = f"models/basic_cnn/all_retinas/readout_{readout}/lr_{lr}_l_{layers}_ch{hidden_channels}_t{num_of_frames}_g_{gamma}_bs_{batch_size}_tr_{num_of_trials}_s_{subsample}_cr_{crop}_ik{input_kernel[1]}_hk{hidden_kernel[1]}_l1_{l1}_mu_{init_mu_range}_sg_{init_sigma}_bn_{args.batch_norm}"

    model_fn = "models.multi_retina_regular_cnn_model"
    model_config = {
        "hidden_channels": hidden_channels,
        "input_kern": input_kernel,
        "seed": model_seed,
        "hidden_kern": hidden_kernel,
        "core_nonlinearity": args.core_nonlin,
        "bias": bias,
        "laplace_padding": None,
        "input_regularizer": input_regularizer,
        "layers": layers,
        "gamma_input": gamma,
        "l1": l1,
        "padding": padding,
        "batch_norm": batch_norm,
        "readout": readout,
        "final_nonlinearity": args.readout_nonlin,
        "init_mu_range": init_mu_range,
        "init_sigma": init_sigma,
        "readout_bias": readout_bias,
        "grid_mean_predictor": True,
    }
    model = builder.get_model(model_fn, model_config, dataloaders=dataloaders)
    model = model.double()
    model = model.to(device)

    loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    loss_function_config = dict(
        loss_function=nn.PoissonNLLLoss, log_input=False, reduction="sum"
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    optimizer_config = dict(optimizer=optim.Adam, lr=lr, weight_decay=args.l2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, verbose=True, min_lr=1e-7
    )
    scheduler_config = dict(
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        mode="max",
        patience=5,
        verbose=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    correlations = []
    valid_losses = []
    train_losses = []
    penalties = []
    prediction_variances = []
    valid_prediction_variances = []
    true_train_prediction_variances = []
    true_valid_prediction_variances = []

    weight_dir = f"{log_dir}/{model_dir}/weights/"
    stats_dir = f"{log_dir}/{model_dir}/stats/"
    config_dir = f"{log_dir}/{model_dir}/config/"

    Path(weight_dir).mkdir(exist_ok=True, parents=True)
    Path(stats_dir).mkdir(exist_ok=True, parents=True)
    Path(config_dir).mkdir(exist_ok=True, parents=True)

    max_avg_valid_corr = -1

    config_dict = dict(
        epochs=epochs,
        base_path=basepath,
        img_w=img_w,
        img_h=img_h,
        model_dir=model_dir,
        log_dir=log_dir,
        size=size,
        scheduler_config=scheduler_config,
        optimizer_config=optimizer_config,
        loss_function_config=loss_function_config,
        model_config=model_config,
        dataloader_config=dataset_config,
    )

    with open(f"{config_dir}/config.yaml", "w") as file:
        yaml.dump(config_dict, file)

    patience = stopper_patience
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()

        epoch_correlations = []
        epoch_train_losses = []
        epoch_penalties = []
        epoch_valid_losses = []
        epoch_prediction_variances = []
        epoch_valid_prediction_variances = []

        if conv3d:
            rf_size = size[1:]
            # rf_size = (20, 20)
        else:
            rf_size = size
        # max_response_count = 0
        # max_response_count_index = None
        # min_len = len(dataloaders['train']['02'])
        # for i, (data_01, data_02, data_03, data_04, data_05) in enumerate(zip(dataloaders['train']['01'],dataloaders['train']['02'],
        #                                             dataloaders['train']['03'], dataloaders['train']['04'],
        #                                             dataloaders['train']['05'])):
        #     if i < min_len:
        #         response_count = sum([torch.sum(x[1]) for x in [data_01, data_02, data_03, data_04, data_05]])
        #         if response_count > max_response_count:
        #             print([torch.sum(x[1]) for x in [data_01, data_02, data_03, data_04, data_05]])
        #             max_response_count = response_count
        #             max_response_count_index = i
        # print(max_response_count)
        # print(max_response_count_index)
        max_response_count_index = 55
        multi_batch_imgs = []
        multi_batch_responses = []
        for retina in ["01", "02", "03", "04", "05"]:
            batch_img = None
            batch_responses = None
            starting_index = batch_size * max_response_count_index
            ending_index = batch_size * (max_response_count_index + 1)
            for i in range(starting_index, ending_index):
                images, responses = dataloaders["train"][retina].dataset.__getitem__(i)
                if batch_img is None:
                    batch_img = torch.unsqueeze(images, dim=0)
                    batch_responses = torch.unsqueeze(responses, dim=0)
                else:
                    batch_img = torch.cat(
                        (batch_img, torch.unsqueeze(images, dim=0)), dim=0
                    )
                    batch_responses = torch.cat(
                        (batch_responses, torch.unsqueeze(responses, dim=0)), dim=0
                    )
            multi_batch_imgs.append(batch_img)
            multi_batch_responses.append(batch_responses)

        for data_key, images, responses in zip(
            ["01", "02", "03", "04", "05"], multi_batch_imgs, multi_batch_responses
        ):
            # loss_scale = np.sqrt(len(dataloaders[data_key].dataset) / images.shape[0]) if scale_loss else 1.0
            # print(batch_no, data_key)
            loss, penalty, prediction_variance, corr = train_step(
                images=images.to(device),
                responses=responses.to(device),
                model=model,
                loss_function=loss_function,
                optimizer=optimizer,
                rf_size=rf_size,
                max_coordinate=max_coordinate,
                h=img_h,
                w=img_w,
                penalty_coef=1,
                data_key=data_key,
            )
            epoch_train_losses.append(float(loss.item()))
            epoch_penalties.append(float(penalty.item()))
            epoch_prediction_variances.append(float(prediction_variance))
            epoch_correlations.append(np.mean(corr.detach().cpu().numpy()[0]))
            print(data_key)
            if epoch == 0:
                true_train_prediction_variances.append(
                    variance_of_predictions(responses)
                )
        if epoch == 0:
            true_train_prediction_variances = sum(
                true_train_prediction_variances
            ) / len(true_train_prediction_variances)
        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
        penalties.append(sum(epoch_penalties) / len(epoch_penalties))
        prediction_variances.append(
            sum(epoch_prediction_variances) / len(epoch_prediction_variances)
        )
        correlations.append(np.mean(epoch_correlations, axis=0))
        print("")
        print(model_dir)
        print("train_correaltion:", correlations[-1])
        print("avg train loss: ", train_losses[-1], "penalty:", penalties[-1])
        print(
            "variances:",
            prediction_variances[-1],
            "actual variance:",
            true_train_prediction_variances,
        )

        if (epoch + 1) % 1000 == 0:
            save_checkpoint(
                epoch,
                model=model,
                path=f"{weight_dir}/overfit_model.m",
                optimizer=optimizer,
                valid_corr=0,
            )

            with torch.no_grad():
                np_correlations = np.array(correlations)
                np_train_losses = np.array(train_losses)
                # np_valid_losses = np.array(valid_losses)
                np_penalties = np.array(penalties)
                np_variances = np.array(prediction_variances)
                # np_valid_variances = np.array(valid_prediction_variances)
                np.save(f"{stats_dir}/penalties", penalties)
                np.save(f"{stats_dir}/variances", prediction_variances)
                np.save(f"{stats_dir}/correlations", np_correlations)
                np.save(f"{stats_dir}/train_losses", np_train_losses)
                # np.save(f'{stats_dir}/valid_losses', np_valid_losses)
                # np.save(f'{stats_dir}/valid_variances', np_valid_variances)
