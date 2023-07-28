import argparse
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from nnfabrik import builder
from torch import nn, optim
from torchviz import make_dot
from tqdm import tqdm

from training.measures import correlation, variance_of_predictions
from training.trainers import model_step, save_checkpoint, train_step
from utils.global_functions import (dataset_seed, get_cell_names, home,
                                    model_seed)

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
parser.add_argument("--num_of_frames", default=15, type=int)
parser.add_argument(
    "--data_path",
    default=None,
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--num_of_trials", default=250, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--gamma", default=0.33, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--l1", default=1, type=float)
parser.add_argument("--time_chunk", default=10, type=int)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--cell_index", default="all", type=str)
parser.add_argument("--retina_index", default=0, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--trial_portion", default=0.1, type=float)
parser.add_argument("--explainable_variance_threshold", default=0.15, type=float)
parser.add_argument("--hidden_kernel_size", default=3, type=int)
parser.add_argument("--input_kernel_size", default=20, type=int)
parser.add_argument("--padding", default=0, type=int)
parser.add_argument("--core_nonlin", default="elu", type=str)
parser.add_argument("--hidden_channels", default=8, type=int)
parser.add_argument("--readout", default="isotropic_gaussian", type=str)
parser.add_argument("--readout_nonlin", default="relu", type=str)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--bias", default=1, type=int)
parser.add_argument("--readout_bias", default=1, type=int)
parser.add_argument("--init_sigma", default=0.1, type=float)
parser.add_argument("--batch_norm", default=0, type=int)
parser.add_argument("--gmp", default=0, type=int)
parser.add_argument("--train_core", default=0, type=int)
parser.add_argument("--train_readout", default=1, type=int)
parser.add_argument("--nonlin", default=0, type=int)


def plot_responses(
    predicted_responses, true_responses, retina_index, max_len=500, fev_thresholds=None
):
    if fev_thresholds is None:
        fev_thresholds = [0]
    predicted_responses = np.concatenate(predicted_responses)
    true_responses = np.concatenate(true_responses)
    for fev_threshold in fev_thresholds:
        corrs = []
        always_present = get_cell_names(retina_index)
        cell_names = get_cell_names(
            retina_index=retina_index,
            correlation_threshold=0,
            explained_variance_threshold=fev_threshold,
        )
        for i, cell in enumerate(always_present):
            if cell in cell_names:
                fig, ax = plt.subplots(figsize=(20, 10))
                corr = correlation(
                    torch.tensor(predicted_responses[:, i]),
                    torch.tensor(true_responses[:, i]),
                    eps=1e-8,
                )
                corrs.append(corr.item())
                plt.plot(
                    np.arange(min(max_len, len(predicted_responses))),
                    predicted_responses[: min(max_len, len(predicted_responses)), i],
                    label="predicted",
                )
                plt.plot(
                    np.arange(min(max_len, len(true_responses))),
                    true_responses[: min(max_len, len(true_responses)), i],
                    label="true",
                )
                plt.legend()
                plt.plot()
                plt.title(f"Cell {cell}, cc {corr.item():.2f}")
                plt.show()
        print(f"threshold: {fev_threshold}, cc: {np.mean(corrs)}")


if __name__ == "__main__":
    args = parser.parse_args()
    with open(
        f"{home}/data/salamander_data/responses/config.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    lr = args.learning_rate
    epochs = args.epochs
    num_of_frames = args.num_of_frames
    num_of_trials = args.num_of_trials
    batch_size = args.batch_size
    layers = 1
    img_w = args.image_width
    img_h = args.image_height
    retina_index = args.retina_index

    crop = config_dict["big_crops"][str(retina_index + 1).zfill(2)]
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

    train_core = False
    if args.train_core == 1:
        train_core = True

    train_readout = True
    if args.train_readout == 0:
        train_readout = False

    batch_norm = True
    if args.batch_norm == 0:
        batch_norm = False

    use_grid_mean_predictor = False
    if args.gmp == 1:
        use_grid_mean_predictor = True

    if retina_index not in range(0, 5):
        retina_index = None
        print("loading data for all retinas")
    # crop = 50, 75, 110, 60
    # crop = 0
    subsample = args.subsample

    # regularization parameters
    l1 = args.l1
    gamma = args.gamma
    input_regularizer = "LaplaceL2norm"

    # full gaussian readout parameters
    readout = args.readout
    init_mu_range = 0.5
    init_sigma = args.init_sigma

    hidden_channels = args.hidden_channels
    explainable_variance_threshold = args.explainable_variance_threshold
    input_kernel = (num_of_frames, args.input_kernel_size, args.input_kernel_size)
    hidden_kernel = (num_of_frames, args.hidden_kernel_size, args.hidden_kernel_size)

    conv3d = True
    overlapping = False
    time_chunk_size = args.time_chunk

    stopper_patience = args.stopper_patience

    if args.cell_index != "all":
        cell = int(args.cell_index)
    else:
        cell = None

    nonlin = True
    if args.nonlin == 0:
        nonlin = False

    if args.data_path is None:
        basepath = os.path.dirname(os.path.abspath(__file__))
    else:
        basepath = args.data_path

    if args.log_dir == "None":
        log_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        log_dir = args.log_dir

    os.listdir(f"{basepath}/data")

    neuronal_data_path = os.path.join(basepath, "data/salamander_data/responses/")
    # neuronal_data_path = os.path.join(basepath, 'data/dummy_data/')
    training_img_dir = os.path.join(
        basepath, "data/salamander_data/non_repeating_stimuli/"
    )
    test_img_dir = os.path.join(basepath, "data/salamander_data/repeating_stimuli/")

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
        conv3d=True,
        num_of_frames=num_of_frames,
        cell_index=cell,
        retina_index=retina_index,
        time_chunk_size=time_chunk_size,
        overlapping=overlapping,
        num_of_layers=layers,
        explainable_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold > 0
        else None,
        config=config_dict,
    )

    dataloaders = builder.get_data(dataset_fn, dataset_config)
    print(dataloaders)

    # debug output
    first_session_ID = list((dataloaders["train"].keys()))[0]
    print(first_session_ID)
    a_dataloader = dataloaders["train"][first_session_ID]
    debug_output = a_dataloader.dataset.__getitem__(99)
    inputs, targets = next(iter(a_dataloader))

    # uncomment in case you want to crop around the receptive field
    # receptive_field = np.load(
    #     f'{basepath}/data/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_25.npy')[
    #     cell]
    # receptive_field = a_dataloader.dataset.transform(receptive_field)
    # temporal_variances = np.var(receptive_field, axis=0)
    # max_coordinate = np.unravel_index(np.argmax(temporal_variances), (img_h, img_w))

    max_coordinate = None
    img_h = inputs.shape[-2]
    img_w = inputs.shape[-1]
    if conv3d:
        size = inputs.shape[2], img_h, img_w
    else:
        size = num_of_frames, img_h, img_w

    print("input shape:", size)
    print("train_core", train_core)
    print("train_readout", train_readout)

    model_dir = f"models/sta_model/retina{retina_index + 1}/cell_{cell}/readout_{readout}/gmp_{args.gmp}/lr_{lr}_l_{layers}_ch_{hidden_channels}_t_{num_of_frames}_g_{gamma}_bs_{batch_size}_tr_{num_of_trials}_s_{subsample}_cr_{crop}_ik_{input_kernel[1]}_l1_{l1}_mu_{init_mu_range}_sg_{init_sigma}_p_{args.padding}_test"

    model_fn = "models.sta_model"
    model_config = {
        "hidden_channels": hidden_channels,
        "input_kern": input_kernel,
        "hidden_kern": hidden_kernel,
        "core_nonlinearity": args.core_nonlin,
        "bias": bias,
        "laplace_padding": None,
        "input_regularizer": input_regularizer,
        "gamma_input": gamma,
        "l1": l1,
        "padding": padding,
        "batch_norm": batch_norm,
        "readout": readout,
        "final_nonlinearity": args.readout_nonlin,
        "init_mu_range": init_mu_range,
        "init_sigma": init_sigma,
        "readout_bias": readout_bias,
        "retina_index": retina_index,
        "data_dir": basepath,
        "use_grid_mean_predictor": args.gmp,
        "train_core": train_core,
        "train_readout": train_readout,
        "explainable_variance_threshold": explainable_variance_threshold,
        "cell_index": cell,
        "nonlin": nonlin,
    }

    model = builder.get_model(
        model_fn, model_config, seed=model_seed, dataloaders=dataloaders
    )
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

    correlations = []
    train_correlations = []
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

        epoch_train_correlations = []
        epoch_valid_correlations = []
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
        # for i, (images, responses) in enumerate(dataloaders['train'][str(retina_index + 1).zfill(2)]):
        #     response_count = torch.sum(responses)
        #     if response_count > max_response_count:
        #         max_response_count = response_count
        #         max_response_count_index = i
        # batch_img = None
        # batch_responses = None
        # starting_index = batch_size*44
        # ending_index = batch_size*45
        # for i in range(starting_index, ending_index):
        #     images, responses = dataloaders['train'][str(retina_index + 1).zfill(2)].dataset.__getitem__(i)
        #     if batch_img is None:
        #         batch_img = torch.unsqueeze(images, dim=0)
        #         batch_responses = torch.unsqueeze(responses, dim=0)
        #     else:
        #         batch_img = torch.cat((batch_img, torch.unsqueeze(images, dim=0)), dim=0)
        #         batch_responses = torch.cat((batch_responses, torch.unsqueeze(responses, dim=0)), dim=0)
        #
        # for batch_images, responses in dataloaders['train'][str(retina_index + 1).zfill(2)][22]:
        if epoch != 0:
            model.train()
            for i, (images, responses) in enumerate(
                tqdm(dataloaders["validation"][str(retina_index + 1).zfill(2)])
            ):
                # images = batch_img
                # responses = batch_responses
                responses = responses.transpose(1, 2)
                responses = torch.flatten(responses, start_dim=0, end_dim=1).to(device)
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
                )
                epoch_train_losses.append(float(loss.item()))
                epoch_penalties.append([x.item() for x in penalty])
                epoch_prediction_variances.append(float(prediction_variance))
                epoch_train_correlations.append(np.mean(corr.detach().cpu().numpy()[0]))
                if epoch == 1:
                    true_train_prediction_variances.append(
                        variance_of_predictions(responses)
                    )
            if epoch == 1:
                true_train_prediction_variances = sum(
                    true_train_prediction_variances
                ) / len(true_train_prediction_variances)
            train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
            epoch_penalties = np.array(epoch_penalties)
            penalties.append(np.mean(epoch_penalties, axis=0))
            prediction_variances.append(
                sum(epoch_prediction_variances) / len(epoch_prediction_variances)
            )
            train_correlations.append(np.mean(epoch_train_correlations, axis=0))
            print("")
            print(model_dir)
            print("avg train correlation", train_correlations[-1])
            print("avg train loss: ", train_losses[-1])
            print(
                "spatial penalty:",
                penalties[-1][0],
                "temporal_penalty:",
                penalties[-1][1],
                "l1 penalty:",
                penalties[-1][2],
            )
            print(
                "variances:",
                prediction_variances[-1],
                "actual variance:",
                true_train_prediction_variances,
            )

        model.eval()
        initial_responses = []
        true_responses = []
        with torch.no_grad():
            for images, responses in tqdm(
                dataloaders["validation"][str(retina_index + 1).zfill(2)]
            ):
                images = images.double().to(device)
                responses = responses.transpose(1, 2)
                responses = torch.flatten(responses, start_dim=0, end_dim=1).to(device)
                output = model_step(
                    images=images,
                    model=model,
                    max_coordinate=max_coordinate,
                    rf_size=rf_size,
                    h=img_h,
                    w=img_w,
                )
                # if epoch == 0:
                #     initial_responses = initial_responses + [x.item() for x in output]
                #     true_responses = true_responses + [x.item() for x in responses]
                valid_prediciton_variance = variance_of_predictions(output)
                corr = correlation(output, responses, 1e-12)
                valid_loss = loss_function(output, responses)
                if cuda:
                    epoch_valid_correlations.append(
                        np.mean(corr.detach().cpu().numpy()[0])
                    )
                    epoch_valid_losses.append(float(valid_loss.detach().cpu().numpy()))
                else:
                    epoch_valid_correlations.append(np.mean(corr.detach().numpy()[0]))
                    epoch_valid_losses.append(float(valid_loss.detach().numpy()))
                epoch_valid_prediction_variances.append(valid_prediciton_variance)
                if epoch == 0:
                    true_valid_prediction_variances.append(
                        variance_of_predictions(responses)
                    )
            if epoch == 0:
                true_valid_prediction_variances = sum(
                    true_valid_prediction_variances
                ) / len(true_valid_prediction_variances)
                print("Initial performance")
            print(model_dir)
            # print('avg train loss: ', train_losses[-1])
            print("max batch valid corr:", max(epoch_valid_correlations))
            single_correlations = np.mean(epoch_valid_correlations, axis=0)
            print("avg valid corr:", single_correlations)
            print("avg valid loss:", sum(epoch_valid_losses) / len(epoch_valid_losses))
            print("max valid loss:", max(epoch_valid_losses))
            print(
                "avg valid variance:",
                sum(epoch_valid_prediction_variances)
                / len(epoch_valid_prediction_variances),
                "true prediction variance:",
                true_valid_prediction_variances,
            )

            plt.plot(np.arange(len(true_responses)), true_responses, label="True")
            plt.plot(
                np.arange(len(initial_responses)), initial_responses, label="Predicted"
            )
            plt.legend()
            plt.title("Population model")
            plt.show()
            if epoch == 0:
                save_checkpoint(
                    epoch,
                    model=model,
                    path=f"{weight_dir}/initial_model.m",
                    optimizer=optimizer,
                    valid_corr=single_correlations,
                )
            if single_correlations > max_avg_valid_corr:
                patience = stopper_patience
                print(
                    f"Saving so far best model:{single_correlations}, previous best:{max_avg_valid_corr}"
                )
                max_avg_valid_corr = single_correlations
                save_checkpoint(
                    epoch,
                    model=model,
                    path=f"{weight_dir}/best_model.m",
                    optimizer=optimizer,
                    valid_corr=single_correlations,
                )
            else:
                patience -= 1
                if patience == 0:
                    break

            valid_losses.append(sum(epoch_valid_losses) / len(epoch_valid_losses))
            correlations.append(single_correlations)
            valid_prediction_variances.append(
                sum(epoch_valid_prediction_variances)
                / len(epoch_valid_prediction_variances)
            )
            scheduler.step(single_correlations)
        if epoch == 0:
            plot_responses(
                retina_index=retina_index,
                predicted_responses=initial_responses,
                true_responses=true_responses,
                fev_thresholds=[0, 0.15, 0.3, 0.45, 0.6],
            )

        with torch.no_grad():
            np_correlations = np.array(correlations)
            np_train_correlations = np.array(train_correlations)
            np_train_losses = np.array(train_losses)
            np_valid_losses = np.array(valid_losses)
            np_penalties = np.array(penalties)
            np_variances = np.array(prediction_variances)
            np_valid_variances = np.array(valid_prediction_variances)
            np.save(f"{stats_dir}/penalties", penalties)
            np.save(f"{stats_dir}/variances", prediction_variances)
            np.save(f"{stats_dir}/correlations", np_correlations)
            np.save(f"{stats_dir}/train_correlations", np_train_correlations)
            np.save(f"{stats_dir}/train_losses", np_train_losses)
            np.save(f"{stats_dir}/valid_losses", np_valid_losses)
            np.save(f"{stats_dir}/valid_variances", np_valid_variances)
