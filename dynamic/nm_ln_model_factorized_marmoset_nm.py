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
from training.trainers import model_step, save_checkpoint, train_step
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
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--l1", default=0.0, type=float)
parser.add_argument("--l2", default=0.0, type=float)
parser.add_argument("--crop", default=0, type=int, nargs=4)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--time_chunk", default=1, type=int)
parser.add_argument("--cell_index", default="None", type=str)
parser.add_argument("--retina_index", default=0, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--gamma", default=0.0, type=float)
parser.add_argument("--stopper_patience", default=7, type=int)
parser.add_argument("--norm", default=0, type=int)
parser.add_argument("--sta", default=1, type=int)
parser.add_argument("--svd", default=0, type=int)
parser.add_argument("--do_nonlin", default=1, type=int)
parser.add_argument("--fancy_nonlin", default=0, type=int)
parser.add_argument("--wandb", default=1, type=int)
parser.add_argument("--model_seed", default=8, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.data_path is None:
        basepath = os.path.dirname(os.path.abspath(__file__))
    else:
        basepath = args.data_path
    with open(
        f"{basepath}/data/marmoset_data/responses/config_s4_nm_sta_zero_mean.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    # crops = {'01': (20, 40, 50, 50), '02': (40, 35, 65, 65), '05': (35, 50, 65, 65)}
    # cell_numbers = {'01': 78, '02': 60, '05': 58}
    lr = args.learning_rate
    epochs = args.epochs
    stopper_patience = args.stopper_patience
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
    print("image path", img_dir)

    # if os.path.isdir(f'{basepath}/{model_dir}'):
    #     continue
    if cells is None:
        cells = [
            x
            for x in range(
                50, config_dict["cell_numbers"][str(retina_index + 1).zfill(2)]
            )
            if x not in config_dict["exclude_cells"][str(retina_index + 1).zfill(2)]
        ]
        # cells.remove(44)

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
        )

        dataloaders = builder.get_data(dataset_fn, dataset_config)
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
        inputs, targets = next(iter(a_dataloader))

        model_dir = f"models/ln_models/nm_ln_models_factorized_4_{filter_type}_nm/{data_type}/retina{retina_index + 1}/cell_{cell}/lr_{lr}_whole_rf_{rf_size}_ch_{num_of_channels}_l1_{l1}_l2_{l2}_g_{gamma}_bs_{batch_size}_tr_{num_of_trials}_s_{subsample}_c_{crop}_n_{args.norm}_fn_{args.fancy_nonlin}_do_n_{args.do_nonlin}/"
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
            receptive_field = a_dataloader.dataset.transform(receptive_field)
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
            num_of_neurons=targets.shape[1],
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

        loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum")
        loss_function_config = dict(
            loss_function=nn.PoissonNLLLoss, log_input=False, reduction="sum"
        )

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        optimizer_config = dict(optimizer=optim.Adam, lr=lr, weight_decay=args.l2)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, threshold=0.0001, verbose=True
        )
        scheduler_config = dict(
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            mode="max",
            patience=10,
            threshold=0.0001,
            verbose=True,
        )

        correlations = []
        train_correlations = []
        train_variances = []
        valid_losses = []
        train_losses = []
        valid_variances = []
        prediction_variances = []
        all_valid_prediction_variances = []
        true_train_prediction_variances = []
        true_valid_prediction_variances = []
        penalties = []

        weight_dir = f"{log_dir}/{model_dir}/weights/seed_{model_seed}/"
        stats_dir = f"{log_dir}/{model_dir}/stats/seed_{model_seed}/"
        config_dir = f"{log_dir}/{model_dir}/config/"
        # if os.path.isfile(f'{weight_dir}/best_model.m'):
        #     continue
        # train_loss_dir = f'{basepath}/{model_dir}/losses/train/'
        # valid_loss_dir = f'{basepath}/{model_dir}/losses/valid/'

        Path(weight_dir).mkdir(exist_ok=True, parents=True)
        Path(stats_dir).mkdir(exist_ok=True, parents=True)
        Path(config_dir).mkdir(exist_ok=True, parents=True)

        max_avg_valid_corr = -1

        config = dict(
            epochs=epochs,
            base_path=basepath,
            img_w=img_w,
            img_h=img_h,
            rf_size=rf_size,
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
            yaml.dump(config, file)
        if wandb_log:
            wandb.init(
                config=config,
                project=f"ln-model-{data_type}-factorized-4-{filter_type}-nm",
                entity="retinal-circuit-modeling",
            )

        if (not learn_filter) and (not fancy_nonlin):
            epochs = 1

        for k, epoch in enumerate(range(epochs)):
            print(
                f"model weight sum: {np.sum(model.conv1.weight.data.detach().cpu().numpy())}"
            )
            print(f"Epoch {k}")
            print(model_dir)
            model.train()

            epoch_correlations = []
            epoch_train_correlations = []
            epoch_train_losses = []
            epoch_valid_losses = []
            epoch_penalties = []
            epoch_train_variances = []
            epoch_valid_variances = []
            if learn_filter or fancy_nonlin:
                for images, responses in tqdm(
                    dataloaders["train"][str(retina_index + 1).zfill(2)]
                ):
                    loss, penalty, prediction_var, corr = train_step(
                        images=images.to(device),
                        responses=responses.to(device),
                        model=model,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        rf_size=size,
                        max_coordinate=max_coordinate,
                        h=img_h,
                        w=img_w,
                        penalty_coef=1,
                        data_key=None,
                    )
                    if loss is None:
                        break
                    epoch_train_losses.append(float(loss.item()))
                    epoch_train_variances.append(float(prediction_var.item()))
                    epoch_train_correlations.append(
                        np.mean(corr.detach().cpu().numpy()[0])
                    )
                    epoch_penalties.append([x.item() for x in penalty])
                    epoch_train_variances.append(float(prediction_var))

                    if epoch == 0:
                        true_train_prediction_variances.append(
                            variance_of_predictions(responses)
                        )
                if epoch == 0:
                    true_train_prediction_variances = sum(
                        true_train_prediction_variances
                    ) / len(true_train_prediction_variances)
                train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
                train_correlations.append(np.mean(epoch_train_correlations))
                train_corr = np.mean(epoch_train_correlations, axis=0)
                penalties.append(np.mean(epoch_penalties, axis=0))
                prediction_variances.append(
                    sum(epoch_train_variances) / len(epoch_train_variances)
                )

                print("")
                print("avg train correlation:", train_correlations[-1])
                print("avg train loss: ", train_losses[-1], "penalty:", penalties[-1])
                print(
                    "variances:",
                    prediction_variances[-1],
                    "actual variance:",
                    true_train_prediction_variances,
                )
                if wandb_log:
                    wandb.log(
                        {
                            "loss/train": train_losses[-1],
                            "spatial_penalty/train": penalties[-1][0],
                            "temporal_penalty/train": penalties[-1][1],
                            "l1_penalty/train": penalties[-1][2],
                            "variance/train": prediction_variances[-1],
                            "correlation/train": train_corr,
                            "variance/true_train": true_train_prediction_variances,
                        },
                        step=epoch,
                    )
            model.eval()

            with torch.no_grad():
                for images, responses in tqdm(
                    dataloaders["validation"][str(retina_index + 1).zfill(2)]
                ):
                    images = images.double().to(device)
                    responses = responses.to(device)
                    output = model_step(
                        images=images,
                        model=model,
                        max_coordinate=max_coordinate,
                        rf_size=size,
                        h=img_h,
                        w=img_w,
                    )
                    valid_prediction_variance = variance_of_predictions(output)
                    corr = correlation(output, responses.squeeze(-1), 1e-12)
                    valid_loss = loss_function(output, responses.squeeze(-1))
                    if cuda:
                        epoch_correlations.append(
                            np.mean(corr.detach().cpu().numpy()[0])
                        )
                        epoch_valid_losses.append(
                            float(valid_loss.detach().cpu().numpy())
                        )
                    else:
                        epoch_correlations.append(np.mean(corr.detach().numpy()[0]))
                        epoch_valid_losses.append(float(valid_loss.detach().numpy()))
                    epoch_valid_variances.append(valid_prediction_variance)
                    if epoch == 0:
                        true_valid_prediction_variances.append(
                            variance_of_predictions(responses)
                        )
                if epoch == 0:
                    true_valid_prediction_variances = sum(
                        true_valid_prediction_variances
                    ) / len(true_valid_prediction_variances)

                print(model_dir)
                print("max cell valid corr:", max(epoch_correlations))
                single_correlations = np.mean(epoch_correlations, axis=0)
                # print('cell valid corr:', single_correlations)
                print("avg valid corr:", single_correlations)
                print(
                    "avg valid loss:", sum(epoch_valid_losses) / len(epoch_valid_losses)
                )
                print("max valid loss:", max(epoch_valid_losses))

                print(
                    "avg valid variance:",
                    sum(epoch_valid_variances) / len(epoch_valid_variances),
                    "true prediction variance:",
                    true_valid_prediction_variances,
                )
                if wandb_log:
                    wandb.log(
                        {
                            "correlation/valid": single_correlations,
                            "loss/valid": sum(epoch_valid_losses)
                            / len(epoch_valid_losses),
                            "variance/valid": sum(epoch_valid_variances)
                            / len(epoch_valid_variances),
                            "variance/true_valid": true_valid_prediction_variances,
                        },
                        step=epoch,
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
                all_valid_prediction_variances.append(np.mean(epoch_valid_variances))
                scheduler.step(single_correlations)

            with torch.no_grad():
                # if (k % 5) == 0:
                #     if cuda:
                #         weights = model.conv1.weight.detach().cpu().numpy()
                #     else:
                #         weights = model.conv1.weight.detach().numpy()
                #     np.save(f'{weight_dir}/w_epoch_{k}', weights)

                np_correlations = np.array(correlations)
                np_train_correlations = np.array(train_correlations)
                np_train_losses = np.array(train_losses)
                np_valid_losses = np.array(valid_losses)
                np_penalties = np.array(penalties)
                np_variances = np.array(prediction_variances)
                np_valid_variances = np.array(all_valid_prediction_variances)
                np.save(f"{stats_dir}/penalties", penalties)
                np.save(f"{stats_dir}/variances", prediction_variances)
                np.save(f"{stats_dir}/correlations", np_correlations)
                np.save(f"{stats_dir}/train_correlations", np_train_correlations)
                np.save(f"{stats_dir}/train_losses", np_train_losses)
                np.save(f"{stats_dir}/valid_losses", np_valid_losses)
                np.save(f"{stats_dir}/valid_variances", np_valid_variances)
        if wandb_log:
            wandb.finish()
