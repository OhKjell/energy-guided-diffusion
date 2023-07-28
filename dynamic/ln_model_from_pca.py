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
from neuralpredictors.regularizers import LaplaceL2norm
from nnfabrik import builder
from torch import nn, optim
from tqdm import tqdm

from datasets.stas import get_stas_from_pcs
from models.ln_model import Model, fit_nonlinearity
from training.measures import correlation, variance_of_predictions
from training.trainers import model_step, save_checkpoint, train_step
from utils.global_functions import (dataset_seed,
                                    get_exclude_cells_based_on_thresholds,
                                    global_config, home, model_seed)

# random.seed(seed)
# np.random.seed(seed)
cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"
# torch.manual_seed(seed)
divnorm = colors.TwoSlopeNorm(vcenter=0.0)


def get_receptive_field_slice(file, cell_index, time_frame_index):
    receptive_fields = np.load(file)
    receptive_field = receptive_fields[cell_index, time_frame_index]
    return receptive_field


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument("--num_of_channels", default=25, type=int)
parser.add_argument("--num_of_pcs", default=16, type=int)
parser.add_argument(
    "--data_path",
    default="/user/vystrcilova/",
    type=str,
    help="path to the data, if None, the root of the project/data is considered",
)
parser.add_argument("--rf_size", default=20, type=int, help="")
parser.add_argument("--num_of_trials", default=250, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--image_height", default=150, type=int)
parser.add_argument("--l1", default=0.01, type=float)
parser.add_argument("--l2", default=0.1, type=float)
parser.add_argument("--crop", default=0, type=int, nargs=4)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--cell_index", default="None", type=str)
parser.add_argument("--retina_index", default=0, type=int)
parser.add_argument("--log_dir", default="None", type=str)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--stopper_patience", default=15, type=int)
parser.add_argument("--norm", default=0, type=int)
parser.add_argument("--pca", default=1, type=int)
parser.add_argument("--fit_nonlin", default=0, type=int)
parser.add_argument("--multi_retinal", default=1, type=int)
parser.add_argument("--dataset", default="validation", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
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
    num_of_pcs = args.num_of_pcs
    if args.data_path is None:
        basepath = os.path.dirname(os.path.abspath(__file__))
    else:
        basepath = args.data_path

    with open(
        f"{home}/data/salamander_data/responses/config.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)

    filter_type = "pca"
    if args.pca == 0:
        filter_type = "sta"
        num_of_pcs = None
    l1 = args.l1
    l2 = args.l2
    gamma = args.gamma
    retina_index = args.retina_index
    fit_non_lin = False
    if args.fit_nonlin == 1:
        fit_non_lin = True
    multi_retinal_pca = False
    multi_ret_str = ""
    if args.multi_retinal == 1:
        multi_retinal_pca = True
        multi_ret_str = "multi_ret_pca_"

    normalize_response = False
    if args.norm == 1:
        normalize_response = True

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

    neuronal_data_path = os.path.join(basepath, "data/salamander_data/responses/")
    # neuronal_data_path = os.path.join(basepath, 'data/dummy_data/')
    training_img_dir = os.path.join(
        basepath, "data/salamander_data/non_repeating_stimuli/"
    )
    test_img_dir = os.path.join(basepath, "data/salamander_data/repeating_stimuli/")

    # if os.path.isdir(f'{basepath}/{model_dir}'):
    #     continue
    if cells is None:
        cells = [
            x
            for x in range(config_dict["cell_numbers"][str(retina_index + 1).zfill(2)])
        ]
    cell_corrs = []
    print("files: ", config_dict["files"])
    for cell in tqdm(cells):
        dataset_fn = "datasets.white_noise_loader"
        dataset_config = dict(
            config=config_dict,
            neuronal_data_dir=neuronal_data_path,
            train_image_path=training_img_dir,
            test_image_path=test_img_dir,
            batch_size=batch_size,
            crop=crop,
            subsample=subsample,
            seed=dataset_seed,
            num_of_trials_to_use=num_of_trials,
            use_cache=True,
            movie_like=True,
            num_of_frames=num_of_channels,
            cell_index=cell,
            cell_indices_out_of_range=False,
            retina_index=retina_index,
            normalize_responses=normalize_response,
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

        model_dir = f"models/ln_pca_models/salamander/retina{retina_index + 1}/cell_{cell}/{filter_type}_rf_full_{rf_size}_ch_{num_of_channels}_bs_{batch_size}_tr_{num_of_trials}_non_lin_{args.fit_nonlin}_crop_{crop}_norm_{args.norm}_pcs_{num_of_pcs}_{multi_ret_str}{args.dataset}"
        half_rf_size = rf_size // 2
        residual = half_rf_size % 2
        if half_rf_size is not None:
            receptive_field = np.load(
                f"{basepath}/data/salamander_data/stas/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_cell_{cell}.npy"
            )
            receptive_field = a_dataloader.dataset.transform(receptive_field)
            temporal_variances = np.var(receptive_field, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(temporal_variances), (img_h, img_w)
            )
            size = (
                max_coordinate[0] + min(half_rf_size, img_h - max_coordinate[0])
            ) + residual - (max_coordinate[0] - min(half_rf_size, max_coordinate[0])), (
                max_coordinate[1] + min(half_rf_size, img_w - max_coordinate[1])
            ) + residual - (
                max_coordinate[1] - min(half_rf_size, max_coordinate[1])
            )

        else:
            size = (img_h, img_w)
        if size != (rf_size, rf_size):
            print("cell too close to border, breaking")
            cell_corrs.append(0)
            continue

        input_shape = size

        model = Model(
            input_shape=input_shape,
            num_of_neurons=targets.shape[1],
            num_of_frames=num_of_channels,
            l1=l1,
            gamma=gamma,
            nonlin=False,
        )
        model_config = dict(
            input_shape=input_shape,
            num_of_neurons=targets.shape[1],
            num_of_channels=num_of_channels,
            l1=l1,
            gamma=gamma,
            nonlin=False,
        )
        if multi_retinal_pca:
            all_retina_indices = list(range(config_dict["Number_of_datasets"]))
        else:
            all_retina_indices = retina_index
        pca_based_stas, pca_mean, cropped_stas = get_stas_from_pcs(
            retina_index=all_retina_indices,
            input_kern=(num_of_channels, rf_size, rf_size),
            plot=False,
            data_dir=home,
            rf_size=rf_size,
            num_of_loadings=num_of_pcs,
            predicted_retina_index_str=f"0{retina_index+1}",
        )
        if filter_type == "pca":
            model.conv1.weight.data[0] = torch.tensor(pca_based_stas[cell])
        else:
            model.conv1.weight.data[0] = torch.tensor(cropped_stas[cell])
        _, targets = next(iter(dataloaders["train"][f"0{retina_index + 1}"]))
        # model.conv1.bias.data = targets.mean(0)
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
        valid_losses = []
        train_losses = []
        valid_variances = []
        prediction_variances = []
        all_valid_prediction_variances = []
        true_train_prediction_variances = []
        true_valid_prediction_variances = []
        penalties = []

        weight_dir = f"{log_dir}/{model_dir}/weights/"
        stats_dir = f"{log_dir}/{model_dir}/stats/"
        config_dir = f"{log_dir}/{model_dir}/config/"

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

        print(f"Epoch 0")
        print(model_dir)
        model.train()

        epoch_correlations = []
        epoch_train_correlations = []
        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_penalties = []
        epoch_train_variances = []
        epoch_valid_variances = []

        model.eval()
        non_lin_sta = (
            pca_based_stas[cell] if filter_type == "pca" else cropped_stas[cell]
        )
        if fit_non_lin:
            coefs = fit_nonlinearity(
                dataloaders,
                model_filter=torch.from_numpy(non_lin_sta),
                device=device,
                retina_index=retina_index,
                max_coordinate=max_coordinate,
                rf_size=size,
                h=img_h,
                w=img_w,
                plot=False,
            )
        all_responses = []
        all_outputs = []
        with torch.no_grad():
            for images, responses in dataloaders[args.dataset][
                str(retina_index + 1).zfill(2)
            ]:
                images = images.double().to(device)
                responses = responses.to(device)
                all_responses = all_responses + [x.item() for x in responses]
                output = model_step(
                    images=images,
                    model=model,
                    max_coordinate=max_coordinate,
                    rf_size=size,
                    h=img_h,
                    w=img_w,
                )
                if fit_non_lin:
                    output = coefs[0] * torch.exp(coefs[1] * output)
                all_outputs = all_outputs + [x.item() for x in output]
                valid_prediciton_variance = variance_of_predictions(output)
                corr = correlation(output, responses, 1e-12)
                valid_loss = loss_function(output, responses)
                if cuda:
                    epoch_correlations.append(np.mean(corr.detach().cpu().numpy()[0]))
                    epoch_valid_losses.append(float(valid_loss.detach().cpu().numpy()))
                else:
                    epoch_correlations.append(np.mean(corr.detach().numpy()[0]))
                    epoch_valid_losses.append(float(valid_loss.detach().numpy()))
                epoch_valid_variances.append(valid_prediciton_variance)

                true_valid_prediction_variances.append(
                    variance_of_predictions(responses)
                )

            true_valid_prediction_variances = sum(
                true_valid_prediction_variances
            ) / len(true_valid_prediction_variances)

            print(model_dir)
            print("max cell valid corr:", max(epoch_correlations))
            single_correlations = np.mean(epoch_correlations, axis=0)
            cell_corrs.append(single_correlations)
            # print('cell valid corr:', single_correlations)
            print("avg valid corr:", single_correlations)
            print("avg valid loss:", sum(epoch_valid_losses) / len(epoch_valid_losses))
            print("max valid loss:", max(epoch_valid_losses))

            print(
                "avg valid variance:",
                sum(epoch_valid_variances) / len(epoch_valid_variances),
                "true prediction variance:",
                true_valid_prediction_variances,
            )
            if single_correlations > max_avg_valid_corr:
                patience = stopper_patience
                print(
                    f"Saving so far best model:{single_correlations}, previous best:{max_avg_valid_corr}"
                )
                max_avg_valid_corr = single_correlations
                save_checkpoint(
                    0,
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
            print(all_responses[:250])
            print(all_outputs[:250])
            # plt.plot(np.arange(len(all_responses))[:250], all_responses[:250], label='True')
            # plt.plot(np.arange(len(all_responses))[:250], all_outputs[:250], label='Predicted')
            # plt.legend()
            # plt.title('Single cell model')
            # plt.show()

        with torch.no_grad():
            # if (k % 5) == 0:
            #     if cuda:
            #         weights = model.conv1.weight.detach().cpu().numpy()
            #     else:
            #         weights = model.conv1.weight.detach().numpy()
            #     np.save(f'{weight_dir}/w_epoch_{k}', weights)

            np_correlations = np.array(cell_corrs)
            np_valid_losses = np.array(valid_losses)
            np_variances = np.array(prediction_variances)
            np_valid_variances = np.array(all_valid_prediction_variances)
            np.save(f"{stats_dir}/cell_correlations", np_correlations)
            np.save(f"{stats_dir}/valid_losses", np_valid_losses)
            np.save(f"{stats_dir}/valid_variances", np_valid_variances)
    for i, corr in enumerate(cell_corrs):
        print(f"cell: {i} - corr: {corr}")
    excluded_cells = get_exclude_cells_based_on_thresholds(
        retina_index=retina_index, explainable_variance_threshold=0.15
    )
    # excluded_cells = exclude_cells[f'0{retina_index+1}']
    print("excluded cells:", excluded_cells)
    print("avg:", np.mean(np.delete(cell_corrs, excluded_cells)))
