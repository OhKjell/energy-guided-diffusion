import os
from pathlib import Path

import numpy as np
import torch
import yaml
from neuralpredictors.training import LongCycler
from nnfabrik import builder
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import models
# from line_profiler_pycharm import profile
import wandb
from dynamic.datasets.stas import (create_cell_file_from_config_version,
                           crop_around_receptive_field, get_cell_sta, show_sta)
from dynamic.models.helper_functions import get_model_and_dataloader
from dynamic.models.ln_model import FactorizedModel, Model
from dynamic.training.measures import correlation, variance_of_predictions
from dynamic.utils.global_functions import (dataset_seed, global_config, home,
                                    model_seed)

scaler = GradScaler()


def model_step(
    images, model, max_coordinate=None, rf_size=None, h=150, w=200, data_key=None
):
    if len(rf_size) == 3:
        cur_rf_size = rf_size[1:]
    else:
        cur_rf_size = rf_size
    if cur_rf_size != (h, w):
        images = crop_around_receptive_field(
            max_coordinate=max_coordinate, images=images, rf_size=cur_rf_size, h=h, w=w
        )
    if images.shape != cur_rf_size:
        output = None
    output = model(images.double(), data_key=data_key)
    if len(output.shape) == 1:
        output = torch.unsqueeze(output, 1)
    elif (len(output.shape) >= 3) and (output.shape[2] == 1):
        output = torch.squeeze(output, 2)

    return output


def train_step(
    images,
    responses,
    optimizer,
    model,
    loss_function,
    rf_size=None,
    max_coordinate=None,
    h=150,
    w=200,
    regularization=True,
    penalty_coef=1,
    data_key=None,
    regularizer_start=0,
    epoch=None,
):
    if regularizer_start != 0:
        assert epoch is not None
    optimizer.zero_grad()
    with autocast():
        output = model_step(
            images, model, max_coordinate, rf_size, h, w, data_key=data_key
        )
        if output is None:
            return None, None, None, None
        prediction_var = variance_of_predictions(output)
        # print(output.shape, responses.shape)
        if output.shape != responses.shape:
            responses = responses.squeeze(-1)
        assert output.shape == responses.shape

        corr = correlation(output, responses, 1e-12)
        if regularization and ((epoch is None) or (epoch >= regularizer_start)):
            core_spatial, core_temporal, readout = model.regularizer(data_key=data_key)
            penalty = core_spatial + core_temporal + readout
        else:
            penalty = 0
            core_spatial, core_temporal, readout = (
                torch.Tensor([0]),
                torch.Tensor([0]),
                torch.Tensor([0]),
            )
        loss = loss_function(output, responses) + penalty_coef * penalty

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss, [core_spatial, core_temporal, readout], prediction_var, corr


def save_checkpoint(epoch, model, path, optimizer, valid_corr):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict,
            "valid_corr": valid_corr,
        },
        path,
    )


def build_dataloader(dataset_config, dataset_fn):
    """
    TODO: does nothing, wtf!?
    :param dataset_config:
    :param dataset_fn:
    :return:
    """
    dataloaders = builder.get_data(dataset_fn, dataset_config)

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
    if dataset_config["conv3d"]:
        size = inputs.shape[2], img_h, img_w
    else:
        size = dataset_config["num_of_frames"], img_h, img_w

    print("input shape:", size)
    return dataloaders


def continue_training_cnn(
    directory,
    model_file,
    model_seed,
    data_dir,
    data_type="salamander",
    model_fn="models.FactorizedEncoder.build_trained",
    device="cuda",
    directory_prefix="factorized_ev_0.15",
    stopper_patience=20,
):
    dataloaders, model, config = get_model_and_dataloader(
        directory,
        model_file,
        model_fn=model_fn,
        device=device,
        data_dir=data_dir,
        test=False,
        seed=model_seed,
        data_type=data_type,
    )
    checkpoint = torch.load(
        f"{directory}/{model_file}/weights/seed_{model_seed}/best_model.m",
        map_location=torch.device(device),
    )
    log_dir = config["log_dir"]
    optimizer_config = config["optmizer_config"]
    scheduler_config = config["scheduler_config"]
    dataloader_config = config["dataloader_config"]
    loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    optimizer.load_state_dict(checkpoint["optimizer"])
    readout = "isotropic_gaussian"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
    model_dir = f'models/{directory_prefix}_cnn/{data_type}/retina{dataloader_config["retina_index"] + 1}/cell_{dataloader_config["cell_index"]}/readout_{readout}/gmp_0/{model_file}'
    weight_dir = f"{log_dir}/{model_dir}/weights/seed_{model_seed}"
    stats_dir = f"{log_dir}/{model_dir}/stats/seed_{model_seed}"

    score, correlations = train(
        epochs=1000,
        dataloaders=dataloaders,
        dataset_config=config["dataset_config"],
        config_dict=config,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        device=device,
        model_dir=model_dir,
        weight_dir=weight_dir,
        stats_dir=stats_dir,
        stopper_patience=stopper_patience,
        size=config["size"],
    )


def train_cnn(
    dataloader_config,
    model_config,
    directory_prefix="basic",
    lr=0.001,
    epochs=15,
    gmp=0,
    readout="isotropic",
    regularizer_start=0,
    stopper_patience=15,
    basepath=None,
    log_dir="None",
    device="cpu",
    dataloader_fn="datasets.white_noise_loader",
    model_fn="models.cnn.BasicEncoder.build_initial",
    seed=None,
    config=None,
    wandb_log=True,
    multiretinal=False,
):
    if config is None:
        config = global_config
    # if dataloader_config['crop'] is None:
    #     dataloader_config['crop'] = config['big_crops'][str(dataloader_config['retina_index'] + 1).zfill(2)]

    assert model_config["retina_index"] == dataloader_config["retina_index"]
    if model_config["retina_index"] not in range(0, 5):
        model_config["retina_index"] = None
        dataloader_config["retina_index"] = None
        print("loading data for all retinas")
    assert model_config["retina_index"] == dataloader_config["retina_index"]

    if regularizer_start > 0:
        model_config[
            "directory_prefix"
        ] = f'{model_config["directory_prefix"]}_rs_{regularizer_start}'

    if isinstance(model_config['hidden_channels'], str):
        if model_config["hidden_channels"].isdigit():
            model_config["hidden_channels"] = int(model_config["hidden_channels"])
        else:
            model_config["hidden_channels"] = model_config["hidden_channels"].split("x")
            if directory_prefix == "input_temporal":
                directory_prefix = "input_temporal_varied_channels"
            else:
                directory_prefix = "varied_channels"
    elif isinstance(model_config['hidden_channels'], int):
        model_config['hidden_channels'] = (model_config['hidden_channels'], ) * model_config["layers"]

    assert len(model_config["hidden_channels"]) == model_config["layers"]
    if model_config["spatial_input_kern"] is not None:
        model_config["input_kern"] = (
            dataloader_config["num_of_frames"],
            model_config["spatial_input_kern"],
            model_config["spatial_input_kern"],
        )

    else:
        model_config["input_kern"] = (
            dataloader_config["num_of_frames"],
            model_config["input_kern"],
            model_config["input_kern"],
        )

    if (dataloader_config["oracle_correlation_threshold"] is not None) and (
        dataloader_config["oracle_correlation_threshold"] > 0
    ):
        directory_prefix = (
            f'{directory_prefix}_ct_{model_config["oracle_correlation_threshold"]:.2f}'
        )
    if (dataloader_config["explainable_variance_threshold"] is not None) and (
        dataloader_config["explainable_variance_threshold"] > 0
    ):
        directory_prefix = f'{directory_prefix}_ev_{model_config["explainable_variance_threshold"]:.2f}'

    assert (
        dataloader_config["oracle_correlation_threshold"]
        == model_config["oracle_correlation_threshold"]
    )
    assert (
        model_config["explainable_variance_threshold"]
        == dataloader_config["explainable_variance_threshold"]
    )

    if "input_temporal" in directory_prefix:
        model_config["hidden_kern"] = (
            1,
            model_config["hidden_kern"],
            model_config["hidden_kern"],
        )
    elif model_config["spatial_hidden_kern"] is not None:
        model_config["hidden_kern"] = (
            dataloader_config["num_of_hidden_frames"],
            model_config["spatial_hidden_kern"],
            model_config["spatial_hidden_kern"],
        )
    else:
        model_config["hidden_kern"] = (
            dataloader_config["num_of_hidden_frames"],
            model_config["hidden_kern"],
            model_config["hidden_kern"],
        )

    if basepath is None:
        basepath = Path(__file__).parent.parent

    if log_dir == "None":
        log_dir = Path(__file__).parent.parent

    os.listdir(f"{basepath}/data")
    print("basepath", basepath)

    data_type = config["data_type"]

    dataset_fn = dataloader_fn
    dataloader_config = dict(dataloader_config)

    dataloaders = builder.get_data(
        dataset_fn=dataset_fn, dataset_config=dataloader_config
    )

    hk_str = "_hk_" if model_config["layers"] > 1 else ""
    stimulus_seed_str = (
        ""
        if dataset_fn == "datasets.white_noise_loader"
        else f'_{dataloader_config["stimulus_seed"]}'
    )
    first_session_ID = list((dataloaders["train"].keys()))[0]
    print(first_session_ID)
    a_dataloader = dataloaders["train"][first_session_ID]
    debug_output = a_dataloader.dataset.__getitem__(3)
    # inputs, targets = next(iter(a_dataloader))

    inputs, targets = debug_output[0], debug_output[1]

    max_coordinate = None
    img_h = inputs.shape[-2]
    img_w = inputs.shape[-1]

    size = inputs.shape[1], img_h, img_w

    if dataloader_config["retina_index"] is None:
        retina_str = "all"
    else:
        retina_str = str(dataloader_config["retina_index"] + 1).zfill(2)
    hdt_str = 'hdt_'+'-'.join([str(x) for x in model_config['hidden_temporal_dilation']]) if model_config['layers'] > 1 else ''
    hd_str = 'hd_'+'-'.join([str(x) for x in model_config['hidden_spatial_dilation']]) if model_config['layers'] > 1 else ''

    nm_string = 'wn' if dataset_fn == "datasets.white_noise_loader" else 'nm'

    model_dir = f'models/{directory_prefix}_cnn/{data_type}/retina{retina_str}/cell_{dataloader_config["cell_index"]}/readout_{readout}/gmp_{gmp}/lr_{lr:.4f}_l_{dataloader_config["num_of_layers"]}_ch_{model_config["hidden_channels"]}_t_{dataloader_config["num_of_frames"]}_bs_{dataloader_config["batch_size"]}_tr_{dataloader_config["num_of_trials_to_use"]}_ik_{"x".join([str(x) for x in model_config["input_kern"]])}{hk_str}{"x".join([str(x) for x in model_config["hidden_kern"]]) if model_config["layers"] > 1 else ""}_g_{model_config["gamma_input"]:.4f}_gt_{model_config["gamma_temporal"]:.4f}_l1_{model_config["l1"]:.4f}_l2_{model_config["l2"]:.4f}_sg_{model_config["init_sigma"]}_d_{int(model_config["spatial_dilation"])}_dt_{int(model_config["temporal_dilation"])}_{hd_str}_{hdt_str}_p_{int(model_config["padding"])}_bn_{int(model_config["batch_norm"])}_s_{int(model_config["subsample"])}norm_{int(dataloader_config["normalize_responses"])}_fn_{int(model_config["final_nonlinearity"])}{stimulus_seed_str}_h_{img_h}_w_{img_w}_{nm_string}'
    print(model_dir)

    optimizer_config = dict(lr=lr, weight_decay=model_config["l2"])

    print(dataloaders)

    # debug output

    # uncomment in case you want to crop around the receptive field
    # receptive_field = np.load(
    #     f'{basepath}/data/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_25.npy')[
    #     cell]
    # receptive_field = a_dataloader.dataset.transform(receptive_field)
    # temporal_variances = np.var(receptive_field, axis=0)
    # max_coordinate = np.unravel_index(np.argmax(temporal_variances), (img_h, img_w))

    # else:
    #     size = dataloader_config['num_of_frames'], img_h, img_w

    print("input shape:", size)
    seed = model_seed if seed is None else seed
    model_fn = eval(model_fn)
    print("model fn: ", model_fn)
    model = builder.get_model(
        model_fn, {"config_dict": model_config}, seed=seed, dataloaders=dataloaders
    )

    model = model.double()
    model = model.to(device)

    loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    loss_function_config = dict(
        loss_function=nn.PoissonNLLLoss, log_input=False, reduction="sum"
    )

    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    optimizer_config = dict(optimizer=optim.Adam, **optimizer_config)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, verbose=True, min_lr=1e-6
    )
    scheduler_config = dict(
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        mode="max",
        patience=5,
        verbose=True,
    )

    weight_dir = f"{log_dir}/{model_dir}/weights/seed_{seed}"
    stats_dir = f"{log_dir}/{model_dir}/stats/seed_{seed}"
    config_dir = f"{log_dir}/{model_dir}/config/"

    config_dict = dict(
        epochs=epochs,
        base_path=basepath,
        config=config,
        img_w=img_w,
        img_h=img_h,
        model_dir="/".join(model_dir.split("/")[:-1]),
        log_dir=log_dir,
        size=size,
        scheduler_config=scheduler_config,
        optimizer_config=optimizer_config,
        loss_function_config=loss_function_config,
        model_config=model_config,
        dataloader_config=dataloader_config,
        model_name=model_dir.split("/")[-1],
    )

    Path(weight_dir).mkdir(exist_ok=True, parents=True)
    Path(stats_dir).mkdir(exist_ok=True, parents=True)
    Path(config_dir).mkdir(exist_ok=True, parents=True)

    with open(f"{config_dir}/config.yaml", "w") as file:
        yaml.dump(config_dict, file)
    if wandb_log:
        wandb.config.update({"seed": seed})
    if multiretinal:
        n_iterations = len(LongCycler(dataloaders["train"]))
        train_multiretinal(
            epochs=epochs,
            data_keys=dataloaders["train"].keys(),
            dataloaders=dataloaders,
            dataset_config=dataloader_config,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            device=device,
            model_dir=model_dir,
            weight_dir=weight_dir,
            stats_dir=stats_dir,
            stopper_patience=stopper_patience,
            regularizer_start=regularizer_start,
            size=size,
            max_coordinate=max_coordinate,
            img_h=img_h,
            img_w=img_w,
            config_dict=config_dict,
            wandb_log=wandb_log,
            n_iterations=n_iterations,
        )
    else:
        score, correlations = train(
            epochs=epochs,
            dataloaders=dataloaders,
            dataset_config=dataloader_config,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            device=device,
            model_dir=model_dir,
            weight_dir=weight_dir,
            stats_dir=stats_dir,
            stopper_patience=stopper_patience,
            regularizer_start=regularizer_start,
            size=size,
            max_coordinate=max_coordinate,
            img_h=img_h,
            img_w=img_w,
            config_dict=config_dict,
            wandb_log=wandb_log,
        )

    return score


def train_ln(
    dataset_fn,
    dataloader_config,
    model_config,
    dataset_config_dict,
    model_dir,
    device,
    max_coordinate,
    img_h,
    img_w,
    wandb_log,
    config,
    filter_type,
    cuda,
    performance,
    nm=False
):
    dataloaders = builder.get_data(dataset_fn, dataloader_config)
    print(dataloaders)

    first_session_ID = list((dataloaders["train"].keys()))[0]
    print(first_session_ID)
    a_dataloader = dataloaders["train"][first_session_ID]

    img_h = int(
        (img_h - a_dataloader.dataset.crop[0] - a_dataloader.dataset.crop[1])
        / dataloader_config["subsample"]
    )
    img_w = int(
        (img_w - a_dataloader.dataset.crop[2] - a_dataloader.dataset.crop[3])
        / dataloader_config["subsample"]
    )
    if not nm:
        sta = a_dataloader.dataset.transform(model_config["sta"])
    else:
        sta = model_config['sta']
    if isinstance(sta, tuple):
        model_config["input_shape"] = sta[0].shape
        model = FactorizedModel(**model_config)
    else:
        model_config['input_shape'] = sta.shape[1:]
        model = Model(**model_config)


    if cuda:
        model = model.to(device)
    model = model.double()

    loss_function = config["loss_function"](**config["loss_function_config"])

    optimizer = config["optimizer"](model.parameters(), **config["optimizer_config"])

    scheduler = config["scheduler"](optimizer=optimizer, **config["scheduler_config"])

    weight_dir = f'{config["log_dir"]}/{config["model_dir"]}/weights/seed_{model_seed}/'
    stats_dir = f'{config["log_dir"]}/{config["model_dir"]}/stats/seed_{model_seed}/'
    config_dir = f'{config["log_dir"]}/{config["model_dir"]}/config/'

    # train_loss_dir = f'{basepath}/{model_dir}/losses/train/'
    # valid_loss_dir = f'{basepath}/{model_dir}/losses/valid/'

    Path(weight_dir).mkdir(exist_ok=True, parents=True)
    Path(stats_dir).mkdir(exist_ok=True, parents=True)
    Path(config_dir).mkdir(exist_ok=True, parents=True)

    max_avg_valid_corr = -1
    nm_str = 'nm' if nm else 'wn'

    with open(f"{config_dir}/config.yaml", "w") as file:
        yaml.dump(config, file)
    if wandb_log:
        wandb.init(
            config=config,
            project=f"ln-model-{dataset_config_dict['data_type']}-{filter_type}_{nm_str}",
            entity="retinal-circuit-modeling",
        )

    score, correlations = train(
        epochs=config["epochs"],
        dataloaders=dataloaders,
        dataset_config=dataloader_config,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        device=device,
        model_dir=model_dir,
        weight_dir=weight_dir,
        stats_dir=stats_dir,
        stopper_patience=config["scheduler_config"]["patience"],
        regularizer_start=config["regularizer_start"],
        size=model_config['input_shape'],
        max_coordinate=max_coordinate,
        img_h=img_h,
        img_w=img_w,
        config_dict=dataset_config_dict,
        wandb_log=wandb_log,
        fancy_nonlin=model_config["fancy_nonlin"],
        learn_filter=model_config["learn_filter"],
        performance=performance,
    )
    print(score, correlations)


def train_multiretinal(
    data_keys,
    epochs,
    dataloaders,
    dataset_config,
    config_dict,
    model,
    optimizer,
    loss_function,
    scheduler,
    device,
    model_dir,
    weight_dir,
    stats_dir,
    stopper_patience,
    size,
    n_iterations,
    max_coordinate=None,
    img_h=150,
    img_w=200,
    regularizer_start=0,
    wandb_log=True,
    fancy_nonlin=False,
    learn_filter=True,
    start_epoch=0,
    continued_training=False,
):
    retina_valid_correlations = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_train_correlations = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_valid_losses = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_train_losses = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_penalties = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_valid_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_true_train_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
    retina_true_valid_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
    max_avg_valid_corr = -1

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()

        epoch_retina_valid_correlations = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_train_correlations = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_valid_losses = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_train_losses = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_penalties = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_valid_prediction_variances = {f"0{x + 1}": [] for x in range(0, 5)}
        epoch_retina_true_train_prediction_variances = {
            f"0{x + 1}": [] for x in range(0, 5)
        }
        epoch_retina_true_valid_prediction_variances = {
            f"0{x + 1}": [] for x in range(0, 5)
        }
        if learn_filter or fancy_nonlin:
            for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders["train"])),
                total=n_iterations,
                desc="Epoch {}".format(epoch),
            ):
                # loss_scale = np.sqrt(len(dataloaders[data_key].dataset) / images.shape[0]) if scale_loss else 1.0
                # print(batch_no, data_key)
                images = data[0]
                responses = data[1]
                responses = responses.transpose(1, 2).to(device).contiguous()
                responses = torch.flatten(responses, start_dim=0, end_dim=1)
                images = images.to(device)
                loss, penalty, prediction_variance, corr = train_step(
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
                    data_key=data_key,
                )
                epoch_retina_train_losses[data_key].append(float(loss.item()))
                epoch_retina_penalties[data_key].append([x.item() for x in penalty])
                epoch_retina_prediction_variances[data_key].append(
                    float(prediction_variance)
                )
                epoch_retina_train_correlations[data_key].append(
                    np.mean(corr.detach().cpu().numpy()[0])
                )
                if epoch == 0:
                    epoch_retina_true_train_prediction_variances[data_key].append(
                        variance_of_predictions(responses)
                    )
            if epoch == 0:
                retina_true_train_prediction_variances[data_key] = np.mean(
                    epoch_retina_true_train_prediction_variances[data_key]
                )
            avg_train_corr = []
            avg_train_loss = []
            for key in data_keys:
                retina_train_correlations[key].append(
                    np.mean(epoch_retina_train_correlations[key], axis=0)
                )
                retina_train_losses[key].append(np.mean(epoch_retina_train_losses[key]))
                retina_penalties[key].append(
                    np.mean(epoch_retina_penalties[key], axis=0)
                )
                retina_prediction_variances[key].append(
                    np.mean(epoch_retina_prediction_variances[key])
                )
                print("")
                avg_train_loss.append(retina_train_losses[key][-1])
                avg_train_corr.append(retina_train_correlations[key][-1])
                print(
                    f"avg train correlation {key}:", retina_train_correlations[key][-1]
                )
                print(
                    f"avg train loss {key}: ",
                    retina_train_losses[key][-1],
                    f"penalty {key}:",
                    retina_penalties[key][-1],
                )
                print(
                    f"variances {key}:",
                    retina_prediction_variances[key][-1],
                    f"actual variance {key}:",
                    retina_true_train_prediction_variances,
                )
                if wandb_log:
                    wandb.log(
                        {
                            f"loss/train/{key}": retina_train_losses[key][-1],
                            f"correlation/train/{key}": retina_train_correlations[key][
                                -1
                            ],
                            f"spatial_penalty/train/ {key}": retina_penalties[key][-1][
                                0
                            ],
                            f"temporal_penalty/train/{key}": retina_penalties[key][-1][
                                1
                            ],
                            f"l1_penalty/train/{key}": retina_penalties[key][-1][2],
                            f"variance/train/{key}": retina_prediction_variances[key][
                                -1
                            ],
                            f"variance/true_train/{key}": retina_true_train_prediction_variances[
                                key
                            ],
                        },
                        step=epoch,
                    )
            print("")
            print("total avg train correlation:", np.mean(avg_train_corr))
            print("total avg train loss:", np.mean(avg_train_loss))
            if wandb_log:
                wandb.log(
                    {
                        "loss/train/avg": np.mean(avg_train_loss),
                        "correlation/train/avg": np.mean(avg_train_corr),
                    },
                    step=epoch,
                )
                if fancy_nonlin:
                    wandb.log(
                        {
                            "a": model.nonlin.a.detach().cpu().item(),
                            "b": model.nonlin.b.detach().cpu().item(),
                            "w": model.nonlin.w.detach().cpu().item(),
                        },
                        step=epoch
                    )

        model.eval()

        with torch.no_grad():
            for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders["validation"]))
            ):
                images, responses = data[0], data[1]
                images = images.double().to(device)
                responses = responses.transpose(1, 2)
                responses = (
                    torch.flatten(responses, start_dim=0, end_dim=1)
                    .to(device)
                    .contiguous()
                )
                output = model_step(
                    images=images,
                    model=model,
                    max_coordinate=max_coordinate,
                    rf_size=size,
                    h=img_h,
                    w=img_w,
                    data_key=data_key,
                )
                valid_prediciton_variance = variance_of_predictions(output)
                corr = correlation(output, responses, 1e-12)
                valid_loss = loss_function(output, responses)
                if device != "cpu":
                    epoch_retina_valid_correlations[data_key].append(
                        np.mean(corr.detach().cpu().numpy()[0])
                    )
                    epoch_retina_valid_losses[data_key].append(
                        float(valid_loss.detach().cpu().numpy())
                    )
                else:
                    epoch_retina_valid_correlations[data_key].append(
                        np.mean(corr.detach().numpy()[0])
                    )
                    epoch_retina_valid_losses[data_key].append(
                        float(valid_loss.detach().numpy())
                    )
                epoch_retina_valid_prediction_variances[data_key].append(
                    valid_prediciton_variance
                )
                if epoch == 0:
                    retina_true_valid_prediction_variances[data_key].append(
                        variance_of_predictions(responses)
                    )

            print(model_dir)
            avg_valid_corr = []
            avg_valid_loss = []
            single_correlations = {}
            for key in data_keys:
                if epoch == 0:
                    retina_true_valid_prediction_variances[key] = np.mean(
                        retina_true_valid_prediction_variances[key]
                    )
                retina_valid_losses[key].append(np.mean(epoch_retina_valid_losses[key]))
                retina_valid_prediction_variances[key].append(
                    np.mean(epoch_retina_valid_prediction_variances[key])
                )
                single_correlations[key] = np.mean(
                    epoch_retina_valid_correlations[key], axis=0
                )
                retina_valid_correlations[key].append(single_correlations[key])

                avg_valid_corr.append(single_correlations[key])
                print(f"avg valid corr {key}:", single_correlations[key])

                print(f"avg valid loss {key}:", retina_valid_losses[key][-1])
                avg_valid_loss.append(retina_valid_losses[key][-1])

                print(
                    f"avg valid variance {key}:",
                    retina_valid_prediction_variances[key][-1],
                ),
                print(
                    f"true prediction variance {key}:",
                    retina_true_valid_prediction_variances[key],
                )
                print()
                if wandb_log:
                    wandb.log(
                        {
                            f"correlation/valid/{key}": single_correlations[key],
                            f"loss/valid/{key}": retina_valid_losses[key][-1],
                            f"variance/valid/{key}": retina_valid_prediction_variances[
                                key
                            ][-1],
                            f"variance/true_valid/{key}": retina_true_valid_prediction_variances[
                                key
                            ],
                        },
                        step=epoch,
                    )
            print(
                "total avg valid correlation:", np.mean(avg_valid_corr), avg_valid_corr
            )
            print("total avg valid loss:", np.mean(avg_valid_loss))
            avg_valid_corr = np.mean(avg_valid_corr)

            print()
            if wandb_log:
                wandb.log(
                    {
                        "loss/valid/avg": np.mean(avg_valid_loss),
                        "correlation/valid/avg": avg_valid_corr,
                    },
                    step=epoch,
                )

            if avg_valid_corr > max_avg_valid_corr:
                patience = stopper_patience
                print(
                    f"Saving so far best model:{avg_valid_corr}, previous best:{max_avg_valid_corr}"
                )
                max_avg_valid_corr = avg_valid_corr
                save_checkpoint(
                    epoch,
                    model=model,
                    path=f"{weight_dir}/best_model.m",
                    optimizer=optimizer,
                    valid_corr=avg_valid_corr,
                )
            else:
                patience -= 1
                if patience == 0:
                    break

            scheduler.step(np.mean(avg_valid_corr))

        with torch.no_grad():
            np_correlations = np.array(list(retina_valid_correlations.values()))
            np_train_correlations = np.array(list(retina_train_correlations.values()))
            np_train_losses = np.array(list(retina_train_losses.values()))
            np_valid_losses = np.array(list(retina_valid_losses.values()))
            np_penalties = np.array(list(retina_penalties.values()))
            np_variances = np.array(list(retina_prediction_variances.values()))
            np_valid_variances = np.array(
                list(retina_valid_prediction_variances.values())
            )
            np.save(f"{stats_dir}/penalties", np_penalties)
            np.save(f"{stats_dir}/variances", np_variances)
            np.save(f"{stats_dir}/correlations", np_correlations)
            np.save(f"{stats_dir}/train_correlations", np_train_correlations)
            np.save(f"{stats_dir}/train_losses", np_train_losses)
            np.save(f"{stats_dir}/valid_losses", np_valid_losses)
            np.save(f"{stats_dir}/valid_variances", np_valid_variances)


def train(
    epochs,
    dataloaders,
    dataset_config,
    config_dict,
    model,
    optimizer,
    loss_function,
    scheduler,
    device,
    model_dir,
    weight_dir,
    stats_dir,
    stopper_patience,
    size,
    max_coordinate=None,
    img_h=150,
    img_w=200,
    regularizer_start=0,
    wandb_log=True,
    fancy_nonlin=False,
    learn_filter=True,
    start_epoch=0,
    continued_training=False,
    performance="validation",
):
    if continued_training:
        correlations = list(np.load(f"{stats_dir}/correlations.npy"))
        train_correlations = list(np.load(f"{stats_dir}/train_correlations.npy"))
        valid_losses = list(np.load(f"{stats_dir}/valid_losses.npy"))
        train_losses = list(np.load(f"{stats_dir}/train_losses.npy"))
        penalties = list(np.load(f"{stats_dir}/penalties.npy"))
        prediction_variances = list(np.load(f"{stats_dir}/variance.npy"))
        valid_prediction_variances = list(np.load(f"{stats_dir}/valid_variances.npy"))
        true_train_prediction_variances = []
        true_valid_prediction_variances = []

    else:
        correlations = []
        train_correlations = []
        valid_losses = []
        train_losses = []
        penalties = []
        prediction_variances = []
        valid_prediction_variances = []
        true_train_prediction_variances = []
        true_valid_prediction_variances = []

    patience = stopper_patience
    max_avg_valid_corr = -1
    if wandb_log:
        wandb.watch(model, log_freq=100, log_graph=True)
    if not learn_filter and not fancy_nonlin:
        epochs = 1
    if performance == "validation":
        perf_str = "valid"
    else:
        perf_str = performance
    # wandb.config(config_dict)
    for epoch in range(start_epoch, epochs + start_epoch):
        print(f"Epoch {epoch}")
        model.train()

        epoch_train_correlations = []
        epoch_valid_correlations = []
        epoch_train_losses = []
        epoch_penalties = []
        epoch_valid_losses = []
        epoch_prediction_variances = []
        epoch_valid_prediction_variances = []

        # if dataset_config['conv3d']:
        print(size)
        rf_size = size
        # rf_size = (20, 20)
        # else:
        #     rf_size = size
        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(
        #             wait=1,
        #             warmup=1,
        #             active=20),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./runs/model'),
        #         record_shapes=True,
        #         profile_memory=False,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        #         with_stack=True
        # ) as p:
        # stas = torch.zeros((dataloaders['train'][str(dataset_config['retina_index'] + 1).zfill(2)].dataset.n_neurons, 40, rf_size[0], rf_size[1]), dtype=torch.float16, device=device)
        # spike_counts = torch.zeros((dataloaders['train'][str(dataset_config['retina_index'] + 1).zfill(2)].dataset.n_neurons), device=device)
        if performance == "test":
            all_predictions = []
            all_responses = []
        if learn_filter or fancy_nonlin:
            for i, (images, responses) in enumerate(
                tqdm(
                    dataloaders["train"][
                        str(dataset_config["retina_index"] + 1).zfill(2)
                    ]
                )
            ):
                # print(i, 'got i')
                responses = responses.transpose(1, 2).to(device).contiguous()
                responses = torch.flatten(responses, start_dim=0, end_dim=1)
                images = images.to(device)
                # continue
                # continue
                # for j, response in enumerate(responses.squeeze()):
                #     if response > 0:
                #         stas[j] = torch.add(stas[j], (response*images).squeeze())
                #     if i % 1000 == 0:
                #         print(j, 'max_sta', torch.max(stas[j]))
                #         print(j, 'min_sta', torch.min(stas[j]))
                #         print('spike_counts', spike_counts[j])
                #         print()
                # spike_counts = torch.add(spike_counts, responses.squeeze())
                #
                # normed_stas = torch.zeros(stas.shape)
                # print('spike counts: ', spike_counts)
                # for i in range(stas.shape[0]):
                #     stas[i] = stas[i] / spike_counts[i]
                #     print(i, 'max_sta', torch.max(stas[i]))
                #     print(i, 'min_sta', torch.min(stas[i]))
                #     normed_stas[i] = stas[i].cpu().numpy()/np.linalg.norm(stas[i].cpu().numpy())
                # stas = np.asarray(stas)
                # for cell in range(dataloaders['train'][str(dataset_config['retina_index'] + 1).zfill(2)].dataset.n_neurons):
                # show_sta(stas[cell].cpu().numpy(), cell)
                # print(f'saving cell {cell}')
                #     show_sta(normed_stas[cell].cpu().numpy(), cell, vmin=np.min(normed_stas[cell]), vmax=np.max(normed_stas[cell]))
                #     np.save(f'/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/stas/cell_data_01_NM_stas_40_cell_{cell}.npy', stas[cell].cpu().numpy(), )
                # break
                # if 'blabla':
                #     continue
                loss, penalty, prediction_variance, corr = train_step(
                    images=images.to(device).contiguous(),
                    responses=responses.to(device).contiguous(),
                    model=model,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    rf_size=rf_size,
                    max_coordinate=max_coordinate,
                    h=img_h,
                    w=img_w,
                    penalty_coef=1,
                    regularizer_start=regularizer_start,
                    epoch=epoch,
                )
                # print(i, 'processed i')
                epoch_train_losses.append(float(loss.item()))
                epoch_penalties.append([x.item() for x in penalty])
                epoch_prediction_variances.append(float(prediction_variance))
                epoch_train_correlations.append(np.mean(corr.detach().cpu().numpy()[0]))
                # p.step()
                if epoch == 0:
                    true_variance = variance_of_predictions(responses)
                    true_train_prediction_variances.append(true_variance)
                # print(i, f'gonna get {i}+1')
            if epoch == 0:
                true_train_variance = sum(true_train_prediction_variances) / len(
                    true_train_prediction_variances
                )
                true_train_prediction_variances = true_train_variance

            epoch_penalties = np.array(epoch_penalties)
            train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
            penalties.append(np.mean(epoch_penalties, axis=0))
            prediction_variances.append(
                sum(epoch_prediction_variances) / len(epoch_prediction_variances)
            )
            train_corr = np.mean(epoch_train_correlations, axis=0)
            train_correlations.append(train_corr)
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
                if fancy_nonlin:
                    wandb.log(
                        {
                            "a": model.nonlin.a.detach().cpu().item(),
                            "b": model.nonlin.b.detach().cpu().item(),
                            "w": model.nonlin.w.detach().cpu().item(),
                        }
                    )
            # np_train_correlations = np.array(train_correlations)
            # np_train_losses = np.array(train_losses)
            # np.save(f'{stats_dir}/train_correlations', np_train_correlations)
            # np.save(f'{stats_dir}/train_losses', np_train_losses)

        # dataloaders['train'][str(dataset_config['retina_index'] + 1).zfill(2)].dataset.purge_cache()
        # save_checkpoint(epoch, model=model, path=f'{weight_dir}/test_model.m', optimizer=optimizer,
        #                 valid_corr=train_corr)
        # continue
        model.eval()

        with torch.no_grad():
            for images, responses in tqdm(
                dataloaders[performance][
                    str(dataset_config["retina_index"] + 1).zfill(2)
                ]
            ):
                images = images.double().to(device).contiguous()
                responses = responses.transpose(1, 2)
                responses = (
                    torch.flatten(responses, start_dim=0, end_dim=1)
                    .to(device)
                    .contiguous()
                )
                output = model_step(
                    images=images,
                    model=model,
                    max_coordinate=max_coordinate,
                    rf_size=rf_size,
                    h=img_h,
                    w=img_w,
                )
                valid_prediciton_variance = variance_of_predictions(output)
                corr = correlation(output, responses, 1e-12)
                valid_loss = loss_function(output, responses)
                if device != "cpu":
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
                if performance == "test":
                    all_predictions.append(output)
                    all_responses.append(responses.squeeze(-1))
            if epoch == 0:
                true_valid_prediction_variances = sum(
                    true_valid_prediction_variances
                ) / len(true_valid_prediction_variances)
            print(model_dir)
            # print('avg train loss: ', train_losses[-1])
            print(f"max batch {perf_str} corr:", max(epoch_valid_correlations))
            single_correlations = np.mean(epoch_valid_correlations, axis=0)
            if performance == "test":
                all_outputs = torch.cat(all_predictions)
                all_responses = torch.cat(all_responses)
                single_correlations = correlation(
                    all_outputs[:, 0], all_responses, 1e-12
                ).item()
            print(f"avg {perf_str} corr:", single_correlations)
            print(
                f"avg {perf_str} loss:",
                sum(epoch_valid_losses) / len(epoch_valid_losses),
            )
            print(f"max {perf_str} loss:", max(epoch_valid_losses))
            print(
                f"avg {perf_str} variance:",
                sum(epoch_valid_prediction_variances)
                / len(epoch_valid_prediction_variances),
                "true prediction variance:",
                true_valid_prediction_variances,
            )
            if wandb_log:
                wandb.log(
                    {
                        f"correlation/{perf_str}": single_correlations,
                        "loss/valid": sum(epoch_valid_losses) / len(epoch_valid_losses),
                        f"variance/{perf_str}": sum(epoch_valid_prediction_variances)
                        / len(epoch_valid_prediction_variances),
                        f"variance/true_{perf_str}": true_valid_prediction_variances,
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
            valid_prediction_variances.append(
                sum(epoch_valid_prediction_variances)
                / len(epoch_valid_prediction_variances)
            )
            scheduler.step(single_correlations)

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
            if performance == "validation":
                np.save(f"{stats_dir}/correlations", np_correlations)
            else:
                np.save(f"{stats_dir}/correlations_{perf_str}", np_correlations)
            np.save(f"{stats_dir}/train_correlations", np_train_correlations)
            np.save(f"{stats_dir}/train_losses", np_train_losses)
            np.save(f"{stats_dir}/{perf_str}_losses", np_valid_losses)
            np.save(f"{stats_dir}/{perf_str}_variances", np_valid_variances)
    if wandb_log:
        wandb.finish()
    return np.max(np_correlations), correlations
