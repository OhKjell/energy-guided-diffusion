import yaml
import matplotlib.pyplot as plt

from dynamic.datasets.stas import unnormalize_source_grid, calculate_position_before_convs
from dynamic.evaluations.parameter_dependant_performance import (
    get_model_config_and_corr,
    get_model_temp_reach,
    get_model_1st_layer_spatial_reach,
    get_model_hidden_spat_reach,
    get_model_1st_layer_temp_reach,
    get_model_overall_spatial_reach,
)
from dynamic.training.regularizers import TimeLaplaceL23d
from dynamic.utils.global_functions import home, model_seed, global_config
import torch
import numpy as np
from nnfabrik import builder
import dynamic.models
import os


def get_seed_model_versions(
    model_name, model_dir, model_fn, seeds=None, device="cuda", config_dict=None, nm=False, data_dir=None,
        data_type='salamander'
):
    if seeds is None:
        seeds = get_possible_seeds(model_name=model_name, model_dir=model_dir)
    if len(seeds) > 0:
        models = []
        for seed in seeds:
            print(seed)
            if not nm:
                _, model, _ = get_model_and_dataloader(
                    directory=model_dir,
                    filename=model_name,
                    model_fn=model_fn,
                    device=device,
                    seed=seed,
                    config_dict=config_dict,
                    data_type=data_type,
                    data_dir=data_dir
                )
            else:
                _, model, _ = get_model_and_dataloader_for_nm(directory=model_dir,
                    filename=model_name,
                    model_fn=model_fn,
                    data_dir=data_dir,
                    device=device,
                    seed=seed,
                    config_dict=config_dict,
                    data_type=data_type
                                                              )
            models.append(model.double())
        return models, seeds
    else:
        raise ValueError(
            f"No models logged in location {os.path.join(model_dir, model_name)} for any of the "
            f"following seeds {seeds}"
        )


def get_center_coordinates(model, cell_index):
    learned_position = model.readout.grid[0, cell_index, 0, :].clone().detach().cpu()
    learned_position = unnormalize_source_grid(
        learned_position.detach().cpu(),
        model.core.get_output_shape(
            model.config_dict["in_shapes_dict"][
                f"0{model.config_dict['retina_index'] +1 }"
            ]
        ),
    )
    input_position, size = calculate_position_before_convs(
        model_config=model.config_dict, center=learned_position
    )
    return input_position, size


def get_model_and_dataloader(
    directory,
    filename,
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    test=False,
    seed=None,
    config_dict=None,
    data_type="salamander",
):
    if config_dict is None:
        config_dict = global_config
    if data_dir is None:
        home_dir = home
    else:
        home_dir = data_dir
    if seed is None:
        seed = model_seed
    model = torch.load(
        f"{directory}/{filename}/weights/seed_{seed}/best_model.m",
        map_location=torch.device(device),
    )
    with open(f"/{directory}/{filename}/config/config.yaml", "r") as config_file:
        config = yaml.unsafe_load(config_file)

    model_config = config["model_config"]
    model_config["readout_type"] = "isotropic"
    if "config" not in model_config.keys():
        model_config["config"] = config_dict
    else:
        config_dict = model_config["config"]
    dataset_fn = "datasets.white_noise_loader"
    config["base_path"] = home_dir
    dataloader_config = config["dataloader_config"]
    dataloader_config[
        "train_image_path"
    ] = f'{home_dir}/{model_config["config"]["training_img_dir"]}'
    dataloader_config[
        "test_image_path"
    ] = f'{home_dir}/{model_config["config"]["test_img_dir"]}'
    if test:
        dataloader_config["time_chunk_size"] = 1
    dataloader_config['time_chunk_size'] = 30
    dataloader_config["batch_size"] = 8
    # dataloader_config['cell_index'] = cell_index
    print(f"data_dir: {home_dir}")
    dataloader_config[
        "neuronal_data_dir"
    ] = f"{home_dir}/data/{data_type}_data/responses/"
    dataloader_config["config"] = config_dict
    dataloader_config['crop'] = model_config['config']['big_crops']['01']
    # if dataloader_config["crop"] is not None:
    dataloader_config["config"]["big_crops"]["01"] = model_config['config']['big_crops']['01']
    dataloader_config["crop"] = model_config['config']['big_crops']['01']
    # dataloader_config
    # dataloader_config['path'] = 'vystrcilova/retinal_circuit_modeling/data'c
    dataloaders = builder.get_data(dataset_fn, dataloader_config)

    # model_fn = 'models.multi_retina_regular_cnn_model'

    if "readout" in model_config.keys():
        del model_config["readout"]
    if data_dir is None:
        model_config["data_dir"] = home
    else:
        model_config["data_dir"] = data_dir

    if "spatial_dilation" not in model_config.keys():
        model_config["spatial_dilation"] = 1
        model_config["temporal_dilation"] = 1
    model_fn = eval(model_fn)
    if model_config["spatial_input_kern"] is None:
        return None, None, None
    model = builder.get_model(
        model_fn,
        model_config={
            "model_dir": directory,
            "model_name": filename,
            "data_dir": home_dir,
            "device": device,
        },
        dataloaders=dataloaders,
        seed=seed,
    )
    model_config[
        "core.temporal_regulazirer.laplace.filter"
    ] = TimeLaplaceL23d().laplace.filter

    # model.load_state_dict(model_dict)
    model = model.double()
    return dataloaders, model, config


def check_hyperparam_for_layers(hyperparameter, layers):
    if isinstance(hyperparameter, (list, tuple)):
        assert (
            len(hyperparameter) == layers
        ), f"Hyperparameter list should have same length {len(hyperparameter)} as layers {layers}"
        return hyperparameter
    elif isinstance(hyperparameter, int):
        return (hyperparameter,) * layers


def get_wn_model_and_dataloader_for_nm(
    directory,
    filename,
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    test=False,
    seed=None,
    config_dict=None,
    data_type="salamander",
):
    if config_dict is None:
        config_dict = global_config
    if data_dir is None:
        home_dir = home
    else:
        home_dir = data_dir
    if seed is None:
        seed = model_seed
    with open(f"/{directory}/{filename}/config/config.yaml", "r") as config_file:
        model_related_config = yaml.unsafe_load(config_file)
    model_config = model_related_config["model_config"]
    model_config["readout_type"] = "isotropic"

    dataset_fn = "datasets.frame_movie_loader"
    model_related_config["base_path"] = home_dir
    dataloader_config_from_model = model_related_config["dataloader_config"]
    new_dataloader_config = {}
    new_dataloader_config["basepath"] = f"{home_dir}"
    new_dataloader_config[
        "img_dir_name"
    ] = f'{home}/{dataloader_config_from_model["config"]["image_path"]}'
    new_dataloader_config[
        "all_image_path"
    ] = f'/user/vystrcilova//{dataloader_config_from_model["config"]["image_path"]}'
    new_dataloader_config[
        "neuronal_data_dir"
    ] = f'{home_dir}/{dataloader_config_from_model["config"]["response_path"]}'

    model_config["base_path"] = f"{home_dir}"
    if test:
        new_dataloader_config["time_chunk_size"] = 1
        new_dataloader_config["batch_size"] = 8
    # dataloader_config['cell_index'] = cell_index
    print(f"data_dir: {home_dir}")
    new_dataloader_config[
        "neuronal_data_dir"
    ] = f"{home_dir}/data/{data_type}_data/responses/"
    new_dataloader_config["config"] = config_dict

    for key, value in dataloader_config_from_model.items():
        if key not in [
            "",
            "",
            "",
        ]:
            new_dataloader_config[key] = value

    # dataloader_config
    # dataloader_config['path'] = 'vystrcilova/retinal_circuit_modeling/data'c
    dataloaders = builder.get_data(dataset_fn, dataloader_config_from_model)

    # model_fn = 'models.multi_retina_regular_cnn_model'

    if "readout" in model_config.keys():
        del model_config["readout"]
    if data_dir is None:
        model_config["data_dir"] = home
    else:
        model_config["data_dir"] = data_dir
    model_config["padding"] = 0
    model_fn = eval(model_fn)
    model = builder.get_model(
        model_fn,
        model_config={
            "model_dir": directory,
            "model_name": filename,
            "data_dir": home_dir,
            "device": device,
        },
        dataloaders=dataloaders,
        seed=seed,
    )
    model_config[
        "core.temporal_regulazirer.laplace.filter"
    ] = TimeLaplaceL23d().laplace.filter
    # model.load_state_dict(model_dict)
    model = model.double()
    return dataloaders, model, config_dict


def get_model_and_dataloader_for_nm(
    directory,
    filename,
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    test=False,
    seed=None,
    config_dict=None,
    data_type="salamander",
    dataloader_config=None,
    stimulus_seed=None,
    fixation_file=None,
    num_of_trials_to_use=None,
    stimulus_dir=None
):
    if config_dict is None:
        config_dict = global_config
    if data_dir is None:
        home_dir = home
    else:
        home_dir = data_dir
    if seed is None:
        seed = model_seed
    with open(f"/{directory}/{filename}/config/config.yaml", "r") as config_file:
        config = yaml.unsafe_load(config_file)
    model_config = config["model_config"]
    model_config["readout_type"] = "isotropic"

    if "config" not in model_config.keys():
        model_config["config"] = config_dict
    else:
        config_dict = model_config["config"]
    dataset_fn = "dynamic.datasets.frame_movie_loader"
    config["base_path"] = home_dir
    if dataloader_config is None:
        dataloader_config = config["dataloader_config"]
    dataloader_config["basepath"] = f"{home_dir}"
    dataloader_config[
        "img_dir_name"
    ] = f'/{home_dir}/{dataloader_config["config"]["image_path"]}'
    dataloader_config[
        "all_image_path"
    ] = f'/{home_dir}/{dataloader_config["config"]["image_path"]}'
    dataloader_config[
        "neuronal_data_dir"
    ] = f'{home}/{dataloader_config["config"]["response_path"]}'
    if fixation_file is not None:
        dataloader_config["config"]["fixation_file"][f"0{model_config['retina_index']}"] = fixation_file
    if stimulus_seed is not None:
        dataloader_config["stimulus_seed"] = stimulus_seed

    model_config["base_path"] = f"{home_dir}"
    # if test:
    # dataloader_config['time_chunk_size'] = 1
    dataloader_config["batch_size"] = 16

    # dataloader_config['cell_index'] = cell_index
    print(f"data_dir: {home}")
    dataloader_config[
        "neuronal_data_dir"
    ] = f"{home_dir}/dynamic_data/data/{data_type}_data/responses/"
    dataloader_config["config"] = config_dict
    if num_of_trials_to_use is not None:
        dataloader_config["num_of_trials_to_use"] = num_of_trials_to_use

    # dataloader_config
    # dataloader_config['path'] = 'vystrcilova/retinal_circuit_modeling/data'c
    print(dataloader_config[fi])
    dataloaders = builder.get_data(dataset_fn, dataloader_config)

    # model_fn = 'models.multi_retina_regular_cnn_model'

    if "readout" in model_config.keys():
        del model_config["readout"]
    if data_dir is None:
        model_config["data_dir"] = home
    else:
        model_config["data_dir"] = data_dir
    model_config["padding"] = 0
    model_fn = eval(model_fn)
    print(home_dir)
    model = builder.get_model(
        model_fn,
        model_config={
            "model_dir": directory,
            "model_name": filename,
            "data_dir": home_dir,
            "device": device,
        },
        dataloaders=dataloaders,
        seed=seed,
    )
    model_config[
        "core.temporal_regulazirer.laplace.filter"
    ] = TimeLaplaceL23d().laplace.filter
    # model.load_state_dict(model_dict)
    model = model.double()
    return dataloaders, model, config


def get_param_for_all_models(
    directory, files, model_seed, param, model_file_subset="l_3"
):
    if files is None:
        files = os.listdir(directory)
    file_correlations = {}
    for file in files:
        if model_seed is None:
            seeds = get_possible_seeds(model_name=file, model_dir=directory)
        else:
            seeds = [model_seed]
        for seed in seeds:
            config, correlations = get_model_config_and_corr(
                directory=directory, file=file, seed=seed
            )
            model_config = config["model_config"]
            optimizer_config = config["optimizer_config"]
            # print(model_config.keys())
            if param == "overall_spat_reach":
                param_value = get_model_overall_spatial_reach(model_config)
            elif param == "overall_temp_reach":
                param_value = get_model_temp_reach(model_config)
            elif param == "layer1_spat_reach":
                print(file)
                param_value = get_model_1st_layer_spatial_reach(model_config)
                print()
            elif param == "hidden_spat_reach":
                param_value = get_model_hidden_spat_reach(model_config)
            elif param == "layer1_temp_reach":
                param_value = get_model_1st_layer_temp_reach(model_config)
            elif (param == "spatial_dilation") or (param == "temporal_dilation"):
                if param in model_config.keys():
                    param_value = model_config[param]
                else:
                    param_value = 1
            elif param == "lr":
                param_value = optimizer_config[param]
            else:
                param_value = model_config[param]
            # print(file, f'max corr: {max(correlations)}, param_value: {param_value}')
            file_correlations[file] = [np.max(correlations), param_value]
    return file_correlations


def plot_responses_vs_predictions(
    all_responses,
    all_predictions,
    cell,
    cell_name,
    save_file,
    max_lenght=500,
    max_cc=None,
    start_index=0,
):
    all_responses = np.concatenate(all_responses)
    all_predictions = np.concatenate(all_predictions)
    all_responses = all_responses[:, cell].flatten()
    all_predictions = all_predictions[:, cell].flatten()
    fig, ax = plt.subplots(figsize=(20, 8))
    plt.plot(
        [
            x + start_index
            for x in range(min(len(all_predictions), max_lenght + start_index))
        ],
        all_predictions[: min(max_lenght + start_index, len(all_predictions))],
        label="Predicted responses",
    )
    plt.plot(
        [
            x + start_index
            for x in range(min(len(all_predictions), max_lenght + start_index))
        ],
        all_responses[: min(max_lenght + start_index, len(all_predictions))],
        label="True responses",
        linestyle="--",
    )
    plt.legend()
    plt.ylim(
        0,
        max(
            1,
            max(
                np.max(all_responses[: min(max_lenght, len(all_predictions))]) * 1.2,
                np.max(all_predictions[: min(max_lenght, len(all_predictions))]) * 1.2,
            ),
        ),
    )
    # plt.ylim(0, 3)
    # plt.xlim(0, 315)
    plt.title(f"Cell {cell_name} cc: {np.round(max_cc, 2)}")
    plt.savefig(save_file)
    plt.show()


def get_possible_seeds(model_name, model_dir):
    seeds = []
    if os.path.isdir(os.path.join(model_dir, model_name, "weights")):
        seed_files = os.listdir(os.path.join(model_dir, model_name, "weights"))
        for file in seed_files:
            if "seed" in file:
                seeds.append(int(file.split("_")[1]))

    return seeds
