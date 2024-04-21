import yaml
import matplotlib.pyplot as plt

from dynamic.datasets.stas import (
    unnormalize_source_grid,
    calculate_position_before_convs,
)
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
    model_name,
    model_dir,
    model_fn,
    seeds=None,
    device="cuda",
    config_dict=None,
    nm=False,
    data_dir=None,
    data_type="salamander",
):
    """
    Get models for all seeds for a given model name
    Args:
    model_name: str
        Name of the model
    model_dir: str
        Directory where the model is stored
    model_fn: str
        Function to build the model
    seeds: list
        List of seeds for which to get the models
    device: str
        Device on which to run the model
    config_dict: dict
        Configuration dictionary for the model
    nm: bool
        Whether to use the NM model
    data_dir: str
        Directory where the data is stored
    data_type: str
        Type of the data
    Returns:
    models: list
        List of models for all seeds
    seeds: list
        List of seeds for which the models were obtained
    """
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
                    data_dir=data_dir,
                )
            else:
                _, model, _ = get_model_and_dataloader_for_nm(
                    directory=model_dir,
                    filename=model_name,
                    model_fn=model_fn,
                    data_dir=data_dir,
                    device=device,
                    seed=seed,
                    config_dict=config_dict,
                    data_type=data_type,
                )
            models.append(model.double())
        return models, seeds
    else:
        raise ValueError(
            f"No models logged in location "
            f"{os.path.join(model_dir, model_name)} for any of the "
            f"following seeds {seeds}"
        )


def get_center_coordinates(model, cell_index):
    """
    Get the center coordinates of the receptive field of a given cell
    Args:
    model: nn.Module
        Model for which to get the center coordinates
    cell_index: int
        Index of the cell for which to get the center coordinates
    Returns:
    input_position: torch.Tensor
        Input position of the cell
    size: torch.Tensor
        Size of the cell
    """
    learned_position = (
        model.readout.grid[0, cell_index, 0, :].clone().detach().cpu()
    )
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


def config_init(config_dict, data_dir, directory, filename):
    """
    Initialize the configuration for the model
    Args:
    config_dict: dict
        Configuration dictionary for the model
    data_dir: str
        Directory where the data is stored
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    Returns:
    model_config: dict
        Configuration dictionary for the model
    model_related_config: dict
        Configuration dictionary for the model
    home_dir: str
        Directory where the data is stored
    config_dict: dict
        Configuration dictionary for the model
    """
    if config_dict is None:
        print("ATTENTION, loading global config, i.e. the one for salamander")
        config_dict = global_config
    if data_dir is None:
        home_dir = home
    else:
        home_dir = data_dir
    if "/" not in filename:
        with open(
            f"/{directory}/{filename}/config/config.yaml", "r"
        ) as config_file:
            model_related_config = yaml.unsafe_load(config_file)
    else:
        filename_abbr = filename.split("/")[0]
        with open(
            f"/{directory}/{filename_abbr}/config/config.yaml", "r"
        ) as config_file:
            model_related_config = yaml.unsafe_load(config_file)

    model_config = model_related_config["model_config"]
    model_config["readout_type"] = "isotropic"
    if "readout" in model_config.keys():
        del model_config["readout"]
    if data_dir is None:
        model_config["data_dir"] = home
    else:
        model_config["data_dir"] = data_dir
    model_config["padding"] = 0
    model_config["base_path"] = f"{home_dir}"

    if "spatial_dilation" not in model_config.keys():
        model_config["spatial_dilation"] = 1
        model_config["temporal_dilation"] = 1

    if model_config["spatial_input_kern"] is None:
        return None, None, None
    return model_config, model_related_config, home_dir, config_dict


def build_model_and_dataloder(
    dataset_fn,
    dataloader_config,
    model_fn,
    model_config,
    directory,
    filename,
    home_dir,
    device,
    seed,
    freeze,
    flip_sta=False,
):
    """
    Build the model and the dataloader
    Args:
    dataset_fn: str
        Function to build the dataset
    dataloader_config: dict
        Configuration dictionary for the dataloader
    model_fn: str
        Function to build the model
    model_config: dict
        Configuration dictionary for the model
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    home_dir: str
        Directory where the data is stored
    device: str
        Device on which to run the model
    seed: int
        Seed for the model
    freeze: bool
        Whether to freeze the model
    flip_sta: bool
        Whether to flip the STA
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    """
    dataloaders = builder.get_data(dataset_fn, dataloader_config)
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
    if "flip_sta" not in model_config:
        model_config["flip_sta"] = flip_sta

    model = model.double()
    if freeze:
        model.eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False

    return dataloaders, model


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
    freeze=True,
    fancy_nonlin=False,
):
    """
    Get the model and the dataloader for the given configuration
    Args:
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    model_fn: str
        Function to build the model
    device: str
        Device on which to run the model
    data_dir: str
        Directory where the data is stored
    test: bool
        Whether to run the model in test mode
    seed: int
        Seed for the model
    config_dict: dict
        Configuration dictionary for the model
    data_type: str
        Type of the data
    freeze: bool
        Whether to freeze the model
    fancy_nonlin: bool
        Whether to use the fancy nonlinearity
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    config: dict
        Configuration dictionary for the model
    """
    model_config, config, home_dir, config_dict = config_init(
        config_dict=config_dict,
        directory=directory,
        data_dir=data_dir,
        filename=filename,
    )
    if fancy_nonlin:
        model_config["fancy_nonlin"] = True
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

    print(f"data_dir: {home_dir}")
    dataloader_config[
        "neuronal_data_dir"
    ] = f"{home_dir}/data/{data_type}_data/responses/"
    dataloader_config["config"] = config_dict
    dataloader_config["crop"] = model_config["config"]["big_crops"]["01"]
    if dataloader_config["retina_index"] is not None:
        dataloader_config["config"]["big_crops"]["01"] = (model_config)[
            "config"
        ]["big_crops"]["01"]
        dataloader_config["crop"] = model_config["config"]["big_crops"]["01"]

    model_fn = eval(model_fn)

    dataloaders, model = build_model_and_dataloder(
        dataset_fn=dataset_fn,
        dataloader_config=dataloader_config,
        model_fn=model_fn,
        model_config=model_config,
        directory=directory,
        filename=filename,
        home_dir=home_dir,
        device=device,
        seed=seed,
        freeze=freeze,
    )
    return dataloaders, model, config


def check_hyperparam_for_layers(hyperparameter, layers):
    """
    Check the hyperparameters for the given layers
    Args:
    hyperparameter: int or list
        Hyperparameter for the model
    layers: int
        Number of layers in the model
    Returns:
    hyperparameter: list
        List of hyperparameters for the given layers
    """
    if isinstance(hyperparameter, (list, tuple)):
        assert len(hyperparameter) == layers, (
            f"Hyperparameter list should have same length "
            f"{len(hyperparameter)} as layers {layers}"
        )
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
    """
    Get the model and the dataloader for the given configuration
    Args:
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    model_fn: str
        Function to build the model
    device: str
        Device on which to run the model
    data_dir: str
        Directory where the data is stored
    test: bool
        Whether to run the model in test mode
    seed: int
        Seed for the model
    config_dict: dict
        Configuration dictionary for the model
    data_type: str
        Type of the data
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    config: dict
        Configuration dictionary for the model
    """
    if config_dict is None:
        config_dict = global_config
    if data_dir is None:
        home_dir = home
    else:
        home_dir = data_dir
    if seed is None:
        seed = model_seed
    with open(
        f"/{directory}/{filename}/config/config.yaml", "r"
    ) as config_file:
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
    new_dataloader_config["all_image_path"] = (
        f"/user/vystrcilova//"
        f'{dataloader_config_from_model["config"]["image_path"]}'
    )
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
    seed=None,
    test=False,
    config_dict=None,
    data_type="salamander",
    dataloader_config=None,
    stimulus_seed=None,
    fixation_file=None,
    num_of_trials_to_use=None,
    freeze=True,
    fancy_nonlin=False,
):
    """
    Get the model and the dataloader for the given configuration
    Args:
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    model_fn: str
        Function to build the model
    device: str
        Device on which to run the model
    data_dir: str
        Directory where the data is stored
    seed: int
        Seed for the model
    test: bool
        Whether to run the model in test mode
    config_dict: dict
        Configuration dictionary for the model
    data_type: str
        Type of the data
    dataloader_config: dict
        Configuration dictionary for the dataloader
    stimulus_seed: int
        Seed for the stimulus
    fixation_file: str
        File with the fixation
    num_of_trials_to_use: int
        Number of trials to use
    freeze: bool
        Whether to freeze the model
    fancy_nonlin: bool
        Whether to use the fancy nonlinearity
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    config: dict
        Configuration dictionary for the model
    """
    print(data_dir, "data dir")
    model_config, config, home_dir, config_dict = config_init(
        config_dict, data_dir, directory=directory, filename=filename
    )
    if "config" not in model_config.keys():
        model_config["config"] = config_dict
    else:
        config_dict = model_config["config"]
    if fancy_nonlin:
        model_config["fancy_nonlin"] = True

    dataset_fn = "datasets.frame_movie_loader"
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
        dataloader_config["config"]["fixation_file"][
            f"0{model_config['retina_index']}"
        ] = fixation_file
    if stimulus_seed is not None:
        dataloader_config["stimulus_seed"] = stimulus_seed

    dataloader_config["batch_size"] = 16
    print(f"data_dir: {home}")
    dataloader_config[
        "neuronal_data_dir"
    ] = f"{home_dir}/data/{data_type}_data/responses/"
    dataloader_config["config"] = config_dict
    if num_of_trials_to_use is not None:
        dataloader_config["num_of_trials_to_use"] = num_of_trials_to_use

    model_fn = eval(model_fn)
    dataloaders, model = build_model_and_dataloder(
        dataset_fn=dataset_fn,
        dataloader_config=dataloader_config,
        model_fn=model_fn,
        model_config=model_config,
        directory=directory,
        filename=filename,
        home_dir=home_dir,
        device=device,
        seed=seed,
        freeze=freeze,
    )

    return dataloaders, model, config


def get_nm_model_and_dataloader_for_wn(
    config_dict,
    data_dir,
    seed,
    directory,
    filename,
    device="cuda",
    test=False,
    data_type="salamander",
    model_fn="models.BasicEncoder.build_trained",
    freeze=True,
    fancy_nonlin=False,
):
    """
    Get the model and the dataloader for the given configuration
    Args:
    config_dict: dict
        Configuration dictionary for the model
    data_dir: str
        Directory where the data is stored
    seed: int
        Seed for the model
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    device: str
        Device on which to run the model
    test: bool
        Whether to run the model in test mode
    data_type: str
        Type of the data
    model_fn: str
        Function to build the model
    freeze: bool
        Whether to freeze the model
    fancy_nonlin: bool
        Whether to use the fancy nonlinearity
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    config: dict
        Configuration dictionary for the model
    """
    model_config, config, home_dir, config_dict = config_init(
        config_dict=config_dict,
        data_dir=data_dir,
        filename=filename,
        directory=directory,
    )
    config["base_path"] = home_dir

    dataloader_config_from_model = config["dataloader_config"]
    new_dataloader_config = {
        "train_image_path": f"{data_dir}/data/{data_type}_data/non_repeating_stimuli_flipped/",
        "test_image_path": f"{data_dir}/data/{data_type}_data/repeating_stimuli_flipped/",
        "neuronal_data_dir": f"{data_dir}/data/{data_type}_data/responses/",
        "conv3d": True,
        "movie_like": False,
        "flip": True,
        "time_chunk_size": 70,
        "batch_size": 16,
        "num_of_trials_to_use": 21,
    }
    # if test:

    print(f"data_dir: {home_dir}")
    new_dataloader_config["config"] = config_dict
    new_dataloader_config["crop"] = model_config["config"]["big_crops"]["01"]

    new_dataloader_config["config"]["big_crops"]["01"] = model_config[
        "config"
    ]["big_crops"]["01"]
    new_dataloader_config["crop"] = model_config["config"]["big_crops"]["01"]
    # dataloader_config_from_model['batch_size'] = 1
    # dataloader_config_from_model['time_chunk_size'] = 1
    if "readout" in model_config.keys():
        del model_config["readout"]
    for key, value in dataloader_config_from_model.items():
        if key not in [
            "all_image_path",
            "frame_file",
            "img_dir_name",
            "full_img_w",
            "full_img_h",
            "padding",
            "stimulus_seed",
            "basepath",
            "hard_coded",
            "conv3d",
            "movie_like",
            "config",
            "neuronal_data_dir",
            "time_chunk_size",
            "batch_size",
            "num_of_trials_to_use",
        ]:
            new_dataloader_config[key] = value
            print(f"copied {key} with value {value} to new dataloader")

    dataloaders, model, _ = get_model_and_dataloader_for_nm(
        directory,
        filename,
        model_fn=model_fn,
        device=device,
        data_dir=data_dir,
        seed=seed,
        test=test,
        config_dict=config_dict,
        data_type=data_type,
        dataloader_config=dataloader_config_from_model,
        stimulus_seed=None,
        num_of_trials_to_use=21,
        freeze=freeze,
        fancy_nonlin=fancy_nonlin,
    )
    dataset_fn = "datasets.white_noise_loader"

    dataloaders = builder.get_data(dataset_fn, new_dataloader_config)

    return dataloaders, model, config


def get_param_for_all_models(
    directory, files, model_seed, param, model_file_subset="l_3"
):
    """
    Get the parameter for all models
    Args:
    directory: str
        Directory where the model is stored
    files: list
        List of files for which to get the parameter
    model_seed: int
        Seed for the model
    param: str
        Parameter for which to get the value
    model_file_subset: str
        Subset of the model files
    Returns:
    file_correlations: dict
        Dictionary with the parameter values for all models
    """
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
            elif (param == "spatial_dilation") or (
                param == "temporal_dilation"
            ):
                if param in model_config.keys():
                    param_value = model_config[param]
                else:
                    param_value = 1
            elif param == "lr":
                param_value = optimizer_config[param]
            else:
                param_value = model_config[param]
            # print(file, f'max corr: {max(correlations)},
            # param_value: {param_value}')
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
    """
    Plot the responses vs the predictions
    Args:
    all_responses: list
        List of all responses
    all_predictions: list
        List of all predictions
    cell: int
        Index of the cell
    cell_name: str
        Name of the cell
    save_file: str
        File to save the plot
    max_lenght: int
        Maximum length of the plot
    max_cc: float
        Maximum correlation coefficient
    start_index: int
        Starting index of the plot
    Returns:
        None
    """
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
                np.max(all_responses[: min(max_lenght, len(all_predictions))])
                * 1.2,
                np.max(
                    all_predictions[: min(max_lenght, len(all_predictions))]
                )
                * 1.2,
            ),
        ),
    )
    # plt.ylim(0, 3)
    # plt.xlim(0, 315)
    plt.title(f"Cell {cell_name} cc: {np.round(max_cc, 2)}")
    plt.savefig(save_file)
    plt.show()


def get_possible_seeds(model_name, model_dir):
    """
    Get the possible seeds for a given model
    Args:
    model_name: str
        Name of the model
    model_dir: str
        Directory where the model is stored
    Returns:
    seeds: list
        List of seeds for the given model
    """
    seeds = []
    if os.path.isdir(os.path.join(model_dir, model_name, "weights")):
        seed_files = os.listdir(os.path.join(model_dir, model_name, "weights"))
        for file in seed_files:
            if "seed" in file:
                seeds.append(int(file.split("_")[1]))

    return seeds


def get_model_and_dataloader_based_on_setting(
    setting,
    directory,
    filename,
    model_fn,
    device,
    data_dir,
    performance,
    seed,
    data_type,
    config_dict,
    dataloader_config=None,
    stimulus_seed=None,
    fixation_file=None,
    freeze=True,
    fancy_nonlin=False,
):
    """
    Get the model and the dataloader for the given setting
    Args:
    setting: str
        Setting for the model
    directory: str
        Directory where the model is stored
    filename: str
        Name of the model
    model_fn: str
        Function to build the model
    device: str
        Device on which to run the model
    data_dir: str
        Directory where the data is stored
    performance: str
        Performance for the model
    seed: int
        Seed for the model
    data_type: str
        Type of the data
    config_dict: dict
        Configuration dictionary for the model
    dataloader_config: dict
        Configuration dictionary for the dataloader
    stimulus_seed: int
        Seed for the stimulus
    fixation_file: str
        File with the fixation
    freeze: bool
        Whether to freeze the model
    fancy_nonlin: bool
        Whether to use the fancy nonlinearity
    Returns:
    dataloaders: dict
        Dataloader for the model
    model: nn.Module
        Model for the given configuration
    config: dict
        Configuration dictionary for the model
    """
    assert setting in ["nm", "wn", "nm_for_wn", "wn_for_nm"]
    if setting == "nm":
        dataloaders, model, config = get_model_and_dataloader_for_nm(
            directory,
            filename,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            test=performance == "test",
            seed=seed,
            data_type=data_type,
            dataloader_config=dataloader_config,
            config_dict=config_dict,
            stimulus_seed=stimulus_seed,
            fixation_file=fixation_file,
            freeze=freeze,
            fancy_nonlin=fancy_nonlin,
        )
    elif setting == "wn":
        dataloaders, model, config = get_model_and_dataloader(
            directory,
            filename,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            config_dict=config_dict,
            test=performance == "test",
            seed=seed,
            data_type=data_type,
            freeze=freeze,
            fancy_nonlin=fancy_nonlin,
        )
    elif setting == "wn_for_nm":
        assert stimulus_seed is not None and fixation_file is not None
        dataloaders, model, config = get_wn_model_and_dataloader_for_nm(
            directory,
            filename,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            test=performance == "test",
            seed=seed,
            config_dict=config_dict,
            data_type=data_type,
            fixation_file=fixation_file,
            stimulus_seed=stimulus_seed,
            fancy_nonlin=fancy_nonlin,
            freeze=freeze,
        )
    else:
        dataloaders, model, config = get_nm_model_and_dataloader_for_wn(
            directory=directory,
            filename=filename,
            config_dict=config_dict,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            test=performance == "test",
            seed=seed,
            data_type=data_type,
            freeze=freeze,
            fancy_nonlin=fancy_nonlin,
        )

    return dataloaders, model, config
