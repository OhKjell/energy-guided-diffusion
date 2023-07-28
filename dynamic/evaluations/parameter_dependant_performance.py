import os.path

import matplotlib.pyplot as plt
import pandas
import numpy as np
import yaml

from utils.global_functions import home


def visualize_parameters_performance(
    parameters,
    search_dir: str,
    search_indices: list,
    filters: dict,
    x_scale=None,
    y_scale=None,
):
    df = pandas.DataFrame()
    for search_index in search_indices:
        file = os.path.join(search_dir, f"search_{search_index}", "search_results.tsv")
        results = pandas.read_csv(file, sep="\t", index_col=None)
        df = df.append(results, ignore_index=True)
    title = ""
    for param, value in filters.items():
        df = df[df[param] == value]
        title = f"{title} {param}:{value}"
    if len(parameters) == 2:
        visualize_two_parameters(
            parameters, df, title=title, x_scale=x_scale, y_scale=y_scale
        )
    elif len(parameters) == 1:
        visualize_single_parameter(parameters, df, title=title, x_scale=x_scale)


def visualize_two_parameters(parameters, df, title=None, x_scale=None, y_scale=None):
    plt.scatter(
        df[parameters[0]].astype(float).tolist(),
        df[parameters[1]].astype(float).tolist(),
        c=df["corr"].tolist(),
        cmap="coolwarm",
    )
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[1])
    plt.colorbar()
    plt.title(title)
    if x_scale is not None:
        plt.xscale(x_scale)
    if y_scale is not None:
        plt.yscale(y_scale)
    plt.show()


def collect_results(directory):
    results = pandas.DataFrame(
        columns=[
            "corr",
            "model_dir",
            "lr",
            "retina_index",
            "batch_size",
            "crop",
            "dataset_seed",
            "num_of_frames",
            "time_chunk",
            "num_of_layers",
            "explainable_variance_threshold",
            "oracle_correlation_threshold",
            "normalize_responses",
            "hidden_channels",
            "input_kern",
            "hidden_kern",
            "core_nonlinearity",
            "x_shift",
            "y_shift",
            "stride",
            "bias",
            "independent_bn_bias",
            "input_regularizer",
            "gamma_input",
            "gamma_temporal",
            "l1",
            "padding",
            "batch_norm",
            "readout_type",
            "final_nonlinearity",
            "readout_nonlin",
            "init_mu_range",
            "init_sigma",
            "readout_bias",
            "gmp",
            "batch_scale_norm",
            "num_of_trials",
        ]
    )
    for file in os.listdir(directory):
        with open(
            os.path.join(directory, file, "config", "config.yaml"), "r"
        ) as config_file:
            config = yaml.unsafe_load(config_file)
        if os.path.isfile(os.path.join(directory, file, "stats", "correlations.npy")):
            correlations = np.load(
                os.path.join(directory, file, "stats", "correlations.npy")
            )
        else:
            correlations = [0]
        row = {
            "corr": np.max(correlations),
            "model_dir": config["model_dir"],
            "lr": config["optimizer_config"]["lr"],
            "retina_index": config["dataloader_config"]["retina_index"],
            "batch_size": config["dataloader_config"]["batch_size"],
            "crop": config["dataloader_config"]["crop"],
            "dataset_seed": config["dataloader_config"]["seed"],
            "num_of_frames": config["dataloader_config"]["num_of_frames"],
            "time_chunk": config["dataloader_config"]["time_chunk_size"],
            "num_of_layers": config["dataloader_config"]["num_of_layers"],
            "explainable_variance_threshold": config["dataloader_config"][
                "explainable_variance_threshold"
            ],
            "oracle_correlation_threshold": config["dataloader_config"][
                "oracle_correlation_threshold"
            ],
            "normalize_responses": config["dataloader_config"]["normalize_responses"],
            "hidden_channels": config["model_config"]["hidden_channels"],
            "input_kern": config["model_config"]["input_kern"],
            "hidden_kern": config["model_config"]["hidden_kern"],
            "core_nonlinearity": config["model_config"]["core_nonlinearity"],
            "x_shift": config["model_config"]["elu_xshift"],
            "y_shift": config["model_config"]["elu_yshift"],
            "stride": config["model_config"]["stride"],
            "bias": config["model_config"]["bias"],
            "independent_bn_bias": config["model_config"]["independent_bn_bias"],
            "input_regularizer": config["model_config"]["input_regularizer"],
            "gamma_input": config["model_config"]["gamma_input"],
            "gamma_temporal": config["model_config"]["gamma_temporal"],
            "l1": config["model_config"]["l1"],
            "padding": config["model_config"]["padding"],
            "batch_norm": config["model_config"]["batch_norm"],
            "readout_type": config["model_config"]["readout_type"],
            "final_nonlinearity": config["model_config"]["final_nonlinearity"],
            "readout_nonlin": config["model_config"]["readout_nonlinearity"],
            "init_mu_range": config["model_config"]["init_mu_range"],
            "init_sigma": config["model_config"]["init_sigma"],
            "readout_bias": config["model_config"]["readout_bias"],
            "gmp": config["model_config"]["use_grid_mean_predictor"],
            "num_of_trials": config["dataloader_config"]["num_of_trials_to_use"],
        }
        results = results.append(row, ignore_index=True)
    results.to_csv(os.path.join(directory, "search_x.txv"), sep="\t")
    return results


def get_model_single_param(model_config, param):
    return model_config[param]


def get_model_config_and_corr(directory, file, seed):
    with open(
        os.path.join(directory, file, "config", "config.yaml"), "r"
    ) as config_file:
        config = yaml.unsafe_load(config_file)
    if os.path.isfile(
        os.path.join(directory, file, "stats", f"seed_{seed}", "correlations.npy")
    ):
        correlation = np.load(
            os.path.join(directory, file, "stats", f"seed_{seed}", "correlations.npy")
        )
    else:
        correlation = [0]
    return config, correlation


def get_model_1st_layer_spatial_reach(model_config):
    if "spatial_dilation" in model_config.keys():
        dilation = model_config["spatial_dilation"]
    else:
        dilation = 1
    if model_config["spatial_input_kern"] is None:
        return 0
    print(model_config["spatial_input_kern"][0], dilation)
    return model_config["spatial_input_kern"][0] * dilation


def get_model_1st_layer_temp_reach(model_config):
    if "temporal_dilation" in model_config.keys():
        dilation = model_config["temporal_dilation"]
    else:
        dilation = 1
    return model_config["temporal_input_kern"] * dilation


def get_model_hidden_spat_reach(model_config):
    if "hidden_spatial_dilation" in model_config.keys():
        dilation = model_config["hidden_spatial_dilation"]
        if isinstance(dilation, (list, tuple)) and (len(dilation) != 0):
            dilation = dilation
    else:
        dilation = [1]*(model_config["layers"]-1)
        print(dilation, model_config[''])
    if model_config["spatial_hidden_kern"] is not None:
        reach = 0
        for i in range(model_config["layers"]-1):
            reach += (model_config["spatial_hidden_kern"][0]-1) * dilation[i]
        return reach
    else:
        return 0


def get_model_overall_spatial_reach(model_config):
    if "spatial_dilation" in model_config.keys():
        dilation = model_config["spatial_dilation"]
        if isinstance(dilation, (list, tuple)):
            dilation = dilation[0]
    else:
        dilation = 1
    if model_config["spatial_input_kern"] is None:
        reach = model_config["input_kern"][1]
    else:
        reach = (model_config["spatial_input_kern"][0]-1) * dilation + 1
    reach += get_model_hidden_spat_reach(model_config)
    return reach


def get_model_temp_reach(model_config):
    if "temporal_dilation" in model_config.keys():
        dilation = model_config["temporal_dilation"]
    else:
        dilation = 1
    reach = (model_config["temporal_input_kern"]-1) * dilation + 1
    layers = model_config["layers"]
    for l in range(1, layers):
        reach += get_model_hidden_temp_reach(model_config)
    return reach


def get_model_hidden_temp_reach(model_config):
    if "hidden_temporal_dilation" in model_config.keys():
        dilation = model_config["hidden_temporal_dilation"]
        if isinstance(dilation, (list, tuple)):
            dilation = dilation[0]
    else:
        dilation = 1
    return (
        (model_config["temporal_hidden_kern"]-1) * dilation
        if model_config["temporal_hidden_kern"] is not None
        else 0
    )


def visualize_single_parameter(parameter, df, title=None, x_scale=None):
    plt.scatter(df[parameter[0]].astype(float).tolist(), df["corr"].tolist())
    plt.xlabel(parameter[0])
    plt.ylabel("correlation")
    if x_scale is not None:
        plt.xscale(x_scale)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    retina = 0
    # spatial_reach_dict = get_param_for_all_models(
    #     f'{home}/models/factorized_ev_0.15_cnn/salamander/retina{retina + 1}/cell_None/readout_isotropic/gmp_0/',
    #     files=None,
    #     model_seed=None, param='overall_spat_reach')
    pass
    # df = collect_results(f'{home}/models/basic_ev_0.15_cnn/retina1/cell_None/readout_isotropic/gmp_0/')
    # visualize_parameters_performance(['gamma_input', 'gamma_temporal'],
    #                                  search_dir=f'{home}/models/basic_ev_0.15_cnn/retina1/cell_None/readout_isotropic/',
    #                                  search_indices=['x'],
    #                                  filters={'num_of_layers': 1,
    #                                           'hidden_channels': 16,
    #                                          },
    #                                  x_scale='log',
    #                                  y_scale='log')
    # visualize_parameters_performance(['gamma_temporal'],
    #                                  search_dir=f'{home}/models/basic_ev_0.15_cnn/retina1/cell_None/readout_isotropic/',
    #                                  search_indices=['x'], filters={'num_of_layers': 1,
    #                                                                              'hidden_channels': 8,
    #                                                                              },
    #                                  x_scale='log'
    #                                  )
