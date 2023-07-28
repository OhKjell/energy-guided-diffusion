import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os
import yaml

from datasets.stas import (
    get_rf_center_grid,
    calculate_pca_on_cropped_stas,
    normalize_source_grid,
    recalculate_positions_after_convs,
    get_rf_size,
    crop_cell_rf,
    get_receptive_field,
    visualize_weights,
    plot_all_stas,
    get_receptive_field_center,
)
from models.helper_functions import get_possible_seeds

from utils.global_functions import get_cell_names, home, global_config
import torch


def visualize_filters(
    model, layer, visualization_dir, max_corr, max_corr_epoch, v_min=0.08, v_max=0.08
):
    if layer == 0:
        conv_str = ""
    else:
        conv_str = f"_{layer}"
    weights = model[f"core.features.layer{layer}.conv{conv_str}.weight"]
    if (weights.shape[1] == 1) and (weights.shape[0] > 8):
        fig, ax = plt.subplots(
            8, int(weights.shape[0] / 8), figsize=(weights.shape[0], weights.shape[0])
        )
    else:
        fig, ax = plt.subplots(
            weights.shape[0],
            weights.shape[1],
            figsize=(weights.shape[1] * 2 + 2, weights.shape[0] * 2),
        )
    timed_frames = [[] for _ in range(weights.shape[2])]
    for out_channel in range(weights.shape[0]):
        for in_channel in range(weights.shape[1]):
            if weights.shape[1] == 1:
                if weights.shape[0] == 1:
                    frames = visualize_weights(
                        weights[out_channel, in_channel],
                        ax,
                        vmin=-v_min,
                        vmax=v_max,
                        weight_index=0,
                    )
                    axis = ax
                elif weights.shape[0] > 8:
                    axis = ax[out_channel % 8, math.floor(out_channel / 8)]
                    frames = visualize_weights(
                        weights[out_channel, in_channel],
                        ax[int(out_channel % 8), math.floor(out_channel / 8)],
                        vmin=v_min,
                        vmax=v_max,
                        weight_index=0,
                    )
                else:
                    frames = visualize_weights(
                        weights[out_channel, in_channel],
                        ax[out_channel],
                        vmin=v_min,
                        vmax=v_max,
                        weight_index=0,
                    )
                    axis = ax[out_channel]
            else:
                frames = visualize_weights(
                    weights[out_channel, in_channel],
                    ax[out_channel, in_channel],
                    vmin=v_min,
                    vmax=v_max,
                    weight_index=0,
                )
                axis = ax[out_channel, in_channel]
            axis.set_title(f"Out {out_channel} - In {in_channel}")
            axis.set_xticks([])
            axis.set_yticks([])

            for i, frame in enumerate(frames):
                timed_frames[i].append(frame[0])
    fig.suptitle(f"Layer {layer} (Max corr {max_corr:.2f} in epoch {max_corr_epoch}")
    fig.tight_layout()
    Path(f"{home}/{visualization_dir}/layer_{layer}").mkdir(exist_ok=True, parents=True)
    anim = animation.ArtistAnimation(
        fig, timed_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(f"{home}/{visualization_dir}/layer_{layer}/conv_filters.mp4")


def visualize_all_gaussian_readout(
    model,
    visualization_dir,
    readout_index=None,
    retina_index=None,
    spatial_str="",
    correlation_threshold=None,
    explainable_variance_threshold=None,
    config=None,
    img_h=150,
    img_w=200,
):
    if readout_index is None:
        for i in range(1, 6):
            if f"readout.0{i}._mu" in model.keys():
                Path(
                    f"{home}/{visualization_dir}/gaussian_readout/readout.0{i}/"
                ).mkdir(exist_ok=True, parents=True)
                visualize_gaussian_readout(
                    model,
                    f".0{i}",
                    visualization_dir,
                    retina_index=i - 1,
                    spatial_str=spatial_str,
                    explainable_variance_threshold=explainable_variance_threshold,
                    correlation_threshold=correlation_threshold,
                    data_dir=home,
                )
    else:
        if "readout._features" in model.keys():
            Path(f"{home}/{visualization_dir}/gaussian_readout/readout/").mkdir(
                exist_ok=True, parents=True
            )
            visualize_gaussian_readout(
                model,
                "",
                visualization_dir,
                retina_index=retina_index,
                spatial_str=spatial_str,
                explainable_variance_threshold=explainable_variance_threshold,
                correlation_threshold=correlation_threshold,
                data_dir=home,
                config=config,
                img_h=150,
                img_w=200,
            )


def visualize_gaussian_readout(
    model,
    readout_index,
    visualization_dir,
    retina_index,
    img_h=70,
    img_w=80,
    spatial_str="",
    correlation_threshold=None,
    explainable_variance_threshold=None,
    data_dir=home,
    file_suffix="_NC.mat.pickle",
    config=None,
):
    if config is None:
        config = global_config
    retina_index_str = f"0{retina_index + 1}"
    if f"readout{readout_index}._mu" in model.keys():
        mus = model[f"readout{readout_index}._mu"].numpy()[0, :, 0, :]
    else:
        mus = (
            model[f"readout.source_grid"].squeeze()
            @ model["readout.mu_transform.0.weight"].T
            + model["readout.mu_transform.0.bias"]
        )
        mus = mus.numpy()
    sigmas = model[f"readout{readout_index}.sigma"].numpy()[0, :, 0, 0]
    if f"readout{readout_index}.source_grid" in model.keys():
        source_grid = model[f"readout{readout_index}.source_grid"].numpy()
        core_output_shape, kernel_size = None, None
        # TODO: What is the core output shape here... When does this happen. I have no idea.
        #       Kind of seems like it should just never  happen
    else:
        output_h = img_h
        output_w = img_w
        layer = 0
        conv_str = ""
        kernel_size = []
        while (
            f"core.features.layer{layer}.conv{spatial_str}{conv_str}.weight"
            in model.keys()
        ):
            kernel = [0] + list(
                model[
                    f"core.features.layer{layer}.conv{spatial_str}{conv_str}.weight"
                ].shape[-2:]
            )
            kernel_size.append(kernel)
            layer += 1
            conv_str = f"_{layer}"
            output_h = output_h - kernel[1] + 1
            output_w = output_w - kernel[2] + 1

        core_output_shape = output_h, output_w
        source_grid = get_rf_center_grid(
            retina_index=retina_index,
            crop=config["big_crops"][retina_index_str],
            explainable_varinace_threshold=explainable_variance_threshold,
            correlation_threshold=correlation_threshold,
            data_dir=data_dir,
            config=config,
        )
        source_grid = recalculate_positions_after_convs(
            source_grid,
            kernel_size,
            img_h=img_h - sum(config["big_crops"][retina_index_str][:2]),
            img_w=img_w - sum(config["big_crops"][retina_index_str][2:]),
        )
        # source_grid = normalize_source_grid(source_grid, core_output_shape=core_output_shape)
    # bias = model[f'readout{readout_index}.bias'].numpy()
    features = model[f"readout{readout_index}._features"].numpy()[0, :, 0, :]
    if features.shape[0] <= 2:
        show_cells_in_feature_space(
            features,
            save_file=f"{home}/{visualization_dir}/gaussian_readout/all_cells_feature_space{readout_index}.png",
            dims=features.shape[0],
            retina_index_str=retina_index_str,
            correlation_threshold=correlation_threshold,
            explainable_variance_threshold=explainable_variance_threshold,
            config=config,
        )

    mus[:, 0] = mus[:, 0] * (core_output_shape[1] / 2) + (core_output_shape[1] / 2)
    mus[:, 1] = mus[:, 1] * (core_output_shape[0] / 2) + (core_output_shape[0] / 2)

    fig, ax = plt.subplots(1, 2, figsize=(15, 15), sharey="row")

    ax[0].imshow(np.zeros((core_output_shape[1], core_output_shape[0])), cmap="Blues")
    ax[0].set_title("Source grid")

    ax[1].imshow(np.zeros((core_output_shape[1], core_output_shape[0])), cmap="Blues")
    ax[1].set_title("Mu predictions")

    for cell in range(source_grid.shape[0]):
        ax[0].scatter(
            [(source_grid[cell][1])],
            [(source_grid[cell][0])],
            color="orange",
            marker="o",
            s=[10],
        )
        ax[0].annotate(
            str(cell), (int(source_grid[cell][1]), int(source_grid[cell][0]))
        )
        ax[1].scatter(
            [(mus[cell][1])], [(mus[cell][0])], color="orange", marker="o", s=[10]
        )
        ax[1].annotate(str(cell), (int(mus[cell][1]), int(mus[cell][0])))
        ax[0].axis("off")
        ax[1].axis("off")
    plt.savefig(
        f"{home}/{visualization_dir}/gaussian_readout/all_cells{readout_index}.png"
    )
    plt.show()
    cell_names_list = get_cell_names(
        retina_index,
        correlation_threshold=correlation_threshold
        if correlation_threshold is not None
        else 0,
        explained_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold is not None
        else 0,
        config=config,
    )
    crop = config["big_crops"][retina_index_str]
    # cell_rfs = np.load(
    #     f'/{data_dir}/data/cell_data_{retina_index_str}_NC_stas_cell_{cell_names_list[cell]}.npy')

    for cell in range(source_grid.shape[0]):
        cell_rf = np.load(
            f'/{data_dir}/data/{config["data_type"]}_data/stas/cell_data_{retina_index_str}_NC_stas_cell_{cell_names_list[cell]}.npy'
        )
        # cell_rf = cell_rfs[cell_names_list[cell]]
        if sum(crop) > 0:
            cell_rf = cell_rf[:, crop[0] : -(crop[1]), crop[2] : -(crop[3])]
        temporal_variances = np.var(cell_rf, axis=0)
        conv = SimpleConv(kernel_size)

        convolved_temporal_variances = conv(
            torch.unsqueeze(
                torch.unsqueeze(torch.tensor(temporal_variances), 0), 0
            ).double()
        )

        fig, ax = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}
        )
        plt.axis("off")
        ax[0].set_axis_off()
        ax[0].axis("off")
        ax0 = fig.add_subplot(
            2,
            1,
            1,
        )

        heatmap = convolved_temporal_variances.squeeze().detach().numpy()
        ax0.imshow(heatmap, cmap="coolwarm")
        ax0.axis("off")
        ax0.scatter(
            [mus[cell][0]],
            [mus[cell][1]],
            color="gold",
            marker="o",
            s=[10],
            label="learned position",
        )
        ax0.scatter(
            [source_grid[cell][1]],
            [source_grid[cell][0]],
            color="limegreen",
            marker="o",
            s=[10],
            label="intialized position",
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        plt.legend(fontsize="x-large")

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.imshow(
            features[:, cell].reshape(max(1, int(features.shape[0] / 16)), -1),
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
        )

        for j in range(features[:, cell].shape[0]):
            ax1.text(
                j % 16,
                math.floor(j / 16),
                f"{features[j, cell].item():.2f}",
                ha="center",
                va="center",
                color="black",
            )
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.title(f"Cell {cell_names_list[cell]}")
        plt.savefig(
            f"{home}/{visualization_dir}/gaussian_readout/readout{readout_index}/cell_{cell_names_list[cell]}.png"
        )
        plt.show()


class SimpleConv(torch.nn.Module):
    def __init__(self, kernel_sizes):
        super().__init__()
        self.conv = torch.nn.Sequential()
        for i, kernel_size in enumerate(kernel_sizes):
            conv_kernel = torch.nn.Conv2d(1, 1, kernel_size[1:])
            conv_kernel.weight = torch.nn.Parameter(
                torch.ones(conv_kernel.weight.shape).double(), requires_grad=False
            )
            self.conv.add_module(f"conv_{i}", conv_kernel)

    def forward(self, x):
        x = self.conv(x)
        return x


def show_cells_in_feature_space(
    features,
    save_file,
    retina_index_str,
    dims=2,
    correlation_threshold=None,
    explainable_variance_threshold=None,
    config=None,
):
    cell_names_list = get_cell_names(
        config=config,
        retina_index=int(retina_index_str) - 1,
        correlation_threshold=0
        if correlation_threshold is None
        else correlation_threshold,
        explained_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold is not None
        else 0,
    )
    cell_names_list = [x for x in range(features.shape[1])]
    if dims == 2:
        fig, ax = plt.subplots(figsize=(10, 10))

        plt.scatter(*features)
        for cell in range(features.shape[1]):
            plt.annotate(str(cell_names_list[cell]), features[:, cell])
    elif dims == 1:
        fig, ax = plt.subplots(figsize=(20, 10), frameon=False)
        # ax = plt.axes(frameon=False)
        y = np.ones(features.shape[1])
        ax.hlines(1, 0, np.max(features) * 1.1)
        ax.set_ylim(0.5, 1.5)
        ax.plot(features[0, :], y, "o", ms=6)
        ax.axes.get_yaxis().set_visible(False)
        y_positions = [
            (np.random.random() / 9) + 0.95 for x in range(features.shape[1])
        ]
        for cell in range(features.shape[1]):
            ax.annotate(
                str(cell_names_list[cell]), xy=(features[:, cell], y_positions[cell])
            )

    plt.savefig(save_file)
    plt.show()


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    # Sigma = Sigma @ Sigma.T
    Sigma = max(1e-3, Sigma) * np.eye(2)
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def visualize_readout(
    model, visualization_dir, retina_index=0, vmin=-0.1, vmax=0.1, config=None
):
    if config is None:
        config = global_config
    spatial_readout = model["readout.spatial"]
    feature_readout = model["readout.features"]
    cell_index = 0
    for cell in range(config["cell_numbers"][str(retina_index + 1).zfill(2)]):
        if cell not in config["exclude_cells"][str(retina_index + 1).zfill(2)]:
            receptive_field = np.load(
                f"{home}/data/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_25.npy"
            )[cell]
            crop = config["big_crops"][str(retina_index + 1).zfill(2)]
            receptive_field = receptive_field[
                :, crop[0] : -crop[1] + 1, crop[2] : -crop[3] + 1
            ]
            # receptive_field = a_dataloader.dataset.transform(receptive_field)
            temporal_variances = np.var(receptive_field, axis=0)
            max_coordinate = np.unravel_index(np.argmax(temporal_variances), (51, 61))

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(
                spatial_readout[cell_index], cmap="coolwarm", vmin=vmin, vmax=vmax
            )
            ax[1].imshow(
                np.array(feature_readout[cell_index]).reshape([-1, 1]),
                cmap="coolwarm",
                vmin=-0.3,
                vmax=0.3,
            )
            for j in range(feature_readout[cell_index].shape[0]):
                ax[1].text(
                    0,
                    j,
                    f"{feature_readout[cell_index, j].item():.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
            plt.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )
            plt.tick_params(
                axis="y",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                labelleft=False,
                labelright=False,
                left=False,
                right=False,
            )

            Path(f"{home}/{visualization_dir}/spxf_readout").mkdir(
                exist_ok=True, parents=True
            )
            ax[0].scatter(
                [max_coordinate[1]],
                [max_coordinate[0]],
                7,
                color="yellow",
                label="RF center from STA",
            )
            ax[0].set_title(f"Cell: {cell} spatial readout", pad=30)
            ax[0].legend(bbox_to_anchor=[0, 1.02], loc="lower left")
            ax[1].set_title("Feature readout")
            plt.savefig(f"{home}/{visualization_dir}/spxf_readout/cell_{cell}.png")
            cell_index += 1
            plt.show()


def visualize_mutli_channel_cnn(
    model_file,
    layers=2,
    vmin=-0.1,
    vmax=0.1,
    corr_threshold=0.2,
    readout_index=None,
    seed_str=None,
    img_h=150,
    img_w=200,
):
    if seed_str is None:
        seed_str = "8"
    checkpoint_dict = torch.load(
        f"{home}/{model_file}", map_location=torch.device("cpu")
    )
    model = checkpoint_dict["model"]
    visualization_dir = os.path.join(
        "/".join(model_file.split("/")[:-2]), "visualizations"
    )
    config_dir = os.path.join("/".join(model_file.split("/")[:-3]), "config")
    correlation = np.load(
        os.path.join(
            home,
            "/".join(model_file.split("/")[:-3]),
            "stats",
            f"seed_{seed_str}",
            "correlations.npy",
        )
    )
    with open(f"{home}/{config_dir}/config.yaml", "r") as config_file:
        config = yaml.unsafe_load(config_file)
    explainable_variance_threshold = config["dataloader_config"][
        "explainable_variance_threshold"
    ]
    oracle_correlation_threshold = config["dataloader_config"][
        "oracle_correlation_threshold"
    ]
    subsample = config["dataloader_config"]["subsample"]
    config_dict = config["config"]
    # config_dict['cell_oracle_correlations'] = config_dict['cell_oracle_correlation']
    max_corr = max(correlation)
    # max_corr = 0.2
    if max_corr >= corr_threshold:
        best_epoech = np.argmax(correlation)
        # visualize_readout(model, visualization_dir, vmin=vmin, vmax=vmax)
        visualize_all_gaussian_readout(
            model,
            visualization_dir,
            readout_index=readout_index,
            retina_index=readout_index - 1,
            correlation_threshold=oracle_correlation_threshold,
            explainable_variance_threshold=explainable_variance_threshold,
            config=config_dict,
        )
        for layer in range(layers):
            visualize_filters(
                model,
                layer,
                visualization_dir,
                max_corr=max_corr,
                max_corr_epoch=best_epoech,
            )


def plot_selective_curves(directory, curve, parameters: dict):
    fig, ax = plt.subplots(len(parameters.keys()), len(list(parameters.values())[0]))
    for i, parameter in enumerate(parameters):
        pass


def get_label_from_model_name(model_name):
    model_name_list = model_name.split("_")
    learning_rate = model_name_list[1]
    num_of_layers = model_name_list[3]
    channels = model_name_list[4][-1]
    input_kernel_size = [x for x in model_name_list if "ikernel" in x]
    input_kernel_size = (
        input_kernel_size[0][-1]
        if len(input_kernel_size[0]) == 8
        else input_kernel_size[0][-2:]
    )

    hidden_kernel_size = [x for x in model_name_list if "hkernel" in x]
    if len(hidden_kernel_size) > 0:
        hidden_kernel_size = (
            hidden_kernel_size[0][-1]
            if len(hidden_kernel_size[0]) == 8
            else input_kernel_size[0][-2:]
        )
    else:
        hidden_kernel_size = 15
    if int(num_of_layers) > 1:
        legend = (
            f"layers: {num_of_layers} lr: {learning_rate} hidden_channels: {channels} input filter size: {input_kernel_size} "
            f"hidden filter size(s): {hidden_kernel_size}"
        )
    else:
        legend = f"layers: {num_of_layers} lr: {learning_rate} hidden_channels: {channels} input filter size: {input_kernel_size}"
    return legend


color_dict = {"factorized_cnn", "basic_cnn", "no_bn_last_layer", "varied_channels"}


def plot_curves(
    directory,
    curve,
    file_suffix="",
    max_lenght=None,
    files=None,
    best_corr_threshold=0,
    seed=None,
    file_substring="_",
):
    plt.figure(figsize=(15, 10))
    # legend= None
    if files is None:
        files = os.listdir(directory)
    file_correlations = {}
    for file in files:
        if file.startswith("lr") and ("l_1" in file) and (file_substring in file):
            if "flipped" in file:
                print("stop")  # and ('tr_150' in file):#or ('ch_16' in file)):
            # and (os.path.isfile(os.path.join(
            # directory, file, 'weights', 'best_model.m'))): if ('padTrue' in file) and ('l_3' in file): # and (
            # 'fc_readout' in file) and ('s_1_' in file): #and ('lr_0.0001' in file):
            if seed is None:
                seeds = get_possible_seeds(model_name=file, model_dir=directory)
            else:
                seeds = [seed]
            print('')
            model_corrs = []
            for model_seed in seeds:
                correlation_file = os.path.join(
                    directory, file, "stats", f"seed_{model_seed}", f"{curve}.npy"
                )
                if os.path.isfile(correlation_file):
                    try:
                        # legend = get_label_from_model_name(file)
                        legend = file
                        correlations = os.path.join(correlation_file)
                        correlations = np.load(correlations)
                        if len(correlations) == 0:
                            continue
                        file_correlations[file, model_seed] = np.max(correlations), len(
                            correlations
                        )
                        if np.max(correlations) > best_corr_threshold:
                            visualize_learning_curve(
                                correlations, file, max_length=max_lenght, label=legend
                            )
                            model_corrs.append(np.max(correlations))
                    except Exception as e:
                        print(e)
            # print(f'model {file} mean: {np.mean(model_corrs)}')
            # print(f'seeds: {seeds}')
    plt.ylabel(f"{curve}")
    plt.xlabel("Epochs")
    plt.ylim(-0.05, 0.35)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")
    # plt.tight_layout()
    # plt.savefig(
    #     f'{home}/models/basic_cnn/visualization_plots/curves/{curve}{file_suffix}.png',
    #     bbox_inches='tight')
    plt.show()
    file_correlations = {
        k: v
        for k, v in sorted(
            file_correlations.items(), key=lambda item: item[1], reverse=True
        )
    }
    for k, v in file_correlations.items():
        print(k, v)
        # print(file_correlations)
    return file_correlations


def visualize_learning_curve(curve, config, max_length=None, label=None):
    if max_length is not None:
        curve = curve[:max_length]
    if label is not None:
        plot_label = label
    else:
        plot_label = config
    plt.plot(np.arange(len(curve)), curve, label=plot_label)


def visualize_epoch_weights(
    ax_row, config_file, cell_id, crop_size=None, plot_epoch=False
):
    all_frames = []
    correlations = np.load(os.path.join(config_file, "stats", f"correlations.npy"))

    filenames = os.listdir(os.path.join(config_file, "weights"))
    sorted_filenames = sorted(
        filenames, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    for i, file in enumerate(sorted_filenames):
        weights = np.load(os.path.join(config_file, "weights", f"{file}"))
        weights = weights[0]
        if crop_size is not None:
            rf_center = get_receptive_field_center(
                cell_id, retina_index=1, crop=crop_size
            )
            rf_size = get_rf_size(
                rf_center,
                crop_size,
            )
            cropped_weights = crop_cell_rf(
                whole_image=weights, rf_size=rf_size, rf_center=rf_center
            )
        cropped_weights = weights.transpose(1, 2, 0)
        epoch = file.split("_")[-1][:-4]
        if plot_epoch:
            ax_row[i].set_title(f"Epoch: {epoch}")
        corr = correlations[int(epoch)]
        ax_row[i].set_xlabel("Corr: {:.2f}".format(corr))
        frames = visualize_weights(cropped_weights, ax_row[i], vmin=-0.1, vmax=0.1)
        all_frames.append(frames)
    return all_frames


def visualize_one_cell_weight_subplots(
    configurations,
    crops,
    ylabels,
    num_of_epochs,
    cell_id,
    num_of_frames,
    true_crops=None,
):
    fig, ax = plt.subplots(len(configurations), num_of_epochs + 1, figsize=(16, 9))
    timed_all_frames = [[] for _ in range(num_of_frames)]

    all_frames = []
    for i, (config, crop) in enumerate(zip(configurations, crops)):
        if i == 0:
            plot_title = True
        else:
            plot_title = False
        ax[i, 0].set_ylabel(ylabels[i])
        config_frames = visualize_epoch_weights(
            ax[i, :], config, cell_id, crop_size=crop, plot_epoch=plot_title
        )
        ground_truth = get_receptive_field(
            cell_id, num_of_frames, crop_size=true_crops[i]
        )
        true_frames = visualize_weights(ground_truth, ax[i, -1], vmin=-0.01, vmax=0.01)
        ax[i, -1].set_title("RF from STA")
        # config_frames.append(true_frames)
        for frame in config_frames:
            for j, time_step in enumerate(frame):
                timed_all_frames[j].append(time_step[0])
        for j, time_step in enumerate(true_frames):
            timed_all_frames[j].append(time_step[0])
        all_frames.append(config_frames)
    for axis in ax.flatten():
        axis.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axis.tick_params(
            axis="y", which="both", right=False, left=False, labelleft=False
        )

    print(f"all frames length: {len(all_frames)}")
    anim = animation.ArtistAnimation(
        fig, timed_all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling/models/ln_models/visualization_plots/cell_{cell_id}.mp4"
    )
