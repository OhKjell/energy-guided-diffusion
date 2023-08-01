import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import animation
import os

from dynamic.datasets.stas import (
    crop_around_receptive_field,
    visualize_weights,
    separate_time_space_sta,
    get_cell_svd,
)
from dynamic.meis.MEI import fit_gaussian
from dynamic.meis.visualizer import save_all_mei_videos, get_logged_array
from dynamic.utils.global_functions import (
    home,
    get_cell_names,
    get_cell_numbers_after_crop,
    model_seed,
    global_config,
)


def plot_space_time_separated_plot_ln(weight, channel):
    # spat_filter, temp_filter = separate_time_space_sta(weight.numpy())
    spat_filter, temp_filter = get_cell_svd(weight)
    plot_spatio_temporal_ln(
        spatial_kern=spat_filter,
        temp_kern=temp_filter[::-1],
        input_ch=channel,
        output_ch=0,
        output_spatial_ch=0,
    )


def space_time_separated_filter(weight, ax, cell):
    ax.imshow(weight, vmin=-0.2, vmax=0.2, cmap="gray")
    ax.set_title(
        "Spatial and temporal kernel from svd \n cell 0 in dataset cell_data_01_NC"
    )
    plt.savefig(
        f"{home}/datasets/visualization_plots/tmp/cell_{cell}_factorized_svd_spat-temp_kernel.png"
    )


def plot_ete_ln(mei, subsample, crop=True, crop_size=21):
    if not crop:
        crop_size = mei.shape[-1]
    mei_2 = np.zeros(
        (mei.shape[0], (mei.shape[1] + 1) // subsample, crop_size, crop_size)
    )
    cropped_meis = np.zeros((mei.shape[0], mei.shape[1], crop_size, crop_size))
    all_frames = [[] for x in range(mei.shape[1])]
    for i in range(mei.shape[0]):
        single_mei = mei[i]
        if crop:
            max_coordinate = np.var(single_mei, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(max_coordinate), (mei.shape[-2], mei.shape[-1])
            )
            cropped_mei = crop_around_receptive_field(
                max_coordinate=max_coordinate,
                images=torch.unsqueeze(torch.tensor(single_mei), 0),
                rf_size=(crop_size, crop_size),
                h=mei.shape[-2],
                w=mei.shape[-1],
            )
        else:
            cropped_mei = single_mei
        cropped_subsampled_mei = cropped_mei[0, ::subsample]
        mei_2[i] = cropped_subsampled_mei
        cropped_meis[i] = cropped_mei
    fig, ax = plt.subplots(1, mei_2.shape[1] + 1, figsize=(25, 10))
    max_value = np.max(np.abs(cropped_meis[0]))
    frames = visualize_weights(
        weights=cropped_meis[0],
        ax=ax[0],
        vmax=max_value,
        vmin=-max_value,
        weight_index=0,
    )
    for i, frame in enumerate(frames):
        all_frames[i].append(frame[0])
        ax[0].grid(False)
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        plt.axis("off")
    ax[0].set(ylabel=f"LN model filter cell 0")
    f = list(range(15))[::2]
    for i in range(1, mei_2.shape[1] + 1):
        ax[i].imshow(mei_2[0, i - 1], vmin=-max_value, vmax=max_value, cmap="gray")
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set_title(f"Frame {f[i-1]}")
        ax[i].grid(False)
        # ax[j, i].set_title(f'Cell {j}', size='15')
    plt.axis("off")
    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/bernstein_plots/cell_63_cropped_sta.mp4"
    )
    # plt.savefig(f'/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/meis.png', dpi=300)
    # plt.show()


def plot_frame_changes(mei, subsample=2, crop=True, crop_size=21):
    mei_2 = np.zeros(
        (mei.shape[0], (mei.shape[1] + 1) // subsample, crop_size, crop_size)
    )
    cropped_meis = np.zeros((mei.shape[0], mei.shape[1], crop_size, crop_size))
    all_frames = [[] for x in range(mei.shape[1])]
    for i in range(mei.shape[0]):
        single_mei = mei[i]
        if crop:
            max_coordinate = np.var(single_mei, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(max_coordinate), (mei.shape[-2], mei.shape[-1])
            )
            cropped_mei = crop_around_receptive_field(
                max_coordinate=max_coordinate,
                images=torch.unsqueeze(torch.tensor(single_mei), 0),
                rf_size=(crop_size, crop_size),
                h=mei.shape[-2],
                w=mei.shape[-1],
            )
        else:
            cropped_mei = single_mei
        cropped_subsampled_mei = cropped_mei[0, ::subsample]
        mei_2[i] = cropped_subsampled_mei
        cropped_meis[i] = cropped_mei

    fig, ax = plt.subplots(1, mei_2.shape[1], figsize=(20, 10))
    all_frames = [[] for _ in range(mei_2.shape[0])]
    for i in range(mei_2.shape[1]):
        evolution = mei_2[:, i]
        frames = visualize_weights(
            weights=evolution, ax=ax[i], vmax=1, vmin=-1, weight_index=0
        )
        for i, frame in enumerate(frames):
            all_frames[i].append(frame[0])
    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/bernstein_plots/mei_63_optimization.mp4"
    )


def plot_meis(mei, subsample, crop=True, crop_size=21):
    mei_2 = np.zeros(
        (mei.shape[0], (mei.shape[1] + 1) // subsample, crop_size, crop_size)
    )
    cropped_meis = np.zeros((mei.shape[0], mei.shape[1], crop_size, crop_size))
    all_frames = [[] for x in range(mei.shape[1])]
    for i in range(mei.shape[0]):
        single_mei = mei[i]
        if crop:
            max_coordinate = np.var(single_mei, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(max_coordinate), (mei.shape[-2], mei.shape[-1])
            )
            cropped_mei = crop_around_receptive_field(
                max_coordinate=max_coordinate,
                images=torch.unsqueeze(torch.tensor(single_mei), 0),
                rf_size=(crop_size, crop_size),
                h=mei.shape[-2],
                w=mei.shape[-1],
            )
        else:
            cropped_mei = single_mei
        cropped_subsampled_mei = cropped_mei[0, ::subsample]
        mei_2[i] = cropped_subsampled_mei
        cropped_meis[i] = cropped_mei
    fig, ax = plt.subplots(mei_2.shape[0], mei_2.shape[1] + 1, figsize=(25, 10))
    f = [x for x in range(mei.shape[1])]
    f = [x for x in range(3, 15 - 3)]
    for j in range(mei_2.shape[0]):
        frames = visualize_weights(
            weights=cropped_meis[j], ax=ax[j, 0], vmax=1, vmin=-1, weight_index=0
        )
        for i, frame in enumerate(frames):
            all_frames[i].append(frame[0])
            ax[j, 0].grid(False)
            ax[j, 0].set_yticklabels([])
            ax[j, 0].set_xticklabels([])
            plt.axis("off")
        ax[j, 0].set(ylabel=f"Cell {63+j}")
        for i in range(1, mei_2.shape[1] + 1):
            ax[j, i].imshow(mei_2[j, i - 1], vmin=-1, vmax=1, cmap="gray")
            ax[j, i].set_yticklabels([])
            ax[j, i].set_xticklabels([])
            ax[j, i].grid(False)
            ax[j, i].set_title(f"Frame {f[i - 1]}")
            # ax[j, i].set_title(f'Cell {j}', size='15')
    plt.axis("off")

    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/bernstein_plots/mei_63_optimization.mp4"
    )
    # plt.savefig(f'/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots//meis.png', dpi=300)
    # plt.show()


def plot_cnn_filter(weight, subsample, crop=True, crop_size=21):
    if not crop:
        crop_size = weight.shape[-1]
    mei_2 = np.zeros(
        (weight.shape[0], (weight.shape[1] + 1) // subsample, crop_size, crop_size)
    )
    cropped_meis = np.zeros((weight.shape[0], weight.shape[1], crop_size, crop_size))
    all_frames = [[] for x in range(weight.shape[1])]
    for i in range(weight.shape[0]):
        single_mei = weight[i]
        if crop:
            max_coordinate = np.var(single_mei, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(max_coordinate), (weight.shape[-2], weight.shape[-1])
            )
            cropped_mei = crop_around_receptive_field(
                max_coordinate=max_coordinate,
                images=torch.unsqueeze(torch.tensor(single_mei), 0),
                rf_size=(crop_size, crop_size),
                h=weight.shape[-2],
                w=weight.shape[-1],
            )
        else:
            cropped_mei = single_mei
        cropped_subsampled_mei = cropped_mei[4:-3]
        mei_2[i] = cropped_subsampled_mei
        cropped_meis[i] = cropped_mei
    fig, ax = plt.subplots(1, mei_2.shape[1] + 1, figsize=(25, 10))
    frames = visualize_weights(
        weights=cropped_meis[0], ax=ax[0], vmax=11, vmin=-11, weight_index=0
    )
    for i, frame in enumerate(frames):
        all_frames[i].append(frame[0])
        ax[0].grid(False)
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        plt.axis("off")
    ax[0].set(ylabel=f'CNN "filter" cell 0')
    f = [x for x in range(3, 15 - 3)]
    for i in range(1, mei_2.shape[1] + 1):
        ax[i].imshow(mei_2[0, i - 1], vmin=-11, vmax=11, cmap="gray")
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set_title(f"Frame {f[i]}")
        ax[i].grid(False)
        # ax[j, i].set_title(f'Cell {j}', size='15')
    plt.axis("off")
    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/cnn_filter_cell_0.mp4"
    )
    # plt.savefig(f'/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/meis.png', dpi=300)
    # plt.show()


def show_non_space_time_separated_plot(weights, subsample):
    weights = weights[:, ::subsample]
    fig, ax = plt.subplots(weights.shape[2], figsize=(8, 20))
    for i in range(weights.shape[2]):
        for j in range(weights.shape[0]):
            ax[i].imshow(weights[j, 0, i], vmin=-20.0, vmax=20.0, cmap="gray")
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[0].set_title(f"MEIs \n for cell 0", size="25")
    plt.axis("off")


def show_ete_ln_filter(file, cell_index):
    model = torch.load(
        f"{home}/models/ln_models/salamander/retina1/cell_{cell_index}/{file}/weights/best_model.m",
        map_location=torch.device("cpu"),
    )
    correlations = np.load(
        f"{home}/models/ln_models/salamander/retina1/cell_{cell_index}/{file}/stats/correlations.npy"
    )
    print("max corr:", np.max(correlations))
    model = model["model"]
    filter = model["conv1.weight"]
    plot_ete_ln(filter[0], 2, crop=False)
    # filter = np.random.random(filter.shape)
    plot_space_time_separated_plot_ln(filter)


def show_cnn_filter(file, cell_index, channel=0):
    model = torch.load(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/models/basic_ev_0.15_cnn/salamader/retina1/cell_None/readout_isotropic/gmp_0/{file}/weights/seed_8/best_model.m",
        map_location=torch.device("cpu"),
    )
    correlations = np.load(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/models/basic_ev_0.15_cnn/salamader/retina1/cell_None/readout_isotropic/gmp_0/{file}/stats/seed_8/correlations.npy"
    )
    print("max corr:", np.max(correlations))
    model = model["model"]
    weight = model[f"core.features.layer0.conv.weight"].numpy()
    features = model[f"readout._features"].numpy()[0, :, 0, cell_index]
    single_cell_filter = np.sum(
        np.concatenate(
            [np.expand_dims(weight[i] * features[i], 0) for i in range(len(features))]
        ),
        axis=0,
    )
    plot_space_time_separated_plot_ln(weight[channel, 0], channel=channel)
    # plot_meis(weight[:3, 0], subsample=2, crop=False, crop_size=15)
    # plot_cnn_filter(single_cell_filter, subsample=2, crop=False)


def show_multi_layer_cnn_filter(file, input_channel, output_channel, output_spatial):
    model = torch.load(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/models/factorized_ev_0.15_cnn/salamander/retina1/cell_None/readout_isotropic/gmp_0/{file}/weights/seed_64/best_model.m",
        map_location=torch.device("cpu"),
    )
    model = model["model"]
    spatial_kern = model[f"core.features.layer0.conv_spatial.weight"].numpy()
    temp_kern = model[f"core.features.layer0.conv_temporal.weight"].numpy()[:, :, ::-1]
    # spatial_kern = np.random.random(spatial_kern.shape)
    # temp_kern = np.random.random(temp_kern.shape)
    plot_spatio_temporal(
        spatial_kern=spatial_kern,
        temp_kern=temp_kern,
        input_ch=input_channel,
        output_ch=output_channel,
        output_spatial_ch=output_spatial,
    )


def plot_factorized_ln_model(file, cell_index):
    model = torch.load(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/models/ln_models/ln_models_factorized_sta/salamander/retina1/cell_{cell_index}/{file}/weights/seed_8/best_model.m"
    )
    model = model["model"]
    spatial = model["conv1.weight"].detach().cpu().numpy()[0, 0, 0]
    spatial_max = np.unravel_index(np.argmax(spatial), spatial.shape)
    spatial = spatial[
        spatial_max[0] - 10 : spatial_max[0] + 10,
        spatial_max[1] - 10 : spatial_max[1] + 10,
    ]
    mu, cov, spatial = fit_gaussian(spatial)
    temporal = model["conv2.weight"].detach().cpu().numpy()[0, 0, :, 0, 0]
    correlations = np.load(
        f"{home}/models/ln_models/ln_models_factorized_svd/marmoset/retina1/cell_{cell_index}/{file}/stats/seed_8/correlations.npy"
    )
    print("max corr:", np.max(correlations))
    plot_spatio_temporal_ln(spatial_kern=spatial, temp_kern=temporal)


def plot_spatio_temporal_ln(
    spatial_kern, temp_kern, input_ch=0, output_ch=0, output_spatial_ch=0
):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    all_frames = [[] for _ in range(temp_kern.shape[0])]
    multiplied_kernel = [
        (spatial_kern * temp_kern[i]) for i in range(temp_kern.shape[0])
    ]
    multiplied_kernel = np.array(multiplied_kernel)
    # multiplied_kernel = multiplied_kernel[0, 0]
    vmin = -1 * np.max(np.abs(multiplied_kernel))
    # frames = visualize_weights(weights=multiplied_kernel, ax=ax[2], vmax=-1 * vmin, vmin=vmin, weight_index=0)

    # for i, frame in enumerate(frames):
    #     all_frames[i].append(frame[0])
    #     ax[2].grid(False)
    #     ax[2].set_yticklabels([])
    #     ax[2].set_xticklabels([])
    #     ax[2].set_title('spatio-temporal kernel', size=15)
    #     ax[2].grid(False)
    #
    #     ax[2].set_ylabel('=', fontsize=25, rotation=90)

    # plt.axis('off')
    vmin = -1 * np.max(np.abs(spatial_kern))
    ax[1].imshow(
        -1 * spatial_kern, vmin=vmin, vmax=-1 * vmin, cmap="gray", origin="lower"
    )
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    # ax[1].set_title('space', size=15)
    ax[1].grid(False)
    ax[1].patch.set_edgecolor("black")
    ax[0].grid(False)
    ax[0].set_facecolor("white")

    ax[0].plot(
        np.arange(0, temp_kern.shape[0]), -1 * temp_kern, color="#4766ac", linewidth=10
    )
    ax[0].plot(
        [0, temp_kern.shape[0]],
        [0, 0],
        color="black",
        linestyle=(0, (5, 7)),
        linewidth=5,
    )
    ax[0].plot(
        [0, 0],
        [np.min(-1 * temp_kern), np.max(-1 * temp_kern)],
        color="black",
        linestyle=(0, (5, 7)),
        linewidth=5,
    )

    # ax[0].set_title('time', size=15)
    # ax[0].set(xlabel='Frame')
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
    # ax[0].set_aspect(1)
    # plt.tight_layout()
    # fig.subplots_adjust(wspace=0.1)
    plt.savefig(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/bernstein_plots/factorized_salamander_cell_63.png",
        transparent=True,
    )
    plt.show()
    # anim = animation.ArtistAnimation(fig, all_frames, interval=400, blit=True, repeat_delay=1000)
    # anim.save(
    #     f'/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/1layer_cnn_ch0.mp4')


def plot_spatio_temporal(
    spatial_kern, temp_kern, input_ch=0, output_ch=0, output_spatial_ch=0
):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    all_frames = [[] for _ in range(temp_kern.shape[0])]
    multiplied_kernel = [
        (spatial_kern * temp_kern[output_ch, input_ch, i, 0, 0])
        for i in range(temp_kern.shape[0])
    ]
    multiplied_kernel = np.concatenate(multiplied_kernel, axis=2)
    multiplied_kernel = multiplied_kernel[0, 0]
    vmin = -1 * np.max(np.abs(multiplied_kernel))
    frames = visualize_weights(
        weights=multiplied_kernel, ax=ax[2], vmax=-1 * vmin, vmin=vmin, weight_index=0
    )

    for i, frame in enumerate(frames):
        all_frames[i].append(frame[0])
        ax[2].grid(False)
        ax[2].set_yticklabels([])
        ax[2].set_xticklabels([])
        ax[2].set_title("spatio-temporal kernel", size=15)
        plt.axis("off")
    vmin = -1 * np.max(np.abs(spatial_kern[output_spatial_ch]))
    ax[1].imshow(
        spatial_kern[output_spatial_ch, 0, 0],
        vmin=vmin,
        vmax=-1 * vmin,
        cmap="gray",
        origin="lower",
    )
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_title("spatial kernel", size=15)
    ax[1].grid(False)
    ax[2].grid(False)

    ax[0].plot(
        np.arange(0, temp_kern.shape[2]), temp_kern[output_ch, input_ch, :, 0, 0]
    )
    ax[0].set_title("temporal kernel", size=15)
    ax[0].plot([0, temp_kern.shape[2]], [0, 0], color="black", linestyle="dashed")
    ax[0].plot(
        [0, 0],
        [
            np.min(temp_kern[output_ch, input_ch, :, 0, 0]),
            np.max(temp_kern[output_ch, input_ch, :, 0, 0]),
        ],
        color="black",
        linestyle="dashed",
    )
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_ylim(
        np.min(temp_kern[output_ch, input_ch, :, 0, 0]),
        np.max(temp_kern[output_ch, input_ch, :, 0, 0]),
    )
    ax[0].set_facecolor("white")

    plt.tight_layout()
    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/3layer_cnn_filter_{input_ch}_{output_ch}_{output_spatial_ch}.mp4"
    )


def plot_mei_optimization(retina_index_str, cell):
    epochs = [j * 50 for j in range(16)]
    changing_mei = []
    for epoch in epochs:
        mei = get_logged_array(
            os.path.join(
                home,
                "meis",
                "data",
                "salamander",
                f"retina{retina_index_str[1]}",
                file,
                f"cell_{cell}",
                f"{hash}_seed_{seed}_lr_{lr}_std_{std}_np_{num_of_preds}",
            ),
            epoch,
        )
        changing_mei.append(mei)
    changing_mei = np.concatenate(changing_mei, axis=0)[:, 0]
    plot_frame_changes(changing_mei[:, -15:-1], subsample=2, crop=True)
    return changing_mei


def get_saved_ln_performance(
    file_name,
    retina_index,
    exclude_cells=False,
    correlation_threshold=None,
    explainable_variance_threshold=0,
    data_type="salamander",
    base_path=f"{home}/models/ln_models/",
    config=global_config,
    save_name="sigle_cell_correlations.npy",
    save_dir=None,
):
    cell_values = []
    retina_index_str = f"0{retina_index + 1}"
    num_of_cells = (
        get_cell_numbers_after_crop(
            retina_index,
            config=config,
            correlation_threshold=correlation_threshold
            if correlation_threshold is not None
            else 0,
            explained_variance_threshold=explainable_variance_threshold
            if explainable_variance_threshold is not None
            else 0,
        )
        if exclude_cells
        else get_cell_numbers_after_crop(retina_index, config=config)
    )
    # num_of_cells -=1
    cell_names_list = (
        get_cell_names(
            retina_index,
            config=config,
            correlation_threshold=correlation_threshold
            if correlation_threshold is not None
            else 0,
            explained_variance_threshold=explainable_variance_threshold
            if explainable_variance_threshold is not None
            else 0,
        )
        if exclude_cells
        else [x for x in range(num_of_cells)]
    )
    for cell in cell_names_list:
        if os.path.isfile(
            f"{home}/{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/test_correlation.npy"
        ):
            test_correlation = np.load(
                f"{home}/{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/test_correlation.npy"
            )
            cell_values.append(test_correlation)
        else:
            print(f"no data for cell {cell}")
    if save_dir is not None:
        np.save(f"{home}/{save_dir}/{save_name}", cell_values)
        print("saved to", f"{home}/{save_dir}/{save_name}")
    return cell_values


def plot_sta(cell, retina_index):
    sta = np.load(
        f"{home}/data/salamander_data/stas/cell_data_0{retina_index+1}_NC_stas_cell_{cell}.npy"
    )
    plot_ete_ln(np.expand_dims(sta[9:-1], 0), subsample=2, crop=True, crop_size=20)


if __name__ == "__main__":
    # with open(f'{home}/data/marmoset_data/responses/config_05.yaml', 'rb') as config_file:
    #     config = yaml.unsafe_load(config_file)
    # with open(f'{home}/data/salamander_data/responses/config.yaml', 'rb') as config_file:
    #     config = yaml.unsafe_load(config_file)
    with open(
        f"{home}/data/marmoset_data/responses/config_s4.yaml", "rb"
    ) as config_file:
        config = yaml.unsafe_load(config_file)
    get_saved_ln_performance(
        file_name="lr_0.001_whole_rf_20_ch_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_10_s_1_c_0_n_0_fn_0_do_n_1",
        retina_index=0,
        exclude_cells=True,
        base_path="models/ln_models_factorized_4_sta_nm/",
        data_type="marmoset",
        save_name="factorized_sta_on_nm",
        save_dir="datasets/visualization_plots/bernstein_plots/",
        config=config,
    )
    exit()
    """plot STA"""
    retina_index = 0
    cell_name = 63
    # plot_sta(cell_name, retina_index)
    """e-t-e ln plotting"""
    retina_index = 0
    model = "lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0"
    # model = 'lr_0.001_whole_rf_60_t_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_11_s_1_c_0_n_0_fn_0_do_n_1'
    #  show_ete_ln_filter(file=model, cell_index=0)

    """cnn filter plotting"""
    file = "lr_0.0100_l_1_ch_16_t_15_bs_10_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1"
    show_cnn_filter(file, cell_index=0)

    """3 layer cnn filter plotting
    file_3 = 'lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x11x11_hk_25x7x7_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1'
    show_multi_layer_cnn_filter(file_3, 0, 0, 0)"""

    """mei plotting
    retina_index = 0
    hash = '2hQ7vp3FO7lE4'
    # hash = '4Dq3qS1uH2Zp'
    seed = [128, 1024, 42, 2048, 64, 256, 8]
    lr = 10
    num_of_preds = 1
    std = 0.025
    file = 'lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x11x11_hk_25x7x7_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1'
    cell_names = get_cell_names(retina_index=retina_index, correlation_threshold=0, explained_variance_threshold=0.15)
    plot_mei_optimization(retina_index_str=f'0{retina_index + 1}', cell=63)
    # mei = save_all_mei_videos(retina_index_str=f'0{retina_index + 1}', file=file,
                        # saving_file=f'/usr/users/vystrcilova/retinal_circuit_modeling/meis/data/salamander/retina{retina_index +1}/{file}/all_meis_{hash}_std_{std}_np_{num_of_preds}',
                        # epoch=-1, cell_names=cell_names, hash=hash, seed=seed, lr=lr, std=std, num_of_preds=num_of_preds)
    # plot_meis(mei[47:50, -15:-1], subsample=2, crop=True)"""

    """factorized ln plotting"""
    model = "lr_0.001_whole_rf_60_ch_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_250_s_1_c_0_n_0_fn_0_do_n_1"
    plot_factorized_ln_model(file=model, cell_index=0)
