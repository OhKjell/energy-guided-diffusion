import math
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, gridspec
import torch
from tqdm import tqdm

from evaluations.parameter_dependant_performance import get_model_overall_spatial_reach

# from training.measures import correlation
from utils.global_functions import (
    get_exclude_cells_based_on_correlation_threshold,
    get_exclude_cells_based_on_explainable_variance_threshold,
    home,
    global_config,
    get_cell_names,
)
from sklearn.decomposition import PCA


def get_receptive_field_center(
    retina_index,
    cell_index,
    crop,
    all_rf_fields=None,
    data_dir=f"{home}/",
    file_suffix="_NC_stas",
    config=None,
    subsample=1,
):
    if config is None:
        config = global_config
    if all_rf_fields is None:
        if "sta_file" not in config.keys():
            cell_file = f"cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_cell_{cell_index}.npy"
        else:
            cell_file = create_cell_file_from_config_version(
                config["sta_file"], cell_index=cell_index, retina_index=retina_index
            )
        all_rf_fields = np.load(
            f'{data_dir}/data/{config["data_type"]}_data/stas/{cell_file}'
        )
        # all_rf_fields = np.load(f'{data_dir}/data/{config["data_type"]}_data/stas/cell_data_{str(retina_index + 1).zfill(2)}{file_suffix}_cell_{cell_index}.npy')
    cell_rf = all_rf_fields
    # cell_rf = all_rf_fields[cell_index]
    if sum(crop) > 0:
        num_of_imgs, h, w = cell_rf.shape
        cell_rf = cell_rf[
            :, crop[0] : h - crop[1] : subsample, crop[2] : w - crop[3] : subsample
        ]

    else:
        # TODO: Stupid but cannot subsample if not cropping
        print("Stupid but cannot subsample if not cropping")
    temporal_variances = np.var(cell_rf, axis=0)
    rf_center = np.unravel_index(
        np.argmax(temporal_variances), (cell_rf.shape[1], cell_rf.shape[2])
    )
    return rf_center


def get_rf_center_grid(
    retina_index,
    crop,
    data_dir=f"{home}/",
    correlation_threshold=None,
    explainable_varinace_threshold=None,
    config=None,
    suffix="_NM_stas",
    subsample=1,
    exclude_cells=True,
):
    if explainable_varinace_threshold is None:
        explainable_varinace_threshold = 0
    if correlation_threshold is None:
        correlation_threshold = 0
    if config is None:
        config = global_config
    if correlation_threshold is not None:
        excluded_cells = get_exclude_cells_based_on_correlation_threshold(
            retina_index, config, correlation_threshold
        )
    else:
        excluded_cells = []
    if explainable_varinace_threshold is not None:
        excluded_cells += get_exclude_cells_based_on_explainable_variance_threshold(
            retina_index, config=config, threshold=explainable_varinace_threshold
        )
    more_excluded_cells = []
    if exclude_cells:
        more_excluded_cells = config["exclude_cells"][str(retina_index + 1).zfill(2)]
    excluded_cells = list(set(excluded_cells + more_excluded_cells))
    if len(excluded_cells) == 0:
        excluded_cells = config["exclude_cells"][str(retina_index + 1).zfill(2)]
    cell = 0
    all_centers = np.zeros(
        (
            config["cell_numbers"][str(retina_index + 1).zfill(2)]
            - len(excluded_cells),
            2,
        )
    )

    for cell_index in range(config["cell_numbers"][str(retina_index + 1).zfill(2)]):
        if cell_index not in excluded_cells:
            print(cell_index)
            rf_center = get_receptive_field_center(
                retina_index,
                cell_index,
                crop,
                data_dir=data_dir,
                all_rf_fields=None,
                file_suffix=suffix,
                config=config,
                subsample=subsample,
            )
            all_centers[cell] = rf_center
            cell += 1
    return all_centers


def get_receptive_field(
    cell_id,
    number_of_frames,
    data_type="salamander",
    crop_size=None,
    img_h=150,
    img_w=200,
    retina_str_index="01",
    file_suffix="_NC_stas",
):
    receptive_fields = np.load(
        # f'{home}/data/{data_type}_data/stas/cell_data_{retina_str_index}{file_suffix}_cell_{cell_id}.npy')
        f"{home}/data/{data_type}_data/stas/cell_data_01_NM_stas_s4_zero_mean_cell_{cell_id}.npy"
    )
    # cell_rf = receptive_fields[cell_id, receptive_fields.shape[1] - number_of_frames:]
    cell_rf = receptive_fields[receptive_fields.shape[0] - number_of_frames :]

    if crop_size is not None:
        temporal_variances = np.var(cell_rf, axis=0)
        rf_center = np.unravel_index(
            np.argmax(temporal_variances), (cell_rf.shape[1], cell_rf.shape[2])
        )
        half_rf_size = crop_size // 2
        rf_size = get_rf_size(
            rf_center=rf_center,
            half_rf_size=half_rf_size,
            img_h=cell_rf.shape[1],
            img_w=cell_rf.shape[2],
        )
        cell_rf = crop_cell_rf(
            whole_image=cell_rf,
            rf_size=rf_size,
            rf_center=rf_center,
            img_h=cell_rf.shape[1],
            img_w=cell_rf.shape[2],
        )
    return cell_rf.transpose(1, 2, 0)


def crop_cell_rf(whole_image, rf_size, rf_center, img_h=150, img_w=200):
    image = whole_image[
        :,
        (rf_center[0] - min(rf_size[0] // 2, rf_center[0])) : (
            rf_center[0]
            + min((rf_size[0] // 2) + (rf_size[0] % 2), img_h - rf_center[0])
        ),
        (rf_center[1] - min(rf_size[1] // 2, rf_center[1])) : (
            rf_center[1]
            + min((rf_size[1] // 2) + (rf_size[1] % 2), img_w - rf_center[1])
        ),
    ]

    left = rf_center[0] - math.floor(rf_size[0] / 2)
    right = img_h - (rf_center[0] + math.ceil(rf_size[0] / 2))
    top = rf_center[1] - math.floor(rf_size[1] / 2)
    bottom = img_w - (rf_center[1] + math.ceil(rf_size[1] / 2))

    new_img = np.zeros((whole_image.shape[0],) + rf_size)
    if left < 0:
        if top < 0:
            new_img[:, -1 * left :, -1 * top :] = image
        elif bottom < 0:
            new_img[:, -1 * left :, :bottom] = image
        else:
            new_img[:, -1 * left :, :] = image
    elif right < 0:
        if top < 0:
            new_img[:, :right, -1 * top :] = image
        elif bottom < 0:
            new_img[:, :right, :bottom] = image
        else:
            new_img[:, :right, :] = image
    elif top < 0:
        new_img[:, :, -1 * top :] = image
    elif bottom < 0:
        new_img[:, :, :bottom] = image

    else:
        new_img = image
    return new_img


def get_rf_size(rf_center, half_rf_size, img_h=150, img_w=200):
    return (rf_center[0] + min(half_rf_size, img_h - rf_center[0])) - (
        rf_center[0] - min(half_rf_size, rf_center[0])
    ), (rf_center[1] + min(half_rf_size, img_w - rf_center[1])) - (
        rf_center[1] - min(half_rf_size, rf_center[1])
    )


def recalculate_positions_after_convs(
    all_centers: np.ndarray, kernel_sizes: list, img_h=50, img_w=60, plot=False
):
    """

    :param all_centers: array of shape nx2
    :param kernel_sizes: list of tuples representing sizes of the convolutional kernels
                         TODO: Currently have to be squared
    :param img_h: height of the input
    :param img_w: height of the output
    :param plot: whether to plot the new coordinates
    :return: array of shape nx2 with coordinates after convolution
    """
    # expects kernel to be square shaped
    new_centers = None
    for kernel in kernel_sizes:
        # assert (kernel[1] % 2) == 1 and (kernel[2] % 2 == 1)
        assert kernel[1] == kernel[2]
        new_centers = np.zeros(all_centers.shape)
        subtract = math.floor(kernel[1] / 2)
        output_shape = img_h - 2 * subtract, img_w - 2 * subtract
        for center in range(all_centers.shape[0]):
            current_center_x, current_center_y = all_centers[center]
            new_x_coord = (
                max(0, current_center_x - subtract)
                if current_center_x < img_h / 2
                else min(current_center_x - subtract, output_shape[0])
            )
            new_y_coord = (
                max(0, current_center_y - subtract)
                if current_center_y < img_w / 2
                else min(current_center_y - subtract, output_shape[1])
            )
            new_centers[center] = new_x_coord, new_y_coord
        img_h, img_w = output_shape
        all_centers = new_centers
    if plot:
        fig, ax = plt.subplots(2, sharex="col", sharey="col")
        ax[0].imshow(np.zeros((img_h, img_w)).T)
        ax[0].scatter(*all_centers.T)
        ax[1].imshow(np.zeros(output_shape).T)
        ax[1].scatter(*new_centers.T)
        plt.show()
    return np.asarray(new_centers)


def calculate_position_before_convs(
    model_config,
    center,
    feature_h=None,
    feature_w=None,
    img_h=None,
    img_w=None,
    plot=False,
):
    reach = int(get_model_overall_spatial_reach(model_config) / 2)
    new_center_x = center[0] + reach
    new_center_y = center[1] + reach

    if plot:
        fig, ax = plt.subplots(2, sharex="col", sharey="col")
        ax[0].imshow(np.zeros((img_h, img_w)).T)
        ax[0].scatter(center[1], center[0])
        ax[1].imshow(np.zeros((feature_h, feature_w)).T)
        ax[1].scatter(new_center_y, new_center_x)
        plt.show()
    return (int(new_center_x), int(new_center_y)), reach * 2 + 1


def unnormalize_source_grid(coordinates, core_output_shape):
    x_shape = core_output_shape[-1]
    y_shape = core_output_shape[-2]

    coordinates[0] *= x_shape / 2
    coordinates[1] *= y_shape / 2

    coordinates[0] += x_shape / 2
    coordinates[1] += y_shape / 2

    return coordinates


def normalize_source_grid(source_grid, core_output_shape):
    x_shape = core_output_shape[0]
    y_shape = core_output_shape[1]
    source_grid[:, 0] -= x_shape / 2
    source_grid[:, 1] -= y_shape / 2

    source_grid[:, 0] /= x_shape / 2
    source_grid[:, 1] /= y_shape / 2

    return source_grid


def do_pca(rf_fields, num_of_comps, return_transformed=True):
    rf_fields_shape = rf_fields.shape
    rf_fields = rf_fields.reshape([rf_fields_shape[0], -1])
    pca = PCA(num_of_comps, random_state=8)
    pca.fit(rf_fields)
    loadings = pca.transform(rf_fields)
    new_rfs = pca.inverse_transform(loadings)
    new_rfs = new_rfs.reshape(rf_fields_shape)
    print(
        "shape:",
        new_rfs.shape,
        "explained variance:",
        sum(pca.explained_variance_ratio_),
    )
    return new_rfs, pca, loadings if return_transformed else new_rfs


def get_cell_sta(cell_file, data_dir, rf_size, h=150, w=200):
    cell_rf = np.load(f"{data_dir}/{cell_file}")
    temporal_variances = np.var(cell_rf, axis=0)
    max_coordinate = np.unravel_index(np.argmax(temporal_variances), (150, 200))
    cell_rf = cell_rf[cell_rf.shape[0] - rf_size[0] :, :, :]
    images = crop_cell_rf(cell_rf, rf_size[1:], max_coordinate, img_w=w, img_h=h)
    if (images.shape[0] < rf_size[1]) or (images.shape[1] < rf_size[2]):
        raise ValueError(f'For cell with STA in file {cell_file} the STA center was either not found or is too close '
                         f'to the edge of the image for filter size {rf_size}')
    return images, max_coordinate


def get_sta(in_shape, cell_name, crop):
    whole_sta = np.zeros(in_shape)
    sta, _ = get_cell_sta(f'cell_data_02_WN_stas_40_cell_{cell_name}.npy',
                          f'/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/stas/',
                          rf_size =(40, 150, 200))
    print(sta.shape)
    max_value = np.max(np.abs(sta))
    sta = sta/max_value
    sta = sta[:, crop[0]:-crop[1], crop[2]:-crop[3]]
    whole_sta[0, 0] = sta[1:]
    return whole_sta

def separate_time_space_sta(sta):
    t, h, w = sta.shape
    temporal_variances = np.var(sta, axis=0)
    max_coordinate = np.unravel_index(np.argmax(temporal_variances), (h, w))
    time_pos = np.argmax(np.abs(sta[:, max_coordinate[0], max_coordinate[1]]))
    mis = sta[time_pos, max_coordinate[0], max_coordinate[1]]

    spat_filter = np.sign(mis) * sta[time_pos]
    spat_filter /= np.linalg.norm(spat_filter)

    temp_filter = sta[:, max_coordinate[0], max_coordinate[1]]
    temp_filter /= np.linalg.norm(temp_filter)
    return spat_filter, temp_filter


def get_cell_svd(sta):
    t, h, w = sta.shape
    sta = sta.reshape(sta.shape[0], -1)
    temp, s, spat = np.linalg.svd(sta)
    temp = temp[:, 0]
    spat = spat[0]
    spat = spat.reshape((h, w))
    return spat, temp


def create_cell_file_from_config_version(original_file, cell_index, retina_index):
    new_file = original_file.replace("1", str(retina_index + 1))
    new_file = f"{new_file}_{cell_index}.npy"
    return new_file


def get_cropped_stas(
    retina_index,
    num_of_images,
    rf_size=None,
    h=150,
    w=200,
    data_dir=f"{home}/",
    file_suffix="_NC_stas",
    config_dict=None,
    cell_file=None,
):
    if rf_size is None:
        rf_size = (20, 20)
    if config_dict is None:
        config_dict = global_config
    # all_rf_fields = np.load(f'{data_dir}/data/salamander_data/stas/cell_data_{str(retina_index + 1).zfill(2)}{file_suffix}.npy')
    # all_rf_fields = all_rf_fields[:, all_rf_fields.shape[1] - num_of_images:, :, :]
    num_of_cells = config_dict["cell_numbers"][str(retina_index + 1).zfill(2)]
    all_rf_fiels = np.zeros((num_of_cells, num_of_images, h, w))
    all_cell_rf_fields_cropped = np.zeros((num_of_cells, num_of_images) + rf_size)
    for cell in range(num_of_cells):
        if cell_file is None:
            cell_file = f"data/salamander_data/stas/cell_data_{str(retina_index + 1).zfill(2)}{file_suffix}_cell_{cell}.npy"
        else:
            cell_file = create_cell_file_from_config_version(
                retina_index=retina_index, cell_index=cell, original_file=cell_file
            )
        images = get_cell_sta(
            cell_file, data_dir, rf_size=(num_of_images,) + rf_size, h=h, w=w
        )
        all_cell_rf_fields_cropped[cell] = images
    return all_cell_rf_fields_cropped, all_rf_fiels


def crop_around_receptive_field(max_coordinate, images, rf_size, h, w):
    assert max_coordinate is not None
    if len(images.shape) == 4:
        images = torch.permute(images, (-2, -1, 0, 1))
        images = images[
            max_coordinate[0]
            - min(rf_size[0] // 2, max_coordinate[0]) : max_coordinate[0]
            + min(rf_size[0] // 2 + rf_size[0] % 2, h - max_coordinate[0]),
            max_coordinate[1]
            - min(rf_size[1] // 2, max_coordinate[1]) : max_coordinate[1]
            + min(rf_size[1] // 2 + rf_size[1] % 2, w - max_coordinate[1]),
        ]
        images = torch.permute(images, (-2, -1, 0, 1))
    elif len(images.shape) == 5:
        images = images[
            :,
            :,
            :,
            max_coordinate[0]
            - min(rf_size[0] // 2, max_coordinate[0]) : max_coordinate[0]
            + min(rf_size[0] // 2 + rf_size[0] % 2, h - max_coordinate[0]),
            max_coordinate[1]
            - min(rf_size[1] // 2, max_coordinate[1]) : max_coordinate[1]
            + min(rf_size[1] // 2 + rf_size[1] % 2, w - max_coordinate[1]),
        ]
    return images


def calculate_pca_on_cropped_stas(
    num_of_images,
    retina_index,
    data_dir=f"{home}/",
    rf_size=None,
    h=150,
    w=200,
    all_cell_rf_fields_cropped=None,
):
    if all_cell_rf_fields_cropped is None:
        all_cell_rf_fields_cropped, _ = get_cropped_stas(
            retina_index,
            num_of_images=num_of_images,
            rf_size=rf_size,
            h=h,
            w=w,
            data_dir=data_dir,
        )
    new_cells, components, loadings = do_pca(
        all_cell_rf_fields_cropped,
        all_cell_rf_fields_cropped.shape[0],
        return_transformed=True,
    )
    return all_cell_rf_fields_cropped, loadings, components


def plot_pca_on_cropped_stas(
    num_of_images, num_of_components_to_show, components, rf_size
):
    fig, ax = plt.subplots(num_of_images, num_of_components_to_show, figsize=(30, 12))
    for i in range(num_of_components_to_show):
        component = components.components_[i]
        component = component.reshape((num_of_images,) + rf_size)
        for frame in range(component.shape[0]):
            ax[frame, i].imshow(
                component[frame], cmap="coolwarm", vmin=-0.05, vmax=0.05
            )
        ax[0, i].set_title(f"PC {i}")
    plt.show()
    explained_variance = sum(
        components.explained_variance_ratio_[: num_of_components_to_show + 1]
    )
    print(f"explained_variance: {explained_variance}")
    plt.bar(
        np.arange(len(components.explained_variance_ratio_)),
        components.explained_variance_ratio_,
        color="blue",
        label="EVR for each component",
    )
    plt.plot(
        np.arange(len(components.explained_variance_ratio_)),
        [
            sum(components.explained_variance_ratio_[: x + 1])
            for x in range(len(components.explained_variance_ratio_))
        ],
        color="orange",
        label="EVR sum up to component on x",
    )
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.legend()
    plt.show()


def visualize_weights(
    weights,
    ax,
    vmin=-1.0,
    vmax=1.0,
    weight_index=2,
    max_coordinate=None,
    mask_border=None,
    cmap="gray",
):
    frames = []
    for i in range(weights.shape[weight_index]):
        if weight_index == 2:
            frames.append([ax.imshow(weights[:, :, i], cmap=cmap, animated=True)])
        else:
            if (max_coordinate is None) and (mask_border is None):
                frames.append(
                    [
                        ax.imshow(
                            weights[i, :, :],
                            cmap=cmap,
                            animated=True,
                            vmin=vmin,
                            vmax=vmax,
                        )
                    ]
                )
            elif mask_border is None:
                frames.append(
                    [
                        ax.imshow(
                            weights[i, :, :],
                            cmap=cmap,
                            animated=True,
                            vmin=vmin,
                            vmax=vmax,
                        ),
                        ax.scatter(
                            [max_coordinate[1]], [max_coordinate[0]], color="lime"
                        ),
                    ]
                )
            else:
                frames.append(
                    [
                        ax.imshow(
                            weights[i, :, :],
                            cmap=cmap,
                            animated=True,
                            vmin=vmin,
                            vmax=vmax,
                        ),
                        ax.scatter(
                            np.where(mask_border != 0)[1],
                            np.where(mask_border != 0)[0],
                            color="lime",
                            s=10,
                        ),
                    ]
                )
    return frames


def plot_all_stas(
    retina_index_str,
    saving_file=None,
    saving_file_suffix=None,
    cell_rfs=None,
    correlations=None,
    vmin=-0.1,
    vmax=0.1,
    cells=None,
    cell_names=None,
    mask_border=None,
    cmap="coolwarm",
    data_type="salamander",
    config=None,
    num_of_frames=25,
    crop_size=60,
    file_suffix="_NC_stas_25",
    weight_index=2,
):
    retina_index = int(retina_index_str) - 1
    if config is None:
        config = global_config
    row_length = 10
    cell_names_list = get_cell_names(
        retina_index,
        config=config,
        correlation_threshold=0,
        explained_variance_threshold=0,
    )[:100]
    # cell_names_list = [2,8,52,154,203,220,256,280,316,326]
    cell_names = cell_names_list
    num_of_cells = cell_rfs.shape[0] if cell_rfs is not None else len(cell_names_list)
    fig, ax = plt.subplots(
        math.ceil(num_of_cells / row_length), row_length, figsize=(30, 30)
    )
    all_frames = [[] for x in range(num_of_frames)]
    max_coordinate = None
    if cells is None:
        cells = config["cell_numbers"][retina_index_str]
    cell_index = 0
    for cell in tqdm(cell_names_list):
        # fig, ax = plt.subplots()
        if cell_rfs is None:
            cell_rf = get_receptive_field(
                cell_id=cell,
                number_of_frames=num_of_frames,
                crop_size=crop_size,
                retina_str_index=retina_index_str,
                file_suffix=file_suffix,
                data_type=data_type,
            )

        else:
            cell_rf = cell_rfs[cell]
        max_value = np.max(np.abs(np.array(cell_rf)))
        frames = visualize_weights(
            cell_rf,
            ax[math.floor(cell_index / row_length), cell_index % row_length],
            vmin=-0.9 * max_value if vmin is None else vmin,
            vmax=0.9 * max_value if vmax is None else vmax,
            weight_index=weight_index,
            max_coordinate=max_coordinate,
            mask_border=mask_border,
            cmap=cmap,
        )

        for i, frame in enumerate(frames):
            all_frames[i].append(frame[0])
        if correlations is None:
            ax[math.floor(cell_index / row_length), cell_index % row_length].set_title(
                f"Cell {cell}"
                if cell_names is None
                else f"Cell {cell_names[cell_index]}"
            )
        else:
            ax[math.floor(cell_index / row_length), cell_index % row_length].set_title(
                f"Cell {cell}"
                if cell_names is None
                else f"Cell {cell_names[cell_index]}"
                + f"\n CC: {round(correlations[cell], 2)}"
            )
        # ax[math.floor(cell / row_length), cell % row_length].set_ticks([])
        cell_index += 1
    anim = animation.ArtistAnimation(
        fig, all_frames, interval=400, blit=True, repeat_delay=1000
    )
    if saving_file is None:
        anim.save(
            f"{home}/datasets/visualization_plots/cell_stas/{data_type}/retina_{retina_index_str}/{saving_file_suffix}.mp4"
        )
    else:
        anim.save(f"{saving_file}.mp4")


def get_cell_sta_from_pcs(
    cell_index, loadings, pca, num_of_loadings, num_of_frames=15, rf_size=10
):
    cell_loadings = loadings[0, cell_index, 0, :num_of_loadings]
    mean = pca.mean_.reshape(num_of_frames, rf_size, rf_size)

    pc_based_sta = (
        cell_loadings.numpy().reshape([1, -1]) @ pca.components_[:num_of_loadings, :]
    )
    pc_based_sta = pc_based_sta.reshape((num_of_frames, rf_size, rf_size)) + mean
    return pc_based_sta


def get_all_stas(
    retina_indices, input_kern, data_dir, h=150, w=200, file_suffix="_NC_stas"
):
    all_rfs_cropped = None
    for retina in retina_indices:
        all_cell_rf_fields_cropped, all_cell_rf_fields = get_cropped_stas(
            retina,
            num_of_images=input_kern[0],
            rf_size=input_kern[1:],
            h=h,
            w=w,
            data_dir=data_dir,
            file_suffix=file_suffix,
        )
        if all_rfs_cropped is None:
            all_rfs_cropped = all_cell_rf_fields_cropped
        else:
            all_rfs_cropped = np.concatenate(
                (all_rfs_cropped, all_cell_rf_fields_cropped), axis=0
            )

    return all_rfs_cropped, None


def get_stas_from_pcs(
    retina_index,
    input_kern,
    data_dir,
    plot=False,
    rf_size=10,
    file_suffix="_NC_stas",
    config=None,
    num_of_loadings=8,
    all_cell_rf_fields_cropped=None,
    predicted_retina_index_str=None,
):
    if config is None:
        config = global_config
    if all_cell_rf_fields_cropped is None:
        if isinstance(retina_index, list):
            all_cell_rf_fields_cropped, all_rf_fields = get_all_stas(
                retina_index,
                input_kern=input_kern,
                h=150,
                w=200,
                data_dir=data_dir,
                file_suffix=file_suffix,
            )
        else:
            all_cell_rf_fields_cropped, all_rf_fields = get_cropped_stas(
                retina_index,
                num_of_images=input_kern[0],
                rf_size=input_kern[1:],
                h=150,
                w=200,
                data_dir=data_dir,
                file_suffix=file_suffix,
            )

    true_rf_fields, loadings, pca = calculate_pca_on_cropped_stas(
        input_kern[0],
        retina_index=retina_index,
        data_dir=data_dir,
        rf_size=(rf_size, rf_size),
        all_cell_rf_fields_cropped=all_cell_rf_fields_cropped,
    )
    if isinstance(retina_index, list):
        num_of_cells = config["cell_numbers"][predicted_retina_index_str]
        cells_before = int(
            np.sum(
                [
                    config["cell_numbers"][key]
                    for key in config["cell_numbers"].keys()
                    if int(key[1]) < int(predicted_retina_index_str[1])
                ]
            )
        )
        loadings = loadings[cells_before : cells_before + num_of_cells]
    loadings = torch.tensor(loadings).unsqueeze(0).unsqueeze(2)
    pca_based_stas = np.zeros(
        (loadings.shape[1],) + all_cell_rf_fields_cropped.shape[1:]
    )
    retina_index = int(predicted_retina_index_str) - 1
    corrs = []
    for cell in range(config["cell_numbers"][f"0{retina_index + 1}"]):
        cell_0_pc_based_sta = get_cell_sta_from_pcs(
            cell_index=cell,
            loadings=loadings,
            pca=pca,
            num_of_loadings=num_of_loadings,
            num_of_frames=input_kern[0],
            rf_size=rf_size,
        )
        pca_based_stas[cell] = cell_0_pc_based_sta
        if plot:
            corr = correlation(
                torch.tensor(cell_0_pc_based_sta.flatten()),
                torch.tensor(all_cell_rf_fields_cropped[cell].flatten()),
                eps=1e-8,
            )
            corrs.append(corr.numpy()[0])
            if plot:
                fig, ax = plt.subplots(cell_0_pc_based_sta.shape[0], 2, figsize=(5, 10))
                for i in range(cell_0_pc_based_sta.shape[0]):
                    ax[i, 0].imshow(cell_0_pc_based_sta[i], cmap="coolwarm")
                    ax[i, 1].imshow(
                        all_cell_rf_fields_cropped[cell, i], cmap="coolwarm"
                    )

                ax[0, 0].set_title("PCA reconstruction")
                ax[0, 1].set_title("STA")
                plt.suptitle(f"Cell: {cell}, cc: {np.round(corr.numpy()[0], 2)}")
                plt.show()
            print(corrs)
            print(np.mean(corrs))
            plt.hist(corrs, bins=20)
            plt.xlabel("Correlation")
            plt.show()
            print("")
    return pca_based_stas, pca.mean_, all_cell_rf_fields_cropped


def calculate_initial_stas(responses, images, cell_index, time_bins, height, width):
    sta = np.zeros((time_bins, height, width))
    spike_count = 0
    cell_responses = responses[cell_index]

    #     for trial in tqdm(range(cell_responses.shape[-1])):
    #         trial_responses = cell_responses[:, trial]
    for i, response in enumerate(cell_responses):
        if i > time_bins and response > 0:
            sta += response * images[i - time_bins : i]
            spike_count += response
    if spike_count > 0:
        sta = sta / spike_count
    # sta /= np.linalg.norm(sta)
    return sta


def save_sta(
    response_file,
    image_files,
    image_dir,
    save_dir,
    save_file,
    time_bins,
    height,
    width,
    file_chunk=12,
):
    with open(response_file, "rb") as f:
        responses = pickle.load(f)
    responses = responses["train_responses"]
    responses = np.moveaxis(responses, 1, 2)
    # responses = responses[:, :, :11]
    # file_size = 5100
    print("loading images")
    all_stas = np.zeros((responses.shape[0],) + (time_bins, height, width))
    print("calculating stas")
    for i, image_file in enumerate(sorted(image_files)[:10]):
        print(image_file)
        # print(f'{i} - trial: {math.floor(i/file_chunk)}', f'responses: {(i%file_chunk)*file_size}:{(i%file_chunk +1)*file_size}')
        # image_responses = responses[:, (i % file_chunk)*file_size:(i % file_chunk + 1)*file_size, math.floor(i/file_chunk)]
        image_responses = responses[:, :, i]
        print(f"responses_shape: {image_responses.shape}")
        image = np.load(f"{image_dir}/{image_file}/all_images.npy")
        image = np.moveaxis(image, -1, 0)
        print(f"image shape: {image.shape}")
        # image /= 255
        # image = (1 - -1) * (image - 0) / (1 - 0) + -1
        height, width = image.shape[1:]
        # image = np.moveaxis(image, 2, 0)
        for cell in tqdm(range(responses.shape[0])):
            # for cell in tqdm(range(20)):
            sta = calculate_initial_stas(
                image_responses,
                image,
                cell_index=cell,
                time_bins=time_bins,
                height=height,
                width=width,
            )
            all_stas[cell] += sta
            # all_stas[cell] = sta
            # show_sta(sta, cell=cell)
            # normed_sta = sta / np.linalg.norm(sta)
            # show_sta(normed_sta, cell, vmin=np.min(normed_sta), vmax=np.max(normed_sta))
            # print(f"saving {save_file}_cell_{cell}")
            # np.save(f"{save_dir}/{save_file}_cell_{cell}", sta)

    # for cell in range(30):
    #     show_sta(all_stas[cell], cell=cell)
    #     print('normed')
    #     divided_sta = all_stas
    #     normed_sta = (divided_sta[cell])/np.linalg.norm(divided_sta[cell])
    #     show_sta(normed_sta, cell=cell, vmin=np.min(normed_sta), vmax=np.max(normed_sta))

    all_stas /= 25
    for cell in range(responses.shape[0]):
        np.save(f'{save_dir}/{save_file}_cell_{cell}', all_stas)

        show_sta(all_stas[cell], cell=cell)

    # all_stas /= len(image_files)



def show_sta(sta, cell, vmin=-1, vmax=1):
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(
        10,
        5,
        width_ratios=[1, 1, 1, 1, 1],
        wspace=0.05,
        hspace=0.05,
        top=0.95,
        bottom=0.05,
        left=0.17,
        right=0.845,
    )
    for i in range(sta.shape[0]):
        x = math.floor(i / 5)
        y = i % 5
        ax = plt.subplot(gs[x, y])
        ax.set_axis_off()
        ax.imshow(sta[i], cmap="gray", vmin=vmin, vmax=vmax)
    ax = plt.subplot(gs[0, 2])
    ax.set_title(f"Cell {cell}")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    # plt.tight_layout()
    plt.show()


# def lalal():
#     pass

if __name__ == "__main__":
    """with open(
        f"{home}/data/marmoset_data/responses/config_s4_nm_sta_zero_mean.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    plot_all_stas(
        "01",
        data_type="marmoset",
        file_suffix="_NC_stas",
        crop_size=30,
        saving_file_suffix="nm_stas_nm_marmoset_nm_all_cells",
        config=config_dict,
        cmap="gray",
    )
    exit()"""
    time_bins = 40
    image_files = os.listdir(
        f"{home}/data/marmoset_data/non_repeating_stimuli_frozennoise4x4bw1blink12750run2550freeze_85Hz_seed1/"
    )
    image_files = [x for x in image_files if x.startswith("trial")]
    # image_files = ['all_images.npy']
    save_sta(
        f"{home}/data/marmoset_data/responses/05_SS_frozennoise4x4bw1blink12750run2550freeze_85Hz_seed1.pkl",
        # f"{home}/data/marmoset_data/responses/cell_responses_02_fixation_movie.pkl",
        image_files=image_files,
        image_dir=f"{home}/data/marmoset_data/non_repeating_stimuli_frozennoise4x4bw1blink12750run2550freeze_85Hz_seed1/",
        save_dir=f"{home}/data/marmoset_data/stas/",
        save_file=f"cell_data_02_NC_stas",
        time_bins=time_bins,
        height=150,
        width=200,
    )

# calculate_pca_on_stas(num_of_images=15, retina_index=0, crop=big_crops[f'01'])
# get_stas_from_pcs(retina_index=0, input_kern=[15], data_dir=f'{home}/')
