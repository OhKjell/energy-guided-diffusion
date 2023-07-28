import math

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from matplotlib import animation
from sklearn.linear_model import LinearRegression

from models.model_visualizations import get_receptive_field
from utils.global_functions import (
    big_crops,
    cell_numbers_after_crop,
    cell_numbers,
    home,
)
import seaborn as sns


def plot_curve(training_data, label, title=None):
    plt.plot(np.arange(training_data), training_data, label=label)
    plt.xlabel("Epochs")
    plt.ylabel(f"{label} value")
    if title is not None:
        plt.title(title)
    plt.show()


def show_rf_field_centers_for_retina(retina_index, img_h=150, img_w=200, rfs=None):
    rfs = np.load(
        f"{home}/data/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_25.npy"
    )

    whole_rf = np.zeros((img_h, img_w))
    all_temporal_variances = np.zeros((img_h, img_w))
    centers = []
    crop = big_crops[str(retina_index + 1).zfill(2)]
    for cell in range(rfs.shape[0]):
        temporal_variances = np.var(rfs[cell], axis=0)
        temporal_variances = np.where(
            temporal_variances > np.max(temporal_variances * 0.1), temporal_variances, 0
        )
        # plt.imshow(temporal_variances, cmap='coolwarm')
        # plt.show()
        all_temporal_variances += temporal_variances
        rf_center = np.unravel_index(np.argmax(temporal_variances), (img_h, img_w))
        whole_rf[rf_center] = 1
        centers.append(rf_center)
    whole_rf = whole_rf[crop[0] : -crop[1], crop[2] : -crop[3]]

    plt.imshow(np.zeros(whole_rf.shape), cmap="Blues")
    for cell in range(rfs.shape[0]):
        if (
            (centers[cell][0] > crop[0])
            and (centers[cell][0] < (150 - crop[1]))
            and (centers[cell][1] > crop[2])
            and (centers[cell][1] < 150)
        ):
            plt.scatter(
                [centers[cell][1] - crop[2]],
                [centers[cell][0] - crop[0]],
                color="orange",
                marker="o",
                s=[7],
            )
            plt.annotate(
                str(cell), (centers[cell][1] - crop[2], centers[cell][0] - crop[0])
            )

    plt.savefig(
        f"{home}/datasets/visualization_plots/all_cell_rfs_for_retinas/cell_rfs_retina_{retina_index}.png"
    )
    plt.show()
    crop = (42, 58, 70, 70)
    crop = big_crops[str(retina_index + 1).zfill(2)]
    all_temporal_variances = all_temporal_variances[
        crop[0] : -crop[1], crop[2] : -crop[3]
    ]
    fig, ax = plt.subplots()
    plt.imshow(all_temporal_variances, cmap="coolwarm")
    for cell in range(rfs.shape[0]):
        # if (centers[cell][0] > crop[0]) and (centers[cell][0] < (150-crop[1])) and (
        #         centers[cell][1] > crop[2]) and (centers[cell][1] < 150):
        ax.scatter(
            [centers[cell][1] - crop[2]],
            [centers[cell][0] - crop[0]],
            color="red",
            marker="o",
            s=[7],
        )
        ax.annotate(str(cell), (centers[cell][1] - crop[2], centers[cell][0] - crop[0]))
    plt.show()


def show_receptive_fields(
    cell_index, retina_index, half_rf_size=30, img_h=150, img_w=200
):
    rfs = np.load(
        f"{home}/data/cell_data_{str(retina_index + 1).zfill(2)}_NC_stas_25.npy"
    )
    cell_rfs = rfs[cell_index]
    temporal_variances = np.var(cell_rfs, axis=0)
    max_coordinate = np.unravel_index(np.argmax(temporal_variances), (img_h, img_w))
    size = (
        max_coordinate[0]
        + min(half_rf_size, img_h - max_coordinate[0])
        - (max_coordinate[0] - min(half_rf_size, max_coordinate[0])),
        (
            max_coordinate[1]
            + min(half_rf_size, img_h - max_coordinate[1])
            - (max_coordinate[1] - min(half_rf_size, max_coordinate[1]))
        ),
    )
    for rf in cell_rfs:
        plt.imshow(rf, cmap="coolwarm")
        plt.plot([max_coordinate[1]], max_coordinate[0], color="green", marker="o")
        plt.plot(
            [max_coordinate[1] - min(size[1] // 2, max_coordinate[1])],
            [max_coordinate[0] - min(size[0] // 2, max_coordinate[0])],
            color="orange",
            marker="o",
            markersize=8,
        )
        plt.plot(
            [max_coordinate[1] - min(size[1] // 2, max_coordinate[1])],
            [
                max_coordinate[0]
                + min(size[0] // 2 + size[1] % 2, img_w - max_coordinate[0])
            ],
            color="orange",
            marker="o",
            markersize=8,
        )
        plt.plot(
            [
                max_coordinate[1]
                + min(size[1] // 2 + size[1] % 2, img_h - max_coordinate[1])
            ],
            [
                max_coordinate[0]
                + min(size[0] // 2 + size[1] % 2, img_w - max_coordinate[0])
            ],
            color="orange",
            marker="o",
            markersize=8,
        )
        plt.plot(
            [
                max_coordinate[1]
                + min(size[1] // 2 + size[1] % 2, img_h - max_coordinate[1])
            ],
            [max_coordinate[0] - min(size[0] // 2, max_coordinate[0])],
            color="orange",
            marker="o",
            markersize=8,
        )

        # plt.show()
    plt.show()
    cut_cell_rfs = cell_rfs[
        :,
        max_coordinate[0]
        - min(size[0] // 2, max_coordinate[0]) : max_coordinate[0]
        + min(size[0] // 2 + size[0] % 2, img_h - max_coordinate[0]),
        max_coordinate[1]
        - min(size[1] // 2, max_coordinate[1]) : max_coordinate[1]
        + min(size[1] // 2 + size[1] % 2, img_w - max_coordinate[1]),
    ]
    for cut_rf in cut_cell_rfs:
        # plt.show([max_coordinate[0]], [max_coordinate[1]])
        plt.imshow(cut_rf, cmap="coolwarm")
        plt.show()


def get_avg_cell_responses(
    test_responses, cell_id, trial_bin_size, step_size=None, scale=False
):
    avg_response = np.mean(np.sum(test_responses[cell_id, :, :], axis=0), axis=0)
    if step_size is None:
        step_size = trial_bin_size

    distance_list = []
    trial_bin_responses = []

    for i in range(test_responses.shape[2] // step_size):
        step_response = np.mean(
            np.sum(
                test_responses[
                    cell_id, :, i * step_size : i * step_size + trial_bin_size
                ],
                axis=0,
            ),
            axis=0,
        )
        distance = np.abs(avg_response - step_response)
        if scale:
            distance = distance / max(step_response, 1)
        distance_list.append(distance)
        trial_bin_responses.append(step_response)

    return distance_list, avg_response, trial_bin_responses


def plot_cell_avg_responses(
    num_of_trials,
    avg_responses,
    bin_avg_responses,
    trial_bin_size,
    step_size,
    retina_number,
    cell_id,
):
    plt.plot(
        np.arange(num_of_trials),
        [avg_responses] * num_of_trials,
        label="avg response",
        linestyle="--",
        color="red",
    )

    for i, step_response in enumerate(bin_avg_responses):
        plt.plot(
            np.arange(i * step_size, i * step_size + trial_bin_size, 1),
            [step_response] * trial_bin_size,
            color="orange",
            label="avg response for trials",
        )
        # TODO: somehow scale this to make it relative to the response counts, i.e. if a cell response count is 25,
        #  and distance 1 it should matter less than if the response count is 3
    plt.ylabel("Response Counts")
    plt.xlabel("Trials")
    plt.title(f"Cell {cell_id}, Retina {retina_number}")
    plt.savefig(
        f"{home}/datasets/visualization_plots/cell_avg_responses_comparison/retina_{retina_number}/cell_{cell_id}_bin_size_{trial_bin_size}.png"
    )
    plt.show()


def plot_cell_response_variations(
    test_responses,
    trial_bin_size,
    step_size=None,
    cells_to_plot=None,
    retina_number=1,
    scale=False,
):
    if cells_to_plot is None:
        cells_to_plot = []
    if step_size is None:
        step_size = trial_bin_size

    all_cell_distances = []

    for cell_id in range(test_responses.shape[0]):
        (
            cell_distances,
            cell_avg_response,
            cell_trial_bin_responses,
        ) = get_avg_cell_responses(
            test_responses, cell_id, trial_bin_size, step_size, scale=scale
        )
        all_cell_distances.append(cell_distances)
        if cell_id in cells_to_plot:
            plot_cell_avg_responses(
                test_responses.shape[2],
                avg_responses=cell_avg_response,
                bin_avg_responses=cell_trial_bin_responses,
                trial_bin_size=trial_bin_size,
                step_size=step_size,
                retina_number=retina_number,
                cell_id=cell_id,
            )
    all_cell_distances = np.asarray(all_cell_distances)
    if scale:
        fig, ax = plt.subplots()
        axis = ax
    else:
        fig, ax = plt.subplots(2, 1)
        axis = ax[0]

    for i in range(test_responses.shape[2] // step_size):
        mean = np.mean(all_cell_distances[:, i])
        std = np.std(all_cell_distances[:, i])
        bin = i * step_size, i * step_size + trial_bin_size
        axis.plot(
            np.arange(bin[0], bin[1]),
            [mean] * trial_bin_size,
            color="orange",
            label="Response count" "\ndifference from mean",
        )
        axis.fill_between(
            np.arange(bin[0], bin[1]),
            [mean] * trial_bin_size,
            [mean - std] * trial_bin_size,
            alpha=0.2,
            color="orange",
            label="std",
        )
        axis.fill_between(
            np.arange(bin[0], bin[1]),
            [mean] * trial_bin_size,
            [mean + std] * trial_bin_size,
            alpha=0.2,
            color="orange",
        )
        axis.set_title(f"Retina {retina_number}")
    axis.set_ylabel("Response count difference \nfrom mean")
    if not scale:
        avg_firing_rate = np.mean(np.sum(test_responses, axis=1), axis=0)
        ax[1].plot(
            np.arange(test_responses.shape[2]),
            avg_firing_rate,
            linestyle="--",
            color="green",
        )
        ax[1].set_ylim(0, 28)
        ax[1].set_ylabel("Mean response counts")
        scale_str = ""
    if scale:
        scale_str = "_scaled"
    plt.xlabel("Trials")
    plt.tight_layout()
    plt.savefig(
        f"{home}/datasets/visualization_plots/response_count_difference_from_mean/retina_{retina_number}_bin_size_{trial_bin_size}{scale_str}.png"
    )
    plt.show()


if __name__ == "__main__":
    # files = ['cell_data_01_NC.mat.pickle', 'cell_data_02_NC.mat.pickle', 'cell_data_03_NC.mat.pickle',
    #          'cell_data_04_NC.mat.pickle', 'cell_data_05_NC.mat.pickle']
    # dir = '{home}/data/responses/'
    # slope_vectors = []
    # fig, ax = plt.subplots(1, 5, sharey='row', figsize=(20, 5))
    # for i, file in enumerate(files):
    #     with open(f'{dir}/{file}', 'rb') as pkl:
    #         neural_data = pickle.load(pkl)
    #     test_responses = neural_data['test_responses']
    #     plot_cell_response_variations(test_responses, 10, step_size=10, cells_to_plot=[0, 6, 8, 20, 32, 56], retina_number=i+1, scale=True)
    # test_responses = test_responses[:, 10:, :]
    # slopes = linear_regression(test_responses, retina_number=i+1, ax=ax[i])
    # slope_vectors.append(slopes)
    # plt.savefig('./visualization_plots/individual_cell_slopes_cut.png')
    # plt.show()
    # sns.boxplot(data=slope_vectors)
    # plt.ylabel('Slope value')
    # plt.xlabel('Retina number')
    # plt.xticks(ticks=[0, 1, 2, 3, 4], labels=np.arange(1, 6, 1))
    # plt.tight_layout()
    # plt.savefig('./visualization_plots/slopes_iqr_cut.png')
    # plt.show()
    # show_receptive_fields(22, 4)
    for i in range(5):
        show_rf_field_centers_for_retina(i)
