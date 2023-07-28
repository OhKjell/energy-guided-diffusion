from pathlib import Path

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import yaml
import math
import seaborn as sns
from evaluations.ln_model_performance import get_ln_model_cell_performance_dict
from evaluations.single_cell_performance import get_basic_cnn_cell_performance_dict
from training.measures import (
    correlation,
    oracle_corr_conservative,
    oracle_corr_jackknife,
)
from utils.global_functions import (
    get_cell_names,
    get_cell_numbers_after_crop,
    get_cell_numbers,
    get_cell_numbers_after_crop,
    get_cell_names,
    get_exclude_cells_based_on_thresholds,
    home,
    global_config,
)


def calculate_retina_reliability(
    retina_index,
    control_stimuli_responses=None,
    response_dir=f"{home}/data/responses/",
    config_dict=None,
    fev_threshold=None,
):
    if config_dict is None:
        config_dict = global_config
    if control_stimuli_responses is None:
        control_stimuli_responses, _ = get_control_and_running_stimuli(
            retina_index, response_dir=response_dir, config=config_dict
        )
    if fev_threshold is not None:
        get_exclude_cells_based_on_thresholds(
            retina_index,
            config=config_dict,
            explainable_variance_threshold=fev_threshold,
        )
    # control_stimuli_responses = np.delete(control_stimuli_responses, get_exclude_cells_based_on_thresholds(retina_index, explainable_variance_threshold=fev_threshold), axis=0)
    print("cells:", fev_threshold, control_stimuli_responses.shape[0])
    avg_response = np.mean(control_stimuli_responses, axis=2)
    correlations = []
    for trial in range(control_stimuli_responses.shape[2]):
        avg_response = np.mean(
            np.delete(control_stimuli_responses, trial, axis=2), axis=2
        )
        corr = correlation(
            torch.tensor(control_stimuli_responses[:, :, trial], dtype=torch.float),
            torch.tensor(avg_response, dtype=torch.float),
            eps=1e-8,
            dim=1,
        )
        correlations.append(corr[:, 0].numpy())
    correlations = np.asarray(correlations)
    means = np.mean(correlations, axis=0)
    print(list(means))
    print(np.mean(means))
    return means


def calculate_r_squared(control_stimuli):
    predictions = np.array(
        [
            control_stimuli[:, :, x]
            for x in range(control_stimuli.shape[2])
            if x < int(control_stimuli.shape[-1] / 2)
        ]
    )  # x % 2 == 0])
    responses = np.array(
        [
            control_stimuli[:, :, x]
            for x in range(control_stimuli.shape[2])
            if x >= int(control_stimuli.shape[-1] / 2)
        ]
    )  # % 2 == 1])
    responses = np.moveaxis(responses, 1, 0)
    responses = np.moveaxis(responses, 1, 2)
    predictions = np.moveaxis(predictions, 1, 0)
    predictions = np.moveaxis(predictions, 1, 2)

    prediction = np.mean(predictions, axis=2)
    response = np.mean(responses, axis=2)
    variances_explained, eves = [], []

    # var_explained, eve = explained_variance(responses, prediction)
    # print('var explained:', variances_explained, 'explainable variance explained:', eve)
    corr = correlation(
        torch.tensor(response, dtype=torch.float),
        torch.tensor(prediction, dtype=torch.float),
        eps=1e-12,
        dim=1,
    )
    print("means corr cells:", corr)
    print("mean corr:", torch.mean(corr))
    return corr[:, 0]


def explained_variance(responses, predictions):
    mse = np.mean(
        (
            np.moveaxis(np.tile(predictions, (responses.shape[2], 1, 1)), 0, 2)
            - responses
        )
        ** 2,
        axis=(1, 2),
    )
    predictions = np.transpose(predictions)
    responses = np.transpose(responses)
    total_variance, explainable_var = [], []
    # responses = np.transpose(responses, axes=(1,2))
    for n in range(45):
        response = responses[:, :, n]
        obs_var = np.mean((np.var(response, axis=0, ddof=1)), axis=0)  # obs variance
        tot_var = np.var(response, axis=(0, 1), ddof=1)  # total variance
        total_variance.append(tot_var)
        explainable_var.append(tot_var - obs_var)  # explainable variance

    total_variance = np.array(total_variance)
    explainable_var = np.array(explainable_var)
    var_explained = total_variance - mse
    eve = var_explained / explainable_var
    return var_explained, eve


def calculate_repeated_variance(control_stimuli_responses):
    variance = np.mean(np.var(control_stimuli_responses, axis=(2), ddof=1), axis=1)
    return variance


def calculate_total_variance(control_stimuli_responses, running_stimuli_responses):
    # all_stimuli = np.concatenate((control_stimuli_responses, running_stimuli_responses), axis=1)
    all_stimuli = control_stimuli_responses
    total_variance = np.var(all_stimuli, axis=(1, 2), ddof=1)
    return total_variance


def get_control_and_running_stimuli(
    retina_index, response_dir=f"{home}/data/responses/", config=global_config
):
    file = config["files"][retina_index]
    with open(f"{response_dir}/{file}", "rb") as pkl:
        neural_data = pickle.load(pkl)
    # running_stimuli = np.moveaxis(neural_data['train_responses'], 1, 2)
    # control_stimuli = np.moveaxis(neural_data['test_responses'], 1, 2)
    running_stimuli = neural_data["train_responses"]
    control_stimuli = neural_data["test_responses"]
    return control_stimuli, running_stimuli


def calculate_explainable_variance_for_retina(
    control_stimuli, running_stimuli, retina_index, config=global_config
):
    if control_stimuli is None:
        control_stimuli, running_stimuli = get_control_and_running_stimuli(
            retina_index, config=config
        )
    # calculate_r_squared(control_stimuli)
    control_variance = calculate_repeated_variance(control_stimuli)
    total_variance = calculate_total_variance(control_stimuli, running_stimuli)
    explainable_variance = (total_variance - control_variance) / total_variance
    return explainable_variance


def plot_reliability_measure(
    retina_index,
    measure_values,
    model_file="lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0",
    reliability_measure="Fraction of explainable variance",
    directory_prefix="basic",
    plot=False,
    performance="validation",
    ln_model=False,
    xlabel=None,
    correlation_threshold=None,
    explainable_variance_threshold=None,
):
    # correlation_threshold = 0
    # explainable_variance_threshold = 0.4
    if ln_model:
        model_performance = get_ln_model_cell_performance_dict(
            model_file,
            retina_index=retina_index,
            correlation_threshold=correlation_threshold,
            explainable_variance_threshold=explainable_variance_threshold,
            exclude_cells=True,
        )
    else:
        (
            model_performance,
            correlation_threshold,
            explainable_variance_threshold,
        ) = get_basic_cnn_cell_performance_dict(
            filename=model_file,
            plot=plot,
            performance=performance,
            retina_index=retina_index,
            directory_prefix=directory_prefix,
        )

    print("avg model performance:", np.mean(list(model_performance.values())))
    fig, ax = plt.subplots(figsize=(10, 7))
    cell_names = get_cell_names(
        retina_index=retina_index,
        correlation_threshold=correlation_threshold,
        explained_variance_threshold=explainable_variance_threshold,
    )
    exclude_cells = get_exclude_cells_based_on_thresholds(
        retina_index=retina_index,
        correlation_threshold=correlation_threshold,
        explainable_variance_threshold=explainable_variance_threshold,
    )
    measure_values = np.delete(measure_values, exclude_cells)
    plt.scatter(list(model_performance.values()), measure_values, color="deepskyblue")
    for cell in range(
        get_cell_numbers_after_crop(
            retina_index=retina_index,
            correlation_threshold=correlation_threshold
            if correlation_threshold is not None
            else 0,
            explained_variance_threshold=explainable_variance_threshold
            if explainable_variance_threshold is not None
            else 0,
        )
    ):
        plt.annotate(
            cell_names[cell],
            xy=(list(model_performance.values())[cell], measure_values[cell]),
        )
    plt.plot(
        np.arange(0, 0.9, 0.1), np.arange(0, 0.9, 0.1), color="black", linestyle="--"
    )
    plt.plot([0, 1], [0.15, 0.15], color="red", linestyle="--", label="threshold")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(reliability_measure)
    plt.title(f"Retina {retina_index + 1}")
    plt.show()
    corr = correlation(
        torch.tensor(list(model_performance.values())),
        torch.tensor(measure_values),
        eps=1e-8,
    )
    print(f"Correlation between {reliability_measure} and performance", corr)
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.hist(measure_values, bins=20, color="hotpink")
    plt.xlim(-0.1, 1)
    plt.xlabel(reliability_measure)
    plt.ylabel("Num of cells")
    plt.title(f"Retina {retina_index + 1}")
    plt.show()
    print(measure_values)
    print(f"{reliability_measure}:", np.mean(measure_values))
    # print(np.sum(np.where(explainable_variance >= threshold, 1, 0)))
    # print([x for x in range(explainable_variance.shape[0]) if explainable_variance[x] < 0.15])


def get_ln_performance_reliability_corr(
    retina_index,
    explainable_variance,
    performance,
    corr_thresholds,
    model_str="LN model",
):
    performances = []
    cell_names = get_cell_names(retina_index)

    for corr_threshold in corr_thresholds:
        exclude_cells = get_exclude_cells_based_on_thresholds(
            retina_index,
            explainable_variance_threshold=corr_threshold,
            correlation_threshold=None,
        )
        corr_explainable_variance = [
            perf
            for i, (perf, fev) in enumerate(zip(performance, explainable_variance))
            if fev > corr_threshold and i in cell_names
        ]
        corr_explainable_variance_2 = np.delete(performance, exclude_cells)
        perf_mean = np.mean(corr_explainable_variance)
        # perf_mean_2 = np.mean(corr_explainable_variance_2)
        performances.append(perf_mean)
    perf_dict = {x: y for x, y in zip(corr_thresholds, performances)}
    print([f"{x}: {y}" for x, y in zip(corr_thresholds, performances)])

    return perf_dict


def get_cnn_performance_reliability_corr(
    retina_index, corr_thresholds, model_file, prefix, data_type="salamander"
):
    performances = []

    for threshold in corr_thresholds:
        threhold_str = "" if threshold <= 0 else f"ev_{threshold:.2f}_"
        file = os.path.join(
            f"{home}/models/{prefix}_{threhold_str}cnn/{data_type}/retina{retina_index+1}/cell_None/readout_isotropic_gaussian/gmp_0/{model_file}",
            "stats",
            "correlations.npy",
        )
        if os.path.isfile(file):
            correlation = np.load(file)
            max_corr = np.max(correlation)
            performances.append(max_corr)
        else:
            performances.append(float("nan"))

    perf_dict = {x: y for x, y in zip(corr_thresholds, performances)}
    print([f"{x}: {y}" for x, y in zip(corr_thresholds, performances)])
    return perf_dict


def plot_model_performances_reliability_corrs(
    thresholds, model_performances, labels, retina_index
):
    assert len(model_performances) == len(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, perf in zip(labels, model_performances):
        if "Oracle" in label:
            plt.plot(thresholds, perf, label=label, color="black", linestyle="--")
        else:
            plt.plot(thresholds, perf, label=label)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel("FEV threshold")
    plt.ylabel(f"Correlation coefficient")
    plt.title(f"Retina {retina_index+1}")
    plt.tight_layout()
    plt.show()


def get_reliability_plot(
    control_responses, cell, r_squared, reliabilities, orracle_corr_cons
):
    cell_r_squared = r_squared[cell]
    cell_reliability = reliabilities[cell]
    cell_responses = control_responses[cell]
    # chunk = int(cell_responses.shape[0]/6)
    chunk = 850
    fig, ax = plt.subplots(4, 3, figsize=(40, 20))
    x = [0, 2, 0, 2, 0, 2]
    y = [0, 0, 1, 1, 2, 2]
    for frames in range(5):
        print(frames)
        total_spikes = np.sum(
            cell_responses[frames * chunk : (frames + 1) * chunk], axis=1
        )
        ax[x[frames], y[frames]].plot(total_spikes, color="black")
        ax[x[frames], y[frames]].set_title(
            f"frames {frames*chunk} to {(frames+1)*chunk}", fontsize=20
        )
        for i in range(cell_responses.shape[1]):
            ax[x[frames] + 1, y[frames]].scatter(
                [
                    min(x * response, x)
                    for x, response in enumerate(
                        cell_responses[frames * chunk : (frames + 1) * chunk, i]
                    )
                ],
                [20 - i] * chunk,
                color="orange" if i >= 11 else "blue",
            )
    ax[0, 0].set_title(
        f"Cell {cell}\nR squared: {cell_r_squared:.2f}, Reliability: {cell_reliability:.2f}",
        fontsize=25,
    )
    fig.subplots_adjust(hspace=0.2)
    Path(
        f"{home}/datasets/visualization_plots/marmoset_data/retina1/cell_{cell}"
    ).mkdir(exist_ok=True)
    plt.savefig(
        f"{home}/datasets/visualization_plots/marmoset_data/retina1/cell_{cell}/cell_frozen_responses_01_fixation_movie.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    retina_index = 1
    with open(f"{home}/data/marmoset_data/responses/config_s4_2022.yaml") as f:
        config = yaml.unsafe_load(f)
    response_dir = f"{home}/data/marmoset_data/responses"
    control_stimuli, running_stimuli = get_control_and_running_stimuli(
        retina_index, response_dir=response_dir, config=config
    )
    # control_stimuli_22 = control_stimuli[:, :, :10]
    # control_stimuli_23 = control_stimuli[:, :, 10:]
    # running_stimuli_22 = running_stimuli[:, :, :10]
    # running_stimuli_23 = running_stimuli[:, :, 10:]
    # r_squared = calculate_r_squared(control_stimuli)
    explainable_variance_22 = calculate_explainable_variance_for_retina(
        control_stimuli=control_stimuli,
        running_stimuli=running_stimuli,
        retina_index=retina_index,
    )
    # explainable_variance_23 = calculate_explainable_variance_for_retina(control_stimuli=control_stimuli_23,
    #                                                                  running_stimuli=running_stimuli_23,
    #                                                                  retina_index=retina_index)
    # explainable_variance = (np.array(explainable_variance_23) + np.array(explainable_variance_22))/2
    # np.save(
    #     f"{home}/datasets/marmoset_data/reliabilities/retina2/explainable_variance_wn.npy",
    #     explainable_variance_22,
    # )
    # np.save(f'{home}/datasets/marmoset_data/reliabilities/retina1/explainable_variance_seed_23.npy', explainable_variance_23)
    print(list(explainable_variance_22))
    print(np.nanmean(explainable_variance_22))
    reliabilities = calculate_retina_reliability(
        retina_index=retina_index,
        control_stimuli_responses=control_stimuli,
        response_dir=response_dir,
        fev_threshold=None,
    )
    correlation_threshold = 0
    plt.scatter(r_squared, explainable_variance_22)
    plt.xlabel("R_squared")
    plt.ylabel("Explainable variance")
    plt.title("White noise salamander retina 02")
    plt.show()
    print("correlation", list(reliabilities))
    print("explainable variance", list(explainable_variance_22))
    for cell in range(control_stimuli.shape[0]):
        get_reliability_plot(control_stimuli, cell, r_squared, reliabilities, None)
    explainable_variance_threshold = 0.15
    orracle_corr_cons = oracle_corr_conservative(np.moveaxis(control_stimuli, 0, 2))
    perf_dict_2 = get_ln_model_cell_performance_dict(
        "lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0",
        retina_index=retina_index,
        explainable_variance_threshold=explainable_variance_threshold,
        correlation_threshold=correlation_threshold,
        exclude_cells=False,
    )
    corr_thresholds = [0, 0.15, 0.3, 0.45, 0.6]
    ln_reliability_corr = get_ln_performance_reliability_corr(
        retina_index,
        explainable_variance,
        performance=list(perf_dict_2.values()),
        corr_thresholds=corr_thresholds,
    )

    cnn_reliability_corr = get_cnn_performance_reliability_corr(
        retina_index,
        corr_thresholds,
        model_file="lr_0.0007_l_1_ch_8_t_15_bs_128_tr_250_s_1_ik_15x11x11_g_2.0_gt_0.0_l1_0.1_l2_0.0_sg_0.15_p_0_bn_0_str_1",
        prefix="basic",
        model_str="1 layer CNN",
    )
    cnn_reliability_corr_3_layer = get_cnn_performance_reliability_corr(
        retina_index,
        corr_thresholds,
        model_file="lr_0.007_l_3_ch_8_t_7_bs_128_tr_250_s_1_ik_7x5x5_hk_7x5x5_g_1.0_gt_0.0_l1_0.1_l2_0.0_sg_0.15_p_0_bn_1_str_1",
        prefix="basic",
        model_str="3 layer CNN",
    )
    orracle_corts = []
    for corr_threshold in corr_thresholds:
        orracle_corts.append(
            np.mean(calculate_retina_reliability(retina_index, corr_threshold))
        )

    if retina_index == 0:
        cnn_reliability_factorized = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_1_ch_8_t_15_g_0.5_bs_128_tr_250_s_1_cr_(50, 50, 75, 65)_ik_11_hk_5_l1_1.2_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="1 layer factorized CNN",
        )
        cnn_reliability_factorized_3_layers = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_3_ch_8_t_15_g_5.0_bs_128_tr_250_s_1_cr_(50, 50, 75, 65)_ik_5_hk_5_l1_1.5_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="4 layer factorized CNN",
        )
        cnn_reliability_factorized_4_layers = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_4_ch_8_t_15_g_2.0_bs_128_tr_250_s_1_cr_(50, 50, 75, 65)_ik_5_hk_5_l1_1.0_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="4 layer factorized CNN",
        )

    elif retina_index == 3:
        cnn_reliability_factorized = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_1_ch_8_t_15_g_0.5_bs_128_tr_250_s_1_cr_(60, 40, 70, 70)_ik_11_hk_5_l1_1.2_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="1 layer factorized CNN",
        )
        cnn_reliability_factorized_3_layers = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_3_ch_8_t_7_g_5.0_bs_128_tr_250_s_1_cr_(60, 40, 70, 70)_ik_5_hk_5_l1_1.5_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="3 layer factorized CNN",
        )
        cnn_reliability_factorized_4_layers = get_cnn_performance_reliability_corr(
            retina_index,
            corr_thresholds,
            model_file="lr_0.001_l_4_ch_8_t_7_g_2.0_bs_128_tr_250_s_1_cr_(60, 40, 70, 70)_ik_5_hk_5_l1_1.0_mu_0.7_sg_0.15_p_0",
            prefix="factorized",
            model_str="4 layer factorized CNN",
        )

    plot_model_performances_reliability_corrs(
        corr_thresholds,
        [
            list(ln_reliability_corr.values()),
            list(cnn_reliability_corr.values()),
            list(cnn_reliability_corr_3_layer.values()),
            list(cnn_reliability_factorized.values()),
            list(cnn_reliability_factorized_3_layers.values()),
            list(cnn_reliability_factorized_4_layers.values()),
            orracle_corts,
        ],
        labels=[
            "LN model",
            "1 layer CNN model",
            "3 layer CNN model",
            "1 layer factorized CNN",
            "3 layer factorized CNN",
            "4 layer factorized CNN",
            "Oracle correlation",
        ],
        retina_index=retina_index,
    )

    print(f"avg conservative corr:", np.mean(orracle_corr_cons))
    orracle_corr_jack = oracle_corr_jackknife(np.moveaxis(control_stimuli, 0, 2))
    print(f"avg jack corr:", np.mean(orracle_corr_jack))
    # plot_reliability_measure(retina_index, measure_values=explainable_variance,
    #                          ln_model=False, reliability_measure='FEV', xlabel='CNN model',
    #                          model_file='lr_0.001_l_1_ch_8_t_15_bs_128_tr_250_s_1_ik_15x11x11_g_2.0_gt_0.0_l1_0.1_l2_0.0_sg_0.15_p_0_bn_0_str_1',
    #                          directory_prefix='basic_ev_0.15')
    # plot_reliability_measure(retina_index, measure_values=explainable_variance,
    #                          ln_model=True, reliability_measure='FEV', xlabel='LN model')
    plot_reliability_measure(measure_values=reliabilities, retina_index=retina_index)
    # oracle_correlation = calculate_retina_reliability(retina_index)
    # print(np.sum(np.where(oracle_correlation >= 0.3, 1, 0)))
