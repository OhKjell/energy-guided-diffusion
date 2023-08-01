from pathlib import Path

import torch
import yaml
from neuralpredictors.training import LongCycler
from tqdm import tqdm
from nnfabrik import builder
import os
import seaborn as sns
import dynamic.models
from dynamic.datasets.stas import get_stas_from_pcs, get_cropped_stas, plot_all_stas
from dynamic.evaluations.ln_model_performance import (
    get_ln_model_cell_performance_dict,
    get_model_and_dataloader_for_ln,
    get_parameters_from_ln_model_file_name,
    get_ln_model_performance_on_test,
)
from dynamic.models.helper_functions import get_seed_model_versions
from dynamic.models.helper_functions import (
    get_model_and_dataloader,
    get_model_and_dataloader_for_nm,
    get_wn_model_and_dataloader_for_nm,
)
from dynamic.models.ln_model import Model
from dynamic.training.regularizers import TimeLaplaceL23d
from dynamic.training.trainers import model_step
from dynamic.training.measures import correlation, variance_of_predictions
import random
import numpy as np
from dynamic.utils.global_functions import (
    global_config,
    model_seed,
    dataset_seed,
    home,
    get_cell_names,
    get_cell_numbers,
)
import matplotlib.pyplot as plt

base_path = "./models/ln_models/"
home_dir = "./"

neuronal_data_path = os.path.join(home, "data/responses/")
# neuronal_data_path = os.path.join(basepath, 'data/dummy_data/')
training_img_dir = os.path.join(home, "data/non_repeating_stimuli/")
test_img_dir = os.path.join(home, "data/repeating_stimuli/")


def get_performance_for_single_cell(
    dataloaders,
    model,
    retina_index,
    device,
    rf_size,
    max_coordinate=None,
    img_h=150,
    img_w=200,
    performance="validation",
):
    all_responses = []
    all_predictions = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, responses in tqdm(
            dataloaders[performance][str(retina_index + 1).zfill(2)]
        ):
            images = images.double().to(device)
            responses = responses.transpose(1, 2)
            responses = torch.flatten(responses, start_dim=0, end_dim=1).to(device)
            responses = responses.to(device)
            if device == "cpu":
                all_responses.append(responses.detach().numpy())
            else:
                all_responses.append(responses.detach().cpu().numpy())
            output = model_step(
                images=images,
                model=model,
                max_coordinate=max_coordinate,
                rf_size=rf_size,
                h=img_h,
                w=img_w,
            )

            if device == "cpu":
                all_predictions.append(output.detach().numpy())

            else:
                all_predictions.append(output.detach().cpu().numpy())
        correlations = torch.squeeze(
            correlation(
                torch.tensor(np.concatenate(all_predictions)),
                torch.tensor(np.concatenate(all_responses)),
                eps=1e-12,
            )
        )
    all_predictions = np.concatenate(all_predictions)
    all_responses = np.concatenate(all_responses)
    # np.save(f'{home}/logs/preds_0_on_2022.npy', all_predictions)
    return correlations, all_predictions, all_responses


def get_multiretinal_performance_for_singel_cells(dataloaders,
    model,
    device,
    rf_size,
    max_coordinate=None,
    img_h=150,
    img_w=200,
    performance="validation",):
    all_responses = {data_key: [] for data_key in dataloaders[performance].keys()}
    all_predictions = {data_key: [] for data_key in dataloaders[performance].keys()}
    model.to(device)
    model.eval()
    n_iterations = len(LongCycler(dataloaders["train"]))
    with torch.no_grad():
        for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders[performance])),
                total=n_iterations,
                # desc="Epoch {}".format(epoch),
        ):
            images = data[0]
            responses = data[1]
            images = images.double().to(device)
            responses = responses.transpose(1, 2)
            responses = torch.flatten(responses, start_dim=0, end_dim=1).to(device)
            responses = responses.to(device)
            if device == "cpu":
                all_responses[data_key].append(responses.detach().numpy())
            else:
                all_responses[data_key].append(responses.detach().cpu().numpy())
            output = model_step(
                images=images,
                model=model,
                max_coordinate=max_coordinate,
                rf_size=rf_size,
                h=img_h,
                w=img_w,
                data_key=data_key
            )

            if device == "cpu":
                all_predictions[data_key].append(output.detach().numpy())

            else:
                all_predictions[data_key].append(output.detach().cpu().numpy())
        correlations = {}
        for data_key in dataloaders[performance].keys():
            correlations[data_key] = torch.squeeze(
            correlation(
                torch.tensor(np.concatenate(all_predictions[data_key])),
                torch.tensor(np.concatenate(all_responses[data_key])),
                eps=1e-12,
            )
        )
            all_predictions[data_key] = np.concatenate(all_predictions[data_key])
            all_responses[data_key] = np.concatenate(all_responses[data_key])
    # np.save(f'{home}/logs/preds_0_on_2022.npy', all_predictions)
    return correlations, all_predictions, all_responses


def get_basic_cnn_cell_performance_across_seeds(
    seeds,
    filename,
    retina_index=0,
    performance="validation",
    directory_prefix="basic",
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    average=True,
):
    dicts = []

    for seed in seeds:
        cell_dict, _, _ = get_basic_cnn_cell_performance_dict(
            filename=filename,
            plot=False,
            retina_index=retina_index,
            performance=performance,
            directory_prefix=directory_prefix,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            seed=seed,
        )
        dicts.append(cell_dict)
    if average:
        final_dict = {k: [] for k in dicts[0].keys()}
        for d in dicts:
            for k, v in d.items():
                final_dict[k].append(v)
        for k, v in final_dict.items():
            final_dict[k] = np.mean(final_dict[k])
        return final_dict
    return dicts


def get_basic_cnn_ensemble_cell_performance(
    seeds,
    filename,
    rf_size,
    retina_index=0,
    performance="validation",
    directory_prefix="basic",
    data_type="salamander",
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    max_coordinate=None,
    img_h=150,
    img_w=200,
):
    directory = f"{home}/models/{directory_prefix}_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    models, seeds = get_seed_model_versions(
        filename, model_dir=directory, model_fn=model_fn, device=device, seeds=seeds
    )
    dataloaders, _, config = get_model_and_dataloader(
        directory,
        filename,
        model_fn=model_fn,
        device=device,
        data_dir=data_dir,
        test=performance == "test",
        seed=seeds[0],
    )
    all_predictions, all_responses = [], []
    with torch.no_grad():
        for images, responses in tqdm(
            dataloaders[performance][str(retina_index + 1).zfill(2)]
        ):
            images = images.double().to(device)
            responses = responses.transpose(1, 2)
            responses = torch.flatten(responses, start_dim=0, end_dim=1).to(device)
            responses = responses.to(device)
            if device == "cpu":
                all_responses.append(responses.detach().numpy())
            else:
                all_responses.append(responses.detach().cpu().numpy())
            outputs = []
            for model in models:
                output = model_step(
                    images=images,
                    model=model,
                    max_coordinate=max_coordinate,
                    rf_size=rf_size,
                    h=img_h,
                    w=img_w,
                )
                outputs.append(output.detach().cpu().numpy())
            avg_output = np.array(outputs)
            print(avg_output.shape)
            avg_output = np.mean(avg_output, 0)
            if device == "cpu":
                all_predictions.append(avg_output.detach().numpy())

            else:
                all_predictions.append(avg_output)
    correlations = torch.squeeze(
        correlation(
            torch.tensor(np.concatenate(all_predictions)),
            torch.tensor(np.concatenate(all_responses)),
            eps=1e-12,
        )
    )
    return correlations, all_predictions, all_responses


def get_multiretinal_cnn_perofrmance_dict(
    filename,
        performance='validation',
        directory_prefix='multiretinal_factorized',
        data_type='salamander',
        model_fn='models.MultiRetinalFactorizedEncoder.build_trained',
    device="cpu",
    data_dir=None,
    seed=None,
    config_dict=None,
    fixation_file=None,


):
    directory = f"{home}/models/{directory_prefix}_cnn/{data_type}/retinaall/cell_None/readout_isotropic/gmp_0/"
    dataloaders, model, config = get_model_and_dataloader(
        directory,
        filename,
        model_fn=model_fn,
        device=device,
        data_dir=data_dir,
        test=performance == "test",
        seed=seed,
        data_type=data_type,
        config_dict=config_dict,
    )
    correlations, all_predictions, all_responses = get_multiretinal_performance_for_singel_cells(
        dataloaders=dataloaders,
        model=model,
        device=device,
        rf_size=(model.config_dict["img_h"], model.config_dict["img_w"]),
        img_h = model.config_dict["img_h"],
        img_w = model.config_dict["img_w"],
        performance=performance
    )
    cell_names = {}
    all_cells = []
    for data_key in correlations.keys():
        cell_names[data_key] = get_cell_names(retina_index=int(data_key)-1,
        explained_variance_threshold=model.config_dict[
            "explainable_variance_threshold"
        ],
        correlation_threshold=model.config_dict["oracle_correlation_threshold"],
        config=model.config_dict["config"],)
        np.save(f"{directory}/{filename}/stats/seed_{seed}/cell_ids_retina_{data_key}.npy", cell_names)
        np.save(
            f"{directory}/{filename}/stats/seed_{seed}/test_correlations_retina_{data_key}.npy",
            correlations,
        )
        print(
            f"saved_correlations to {directory}/{filename}/stats/seed_{seed}/test_correlations_{data_key}.npy"
        )
        print(f'retina: {data_key}')
        for cell_name, corr in zip(cell_names[data_key], correlations[data_key]):
            print(f'cell: {cell_name} - corr: {corr}')
        print(np.mean(correlations[data_key].detach().numpy()))
        all_cells += list(correlations[data_key].detach().numpy())
    print(f'per cell avg: {np.mean(all_cells)}')




def get_basic_wm_cnn_cell_performance_dict_for_nm(
    filename,
    plot=True,
    retina_index=0,
    performance="validation",
    directory_prefix="basic",
    data_type="salamander",
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    seed=None,
    config_dict=None,
    fixation_fie=None,
):
    directory = f"{home}/models/{directory_prefix}_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    dataloaders, model, config = get_wn_model_and_dataloader_for_nm(
        directory,
        filename,
        model_fn=model_fn,
        device=device,
        data_dir=data_dir,
        test=performance == "test",
        seed=seed,
        data_type=data_type,
        config_dict=config_dict,
        fixation_file=fixation_fie,
    )

    correlations, all_predictions, all_responses = get_performance_for_single_cell(
        model=model,
        dataloaders=dataloaders,
        performance=performance,
        device=device,
        retina_index=retina_index,
        rf_size=(model.config_dict["img_h"], model.config_dict["img_w"]),
        img_h=model.config_dict["img_h"],
        img_w=model.config_dict["img_w"],
    )

    cell_names = get_cell_names(
        retina_index=retina_index,
        explained_variance_threshold=model.config_dict[
            "explainable_variance_threshold"
        ],
        correlation_threshold=model.config_dict["oracle_correlation_threshold"],
        config=model.config_dict["config"],
    )
    cell_dict = {k: v for k, v in zip(cell_names, correlations)}
    np.save(f"{directory}/{filename}/stats/seed_{seed}/cell_ids_on_nm.npy", cell_names)
    np.save(
        f"{directory}/{filename}/stats/seed_{seed}/test_correlations_on_nm.npy",
        correlations,
    )
    print(
        f"saved_correlations to {directory}/{filename}/stats/seed_{seed}/test_correlations_on_nm.npy"
    )

    return (
        cell_dict,
        model.config_dict["oracle_correlation_threshold"],
        model.config_dict["explainable_variance_threshold"],
    )


def get_basic_cnn_cell_performance_dict(
    filename,
    plot=True,
    retina_index=0,
    performance="validation",
    directory_prefix="basic",
    data_type="salamander",
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    seed=None,
    nm=False,
    dataloader_config=None,
    stimulus_seed=None,
    fixation_file=None,
):
    directory = f"{home}/models/{directory_prefix}_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    if not nm:
        dataloaders, model, config = get_model_and_dataloader(
            directory,
            filename,
            model_fn=model_fn,
            device=device,
            data_dir=data_dir,
            test=performance == "test",
            seed=seed,
            data_type=data_type,
        )
        if model is None:
            return
    else:
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
            stimulus_seed=stimulus_seed,
            fixation_file=fixation_file,
        )

    correlations, all_predictions, all_responses = get_performance_for_single_cell(
        model=model,
        dataloaders=dataloaders,
        performance=performance,
        device=device,
        retina_index=retina_index,
        rf_size=(model.config_dict["img_h"], model.config_dict["img_w"]),
        img_h=model.config_dict["img_h"],
        img_w=model.config_dict["img_w"],
    )
    if plot:
        plt.plot(
            np.arange(
                (len(all_responses) - len(all_predictions)),
                len(all_responses),
                all_predictions,
            )
        )
        plt.plot(np.arange(len(all_responses[:, 11])), all_responses[:, 11])
        plt.show()

    cell_names = get_cell_names(
        retina_index=retina_index,
        explained_variance_threshold=model.config_dict[
            "explainable_variance_threshold"
        ],
        correlation_threshold=model.config_dict["oracle_correlation_threshold"],
        config=model.config_dict["config"],
    )
    cell_dict = {k: v for k, v in zip(cell_names, correlations)}
    np.save(f"{directory}/{filename}/stats/seed_{seed}/cell_ids.npy", cell_names)
    np.save(
        f"{directory}/{filename}/stats/seed_{seed}/{performance}_correlations.npy", correlations
    )
    print(
        f"saved_correlations to {directory}/{filename}/stats/seed_{seed}/{performance}_correlations.npy"
    )
    print(f"mean performance: {np.mean(list(cell_dict.values()))}")
    return (
        cell_dict,
        model.config_dict["oracle_correlation_threshold"],
        model.config_dict["explainable_variance_threshold"],
    )


def check_cell_performance_logs(retina_index, config):
    if config is None:
        config = global_config
    for cells in range(get_cell_numbers(retina_index, config)):
        if os.path.isfile(
            f'{home}/models/ln_models/{config["data_type"]}/retina{retina_index + 1}/cell_{cells}/lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0/stats/correlations.npy'
        ):
            corrs = np.load(
                f"{home}/models/ln_models/retina{retina_index + 1}/cell_{cells}/lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0/stats/correlations.npy"
            )

            print(
                cells,
                "num_of_epochs",
                len(corrs),
                "best_epoch",
                np.argmax(corrs),
                "best corr",
                np.max(corrs),
            )
        else:
            print(cells, "missing")


def create_scatter_plot_from_dicts(
    perf_dict_1,
    perf_dict_2,
    model_1_name,
    model_2_name,
    retina_index,
    correlation_threshold=0,
    explainable_variance_threshold=0,
    save_file=f"{home}/evaluations/visualizations/ln_vs_best_basic_cnn.png",
):
    fig, ax = plt.subplots()
    cell_names_list = get_cell_names(
        retina_index=retina_index,
        explained_variance_threshold=explainable_variance_threshold,
        correlation_threshold=correlation_threshold,
    )
    avg_model_1 = []
    avg_model_2 = []
    for cell in range(len(cell_names_list)):
        if (cell_names_list[cell] in perf_dict_1.keys()) and (
            cell_names_list[cell] in perf_dict_2.keys()
        ):
            sns.scatterplot(
                x=[perf_dict_1[cell_names_list[cell]]],
                y=[perf_dict_2[cell_names_list[cell]]],
                color="deepskyblue",
            )
            plt.annotate(
                cell_names_list[cell],
                xy=(
                    perf_dict_1[cell_names_list[cell]],
                    perf_dict_2[cell_names_list[cell]],
                ),
            )
            avg_model_1.append(perf_dict_1[cell_names_list[cell]])
            avg_model_2.append(perf_dict_2[cell_names_list[cell]])
    plt.plot(
        np.arange(0, np.max([v for v in perf_dict_2.values()]) + 0.1, 0.1),
        np.arange(0, np.max([v for v in perf_dict_2.values()]) + 0.1, 0.1),
        linestyle="--",
        color="black",
    )
    plt.xlabel(
        f"{model_1_name} CC; avg CC {np.round(np.mean(avg_model_1), 2)}", fontsize="15"
    )
    plt.ylabel(
        f"{model_2_name} CC; avg CC {np.round(np.mean(avg_model_2), 2)}", fontsize="15"
    )
    plt.tick_params(labelsize="15")
    #     plt.title(f'Retina {retina_index + 1}', fontsize='xx-large')
    if save_file is not None:
        plt.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.show()


def visualize_cell_kernels(directory, model_file, cell_names, seed=None):
    scale = 0.1
    if seed is None:
        seed = model_seed
    model = torch.load(
        f"{directory}/{model_file}/weights/seed_{seed}/best_model.m",
        map_location=torch.device("cpu"),
    )
    feature_vector = model["model"]["readout._features"].squeeze()
    cropped_stas, _ = get_cropped_stas(
        retina_index=0, num_of_images=20, rf_size=(21, 21)
    )
    weights = model["model"]["core.features.layer0.conv.weight"]
    scale = weights.max()
    Path(f"{directory}/{model_file}/visualizations/reconstructed_filters/").mkdir(
        parents=True, exist_ok=True
    )
    all_filters = np.zeros(((feature_vector.shape[1],) + cropped_stas[0].shape))
    for cell_index in range(feature_vector.shape[1]):
        sta = cropped_stas[cell_names[cell_index]]
        features = feature_vector[:, cell_index]
        flat_weight = weights.reshape([weights.shape[0], -1])
        cell_filters = features[:].reshape([1, -1]) @ flat_weight
        cell_filters = cell_filters.reshape(weights.shape[2:])
        fig, ax = plt.subplots(
            cell_filters.shape[0], 2, sharex=True, sharey=True, figsize=(10, 35)
        )
        for time in range(weights.shape[2]):
            im = ax[time, 0].imshow(
                cell_filters[time].cpu(), cmap="coolwarm", vmin=-scale, vmax=scale
            )

            im2 = ax[time, 1].imshow(
                sta[time], cmap="coolwarm", vmin=-0.1 / 8, vmax=0.1 / 8
            )
        ax[0, 0].set_title("Reconstructed", size="20")
        ax[0, 1].set_title("STA", size="20")

        plt.suptitle(f"Cell {cell_names[cell_index]}", size="25")
        plt.tight_layout()
        plt.savefig(
            f"{directory}/{model_file}/visualizations/reconstructed_filters/cell_{cell_names[cell_index]}",
            bbox_inches="tight",
        )
        plt.show()
        all_filters[cell_index] = cell_filters.cpu()
    plot_all_stas(
        retina_index_str="01",
        saving_file_suffix="reconstructed_filters-less_saturated",
        cell_rfs=all_filters,
        cells=62,
        cell_names=cell_names,
        vmin=-scale,
        vmax=scale,
    )


def plot_multiple_seed_dicts(dicts, save_file):
    for d in dicts:
        plt.scatter(np.arange(0, len(d.keys())), list(d.values()))
    plt.savefig(save_file, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    seeds = [128, 1024, 16, 64, 256, 8]
    # 1 layer no nonlin lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_20.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90
    # 1 layer yes nonlin lr_0.0073_l_1_ch_[64]_t_25_bs_32_tr_250_ik_25x(33, 33)x(33, 33)_g_20.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90
    for seed in seeds:
        get_multiretinal_cnn_perofrmance_dict(filename='lr_0.0073_l_1_ch_[64]_t_25_bs_32_tr_250_ik_25x(33, 33)x(33, 33)_g_20.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90',
                                          performance='test',
                                          directory_prefix='multiretinal_factorized_ev_0.15',
                                          device='cuda', seed=seed)
    # lr_0.0100_l_1_ch_16_t_15_bs_10_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1 1 layer salamander
    # 'lr_0.0100_l_1_ch_16_t_15_bs_10_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1' 1 layer salamander
    # 'lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x11x11_hk_25x7x7_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1' 3 layer salamander
    # lr_0.0010_l_4_ch_16_t_20_bs_64_tr_250_ik_20x11x11_hk_20x7x7_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1 4 layer salamander
    # lr_0.0020_l_1_ch_16_t_25_bs_128_tr_11_ik_25x(20, 20)x(20, 20)_g_47.0000_gt_1.3300_l1_1.2000_l2_0.0000_sg_0.35_p_0_bn_1_s_1norm_0_fn_1 1 layer marmoset
    # lr_0.0020_l_3_ch_16_t_25_bs_128_tr_11_ik_25x(20, 20)x(20, 20)_hk_25x(7, 7)x(7, 7)_g_47.0000_gt_1.3300_l1_1.2000_l2_0.0000_sg_0.35_p_0_bn_1_s_1norm_0_fn_1 3 layer marmoset
    # lr_0.0020_l_1_ch_16_t_25_bs_16_tr_10_ik_25x(20, 20)x(20, 20)_g_47.0000_gt_0.0030_l1_1.2000_l2_0.0000_sg_0.2_p_0_bn_1_s_1norm_0_fn_1 1 layer marmoset nm
    exit()
    retina_index = 0
    # filenames = [
    #     'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(25, 25)x(25, 25)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90',
    #     'lr_0.0073_l_1_ch_[16]_t_25_bs_16_tr_250_ik_25x(33, 33)x(33, 33)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90',
    #     'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90',
    #     'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(25, 25)x(25, 25)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90',
    #     'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_1_h_80_w_90']
    filenames = ['lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(25, 25)x(25, 25)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90',
                'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90',
                'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90',
                'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90',
                'lr_0.0073_l_1_ch_[16]_t_25_bs_32_tr_250_ik_25x(17, 17)x(17, 17)_g_47.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1___p_0_bn_1_s_1norm_0_fn_0_h_80_w_90']
    perf_dict_1 = get_basic_cnn_cell_performance_across_seeds(
        [128, 1024, 16, 64, 256, 8],
        # [256, 8],
        filename=filenames[retina_index],
        performance="test",
        retina_index=retina_index,
        directory_prefix="factorized_ev_0.15",
        model_fn="models.cnn.FactorizedEncoder.build_trained",
        device="cuda",
    )
    print(perf_dict_1)

    for retina_index in range(5):
        cell_names = get_cell_names(retina_index, explained_variance_threshold=0.0)
        # visualize_cell_kernels(f'//Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling/models/basic_ev_0.15_cnn/retina{retina_index+1}/cell_None/readout_isotropic/gmp_0',
        #                        model_file='lr_0.0130_l_1_ch_16_t_20_bs_12_tr_150_ik_20x21x21_g_2.3966_gt_3.8047_l1_0.0419_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_0',
        #                        cell_names=cell_names)
        # with open(f'/usr/users/vystrcilova/retinal_circuit_modeling/models/factorized_4_cnn/marmoset/retina1/cell_None/readout_isotropic/gmp_0/lr_0.0090_l_1_ch_16_t_15_bs_16_tr_9_ik_15x(20, 20)x(20, 20)_g_1.3300_gt_0.0030_l1_0.5000_l2_0.0000_sg_0.2_p_0_bn_1_s_1norm_0_fn_1/config/config.yaml', 'r') as config_file:
        #     config = yaml.unsafe_load(config_file)
        #     dataloader_config = config['dataloader_config']
        # with open(
        #     f"{home}/data/marmoset_data/responses/config_s4_2022.yaml", "rb"
        # ) as config_file:
        #     config_dict = yaml.unsafe_load(config_file)

        # perf_dict_2 = get_basic_wm_cnn_cell_performance_dict_for_nm(
        #     directory_prefix="factorized_ev_0.15",
        #     filename="lr_0.0020_l_1_ch_16_t_25_bs_128_tr_11_ik_25x(20, 20)x(20, 20)_g_47.0000_gt_1.3300_l1_1.2000_l2_0.0000_sg_0.35_p_0_bn_1_s_1norm_0_fn_1",
        #     plot=False,
        #     performance="test",
        #     retina_index=retina_index,
        #     model_fn="models.cnn.FactorizedEncoder.build_trained",
        #     seed=8,
        #     data_type="marmoset",
        #     config_dict=config_dict,
        # )
        #

        """ exit()
        cell_stats, cell_means = get_ln_model_performance_on_test(
            "lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0",  #'lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0', lr_0.001_whole_rf_60_ch_25_l1_1.0_l2_0.1_g_1.0_bs_128_tr_250_s_1_crop_0_norm_1
            retina_index=retina_index,
            explainable_variance_threshold=0.15,
            correlation_threshold=0,
            exclude_cells=True,
            device="cuda",
        )
        print(cell_stats)
        print(cell_means)"""
        perf_dict_1 = get_basic_cnn_cell_performance_across_seeds(
            [128, 1024, 16, 64, 256, 8],
            # [256, 8],
            filename=filenames[retina_index],
            performance="test",
            retina_index=retina_index,
            directory_prefix="factorized_ev_0.15",
            model_fn="models.cnn.FactorizedEncoder.build_trained",
            device="cuda",
        )
        print(perf_dict_1)
    exit()
    # get_ln_model_pca_results('lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0',
    #                          directory='/Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling/models/ln_models/retina1/cell_4/',
    #                          retina_index=0,
    #                          explainable_variance_threshold=0)
    # check_cell_performance_logs(retina_index)
    # exit()
    (
        perf_dict_1,
        correlation_threshold,
        explainable_variance_threshold,
    ) = get_basic_cnn_cell_performance_dict(
        filename="lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-2-2_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_90_w_100",
        plot=False,
        performance="validation",
        retina_index=retina_index,
        directory_prefix="factorized_ev_0.15",
        model_fn="models.cnn.FactorizedEncoder.build_trained",
        device="cuda",
        seed=64,


    )
    #

    # perf_dict_2, correlation_threshold, explainable_variance_threshold = get_basic_cnn_cell_performance_dict(
    #     filename='lr_0.0094_l_3_ch_16_t_25_bs_15_tr_150_ik_25x11x11_hk_25x7x7_g_47.0000_gt_1.1453_l1_1.2521_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1',
    #     plot=True, performance='validation', retina_index=retina_index, directory_prefix='factorized_ev_0.15',
    #     model_fn='models.cnn.FactorizedEncoder.build_trained', device='cuda')
    # perf_dict_2 = get_basic_cnn_cell_performance_dict(filename='lr_0.001_l_1_ch_8_t_15_g_0.8_bs_128_tr_250_s_1_cr_(50, 50, 75, 65)_ik_11_hk_5_l1_1.0_mu_0.7_sg_0.15_p_0', plot=True, performance='train', retina_index=retina_index)

    # print(perf_dict_1)
    # perf_dict_2, _ = get_ln_model_performance_on_test('lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0',
    #                                                retina_index=retina_index,
    #                                                explainable_variance_threshold=explainable_variance_threshold,
    #                                                correlation_threshold=correlation_threshold, exclude_cells=True,
    #                                                 start_index=72)
    exit()
    perf_dict_2 = get_ln_model_cell_performance_dict(
        "lr_0.001_whole_rf_60_ch_25_l1_1.2_l2_0.1_g_0.5_bs_128_tr_250_s_1_crop_0_norm_1",
        retina_index=retina_index,
        explainable_variance_threshold=0.15,
        correlation_threshold=None,
        exclude_cells=True,
    )
    ln_correlations = list(perf_dict_2.values())
    # cnn_correlations = list(perf_dict_1.values())
    plt.hist(ln_correlations, bins=20, range=(0, 1), histtype="step")
    plt.hist(cnn_correlations, bins=20, range=(0, 1), histtype="step")
    plt.show()
    avg_correlation_ln_model = np.mean([v for v in perf_dict_2.values()])
    avg_correlation_cnn = np.mean([v for v in perf_dict_1.values()])
    print("ln_model_avg:", avg_correlation_ln_model)
    print("cnn avg:", avg_correlation_cnn)
    create_scatter_plot_from_dicts(
        perf_dict_1,
        perf_dict_2,
        "CNN test",
        "LN test",
        retina_index,
        explainable_variance_threshold=explainable_variance_threshold
        if explainable_variance_threshold is not None
        else 0,
        correlation_threshold=correlation_threshold
        if correlation_threshold is not None
        else 0,
    )

    # python3 run_factorized_cnn.py  --retina_index 0  --num_of_trials 11 --epochs 1000 --data_path /scratch/usr/nibmvyst/ --batch_size=16  --num_of_frames 25 --wandb 0 --layers 3 --gamma 47 --l1 1.2 --lr 0.002 --spatial_input_kernel_size 11
