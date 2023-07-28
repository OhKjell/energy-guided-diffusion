import os.path
import numpy as np
from pathlib import Path


from datasets.stas import get_stas_from_pcs
from models.ln_model import Model, FactorizedModel
from models.helper_functions import plot_responses_vs_predictions
from training.measures import variance_of_predictions, correlation
from training.trainers import model_step
from utils.global_functions import (
    dataset_seed,
    get_cell_numbers_after_crop,
    get_cell_names,
    home,
    model_seed,
    global_config,
)
import random
from nnfabrik import builder
import torch
import yaml
from tqdm import tqdm


home_dir = "/usr/users/vystrcilova/retinal_circuit_modeling"


def get_ln_model_cell_performance_dict(
    file_name,
    retina_index,
    metric="correlations",
    exclude_cells=False,
    base_path=f"{home}/models/ln_models/",
    correlation_threshold=None,
    explainable_variance_threshold=None,
    config=global_config,
):
    cell_stats = {}
    retina_index_str = f"0{retina_index+1}"
    num_of_cells = (
        get_cell_numbers_after_crop(
            retina_index,
            correlation_threshold=correlation_threshold
            if correlation_threshold is not None
            else 0,
            explained_variance_threshold=explainable_variance_threshold
            if explainable_variance_threshold is not None
            else 0,
        )
        if exclude_cells
        else global_config["cell_numbers"][retina_index_str]
    )
    cell_names_list = (
        get_cell_names(
            retina_index,
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
    for cell in range(num_of_cells):
        if os.path.isfile(
            f"{base_path}/salamander/retina{retina_index+1}/cell_{cell_names_list[cell]}/{file_name}/stats/{metric}.npy"
        ):
            data = np.load(
                f"{base_path}/salamander/retina{retina_index+1}/cell_{cell_names_list[cell]}/{file_name}/stats/{metric}.npy"
            )
            cell_stats[cell_names_list[cell]] = np.max(data)
    return cell_stats


def get_model_and_dataloader_for_ln(
    config,
    filename,
    cell_number,
    retina_index,
    device="cpu",
    seed=None,
    data_type="salamander",
    base_path=f"{home}/models/ln_models/",
    factorized=False,
):
    if seed is None:
        seed = model_seed
    # rf_size, channels, l1, l2, g = get_parameters_from_ln_model_file_name(filename)
    if not os.path.isfile(
        f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell_number}/{filename}/weights/seed_{seed}/best_model.m"
    ):
        return None, None
    model = torch.load(
        f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell_number}/{filename}/weights/seed_{seed}/best_model.m",
        map_location=torch.device(device),
    )

    with open(
        f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell_number}/{filename}/config/config.yaml",
        "r",
    ) as config_file:
        config = yaml.unsafe_load(config_file)
    model_dict = model["model"]
    dataset_fn = "datasets.white_noise_loader"
    # config['base_path'] = home_dir
    dataloader_config = config["dataloader_config"]
    if "num_of_frames" in config["model_config"].keys() and factorized:
        config["model_config"]["num_of_channels"] = config["model_config"][
            "num_of_frames"
        ]
        del config["model_config"]["num_of_frames"]
    dataloader_config[
        "train_image_path"
    ] = f'{home_dir}/{dataloader_config["config"]["training_img_dir"]}'
    dataloader_config[
        "test_image_path"
    ] = f'{home_dir}/{dataloader_config["config"]["test_img_dir"]}'
    dataloader_config[
        "neuronal_data_dir"
    ] = f'{home_dir}/{dataloader_config["config"]["neuronal_data_path"]}'

    batch_size = 128

    dataloaders = builder.get_data(dataset_fn, dataloader_config)
    if not factorized:
        model = Model(**config["model_config"])
    else:
        model = FactorizedModel(**config["model_config"])
    model.load_state_dict(model_dict)

    return dataloaders, model


def get_model_and_dataloader_for_ln_on_nm(
    config,
    filename,
    cell_number,
    retina_index,
    device="cpu",
    seed=None,
    data_type="salamander",
    base_path=f"{home}/models/ln_models/",
    factorized=False,
):
    if seed is None:
        seed = model_seed
    # rf_size, channels, l1, l2, g = get_parameters_from_ln_model_file_name(filename)
    if not os.path.isfile(
        f"{home}/{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/seed_{seed}/best_model.m"
    ):
        return None, None
    model = torch.load(
        f"{home}/{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/seed_{seed}/best_model.m",
        map_location=torch.device(device),
    )

    with open(
        f"{home}/{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/config/config.yaml",
        "r",
    ) as config_file:
        config = yaml.unsafe_load(config_file)
    model_dict = model["model"]
    dataset_fn = "datasets.frame_movie_loader"
    # config['base_path'] = home_dir
    dataloader_config = config["dataloader_config"]
    # dataloader_config['image_path'] = f'{home_dir}/{dataloader_config["config"]["image_path"]}'
    # dataloader_config['fixation_file'] = f'{home_dir}/{dataloader_config["fixation_file"][f"0{retina_index+1}"]}'
    # dataloader_config['response_path'] = f'{home_dir}/{dataloader_config["config"]["response_path"]}'

    batch_size = 128

    dataloaders = builder.get_data(dataset_fn, dataloader_config)
    if not factorized:
        model = Model(**config["model_config"])
    else:
        model = FactorizedModel(**config["model_config"])
    model.load_state_dict(model_dict)

    return dataloaders, model


def get_parameters_from_ln_model_file_name(filename):
    parameters = filename.split("_")
    rf_size = int(int(parameters[4]))
    channels = int(parameters[6])
    l1 = float(parameters[8])
    l2 = float(parameters[10])
    g = float(parameters[12])
    fancy_nonlin = float(parameters[-3])
    return rf_size, channels, l1, l2, g, fancy_nonlin


def get_ln_model_performance_on_test(
    file_name,
    retina_index,
    exclude_cells=False,
    correlation_threshold=None,
    explainable_variance_threshold=0,
    device="cuda",
    rf_size=60,
    img_h=150,
    img_w=200,
    save_file=None,
    start_index=0,
    data_type="salamander",
    base_path=f"{home}/models/ln_models/",
    factorized=False,
    config=global_config,
    nm=False,
):
    cell_stats = {}
    cell_means = []
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
    # cell_index = cell_names_list.index(138)
    # cell_names_list.remove(58)
    for i, cell in enumerate(cell_names_list):
        if os.path.isfile(
            f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/correlations_test.npy"
        ):
            correlations = np.load(
                f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/correlations_test.npy"
            )
            cell_stats[cell] = correlations[0]
            cell_means.append(correlations[0])
            continue
        elif os.path.isfile(
            f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/test_correlation.npy"
        ):
            correlation = np.load(
                f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/test_correlation.npy"
            )
            cell_stats[cell] = correlations[0]
            cell_means.append(correlations[0])
            continue
        if not nm:
            dataloaders, model = get_model_and_dataloader_for_ln(
                config=None,
                retina_index=retina_index,
                cell_number=cell,
                filename=file_name,
                device=device,
                base_path=base_path,
                factorized=factorized,
                data_type=data_type,
            )
        else:
            dataloaders, model = get_model_and_dataloader_for_ln_on_nm(
                config=None,
                retina_index=retina_index,
                cell_number=cell,
                filename=file_name,
                device=device,
                base_path=base_path,
                factorized=factorized,
                data_type=data_type,
            )
        if model is None:
            continue
        model.double()
        model.to(device)
        model.eval()
        first_session_ID = list((dataloaders["train"].keys()))[0]
        print(first_session_ID)
        a_dataloader = dataloaders["train"][first_session_ID]
        img_h = int(
            (img_h - a_dataloader.dataset.crop[0] - a_dataloader.dataset.crop[1]) / 1
        )
        img_w = int(
            (img_w - a_dataloader.dataset.crop[2] - a_dataloader.dataset.crop[3]) / 1
        )
        if rf_size is not None:
            half_rf_size = int(rf_size / 2)
            receptive_field = np.load(
                f'{home}/{config["stas_path"]}/{config["sta_file"]}_{cell}.npy'
            )
            receptive_field = dataloaders["train"][
                f"0{retina_index+1}"
            ].dataset.transform(receptive_field)
            temporal_variances = np.var(receptive_field, axis=0)
            max_coordinate = np.unravel_index(
                np.argmax(temporal_variances), (img_h, img_w)
            )
            size = (
                max_coordinate[0] + min(half_rf_size, img_h - max_coordinate[0])
            ) - (max_coordinate[0] - min(half_rf_size, max_coordinate[0])), (
                max_coordinate[1] + min(half_rf_size, img_w - max_coordinate[1])
            ) - (
                max_coordinate[1] - min(half_rf_size, max_coordinate[1])
            )
        test_correlations = []
        test_losses = []
        test_variances = []
        all_outputs = []
        all_responses = []
        for images, responses in tqdm(
            dataloaders["test"][str(retina_index + 1).zfill(2)]
        ):
            images = images.double().to(device)
            responses = responses.to(device)
            all_responses.append(responses.squeeze(-1))
            output = model_step(
                images=images,
                model=model,
                max_coordinate=max_coordinate,
                rf_size=size,
                h=img_h,
                w=img_w,
            )
            all_outputs.append(output)
            prediciton_variance = variance_of_predictions(output)
            corr = correlation(output, responses.squeeze(-1), 1e-12)
            test_variances.append(float(prediciton_variance.item()))
            test_correlations.append(corr.item())
            Path(
                f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/visualizations/cell_responses/"
            ).mkdir(exist_ok=True, parents=True)
            # plot_responses_vs_predictions([responses.squeeze(-1).detach().cpu()], [output.detach().cpu()], cell=0, cell_name=cell, save_file=f'{base_path}/retina{retina_index+1}/cell_{cell}/{file_name}/visualizations/cell_responses/cell_{cell}_test.png',
            #                              max_cc=np.mean(test_correlations))
        all_outputs = torch.cat(all_outputs)
        all_responses = torch.cat(all_responses)
        # uncomment for salamander sta based models not end-to-end learned
        # test_correlations = correlation(all_outputs[start_index:-1], all_responses[start_index+1:], 1e-12)
        test_correlations = correlation(
            all_outputs[start_index:], all_responses[start_index:], 1e-12
        )

        np.save(
            f"{home}/{base_path}/{data_type}/retina{retina_index+1}/cell_{cell}/{file_name}/stats/seed_{model_seed}/test_correlation.npy",
            test_correlations.item(),
        )
        cell_stats[cell] = test_correlations.item()
        cell_means.append(test_correlations.item())
        print(f"cell {cell}:", test_correlations.item())
    print("Test correlation: ", np.mean(cell_means))
    print(cell_stats)
    return cell_stats, cell_means


if __name__ == "__main__":
    # with open(
    #     f"{home}/data/marmoset_data/responses/config_05.yaml", "rb"
    # ) as config_file:
    #     config = yaml.unsafe_load(config_file)
    with open(f'{home}/data/salamander_data/responses/config.yaml', 'rb') as config_file:
        config = yaml.unsafe_load(config_file)
    retina_perfs = []
    cell_perfs = []
    for retina in range(5):
        print(f"Retina {retina+1}")
        _, cell_means = get_ln_model_performance_on_test(
            # "lr_0.001_whole_rf_60_t_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_250_s_1_c_0_n_0_fn_0_do_n_1",
            "lr_0.001_whole_rf_20_ch_15_l1_0.0_l2_0.0_g_0.0_bs_128_tr_250_s_1_c_0_n_0_fn_0_do_n_1",
            # lr_0.001_whole_rf_60_t_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_250_s_1_c_0_n_0_fn_0_do_n_1
            retina_index=retina,
            exclude_cells=True,
            base_path="models/ln_models/ln_models_factorized_sta/",
            rf_size=20,
            factorized=True,
            data_type="salamander",
            config=config,
            nm=False,
        )
        retina_perfs.append(np.mean(cell_means))
        cell_perfs += cell_means
    print(np.mean(retina_perfs), retina_perfs)
    print('cell mean', np.mean(cell_perfs))

    # lr_0.001_whole_rf_60_ch_25_l1_0.0_l2_0.0_g_0.0_bs_256_tr_11_s_1_c_0_n_0_fn_0_do_n_1/ marmoset factorized
    # lr_0.001_whole_rf_20_ch_25_l1_0.0_l2_0.0_g_0.0_bs_256_tr_11_s_1_c_0_n_0_fn_0_do_n_1 marmoset factorized cropped
    # lr_0.001_whole_rf_20_ch_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_250_s_1_c_0_n_0_fn_1_do_n_1 salamander cropped

    # marmoset natural movie sta predictions lr_0.001_whole_rf_20_ch_25_l1_0.0_l2_0.0_g_0.0_bs_128_tr_10_s_1_c_0_n_0_fn_0_do_n_1
