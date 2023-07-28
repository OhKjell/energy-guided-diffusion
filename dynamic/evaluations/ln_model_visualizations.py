from pathlib import Path

from datasets.stas import plot_all_stas, get_cropped_stas
from training.measures import correlation
from utils.global_functions import cell_numbers, cell_names, home
import torch, os
import numpy as np


def collect_best_models_for_all_cells(
    base_path, filename, retina_index, data_type="salamander"
):
    all_cell_filters = None
    correlations = []
    for cell_number in range(cell_numbers[f"0{retina_index + 1}"]):
        if os.path.isfile(
            f"{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/best_model.m"
        ):
            model = torch.load(
                f"{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/best_model.m",
                map_location=torch.device("cpu"),
            )
            corrs = np.load(
                f"{base_path}/{data_type}/retina{retina_index + 1}/cell_{cell_number}/{filename}/stats/correlations.npy"
            )
            correlations.append(np.max(corrs))
            model = model["model"]
            weights = model["conv1.weight"][0]
        else:
            correlations.append(float("nan"))
            weights = np.zeros((15, 20, 20))
        if cell_number == 0:
            all_cell_filters = np.zeros(
                (cell_numbers[f"0{retina_index + 1}"],) + weights.shape
            )
        all_cell_filters[cell_number] = weights
    return all_cell_filters, correlations


def ln_sta_correlation(retina_index, base_path, filename, plot=False):
    correlations = []
    weights = []
    stas, _ = get_cropped_stas(retina_index, num_of_images=15, rf_size=None)
    for cell_number in range(cell_numbers[f"0{retina_index + 1}"]):
        if os.path.isfile(
            f"{base_path}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/best_model.m"
        ):
            model = torch.load(
                f"{base_path}/retina{retina_index + 1}/cell_{cell_number}/{filename}/weights/best_model.m",
                map_location=torch.device("cpu"),
            )
            weight = model["model"]
            weight = weight["conv1.weight"]
            sta = stas[cell_number]
            corr = correlation(
                torch.flatten(torch.tensor(sta)), torch.flatten(weight), 1e-8
            )
            correlations.append(corr.item())
            weights.append(weight)
    print(correlations)
    print(np.mean(correlations))
    if plot:
        plot_all_stas(f"0{retina_index+1}", saving_file="")


if __name__ == "__main__":
    filename = "lr_0.001_rf_10_ch_15_l1_0.05_l2_0.1_g_0.1_bs_128_tr_250_s_1_crop_0"
    retina_index = 0
    data_type = "salamander"
    ln_sta_correlation(retina_index, f"{home}/models/ln_models/", filename)
    all_cell_filters, correlations = collect_best_models_for_all_cells(
        f"{home}/retinal_circuit_modeling/models/ln_models/",
        retina_index=retina_index,
        filename=filename,
        data_type=data_type,
    )
    Path(
        f"{home}/models/ln_models/{data_type}/retina1{retina_index + 1}visualizations/{filename}/"
    ).mkdir(exist_ok=True, parents=True)
    plot_all_stas(
        cell_rfs=all_cell_filters,
        retina_index_str=f"0{retina_index + 1}",
        saving_file=f"{home}/models/ln_models/{data_type}/retina{retina_index + 1}/visualizations/{filename}/all_ln_model_filters",
        correlations=correlations,
        vmin=-0.075,
        vmax=0.075,
        data_type=data_type,
    )
