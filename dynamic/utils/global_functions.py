import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

cuda = torch.cuda.is_available()
dataset_seed = 18
model_seed = 8
mei_seed = 28

home = Path(__file__).absolute().parent.parent
with open(f"{home}/data/salamander_data/responses/config.yaml", "rb") as config_file:
    global_config = yaml.unsafe_load(config_file)
# big_crops = {#'01': (50, 50, 75, 65)
#              '01': (40, 40, 65, 55),
#              '02': (50, 50, 70, 70),
#              # '04': (60, 40, 70, 70),
#              '04': (50, 30, 60, 60),
#              #'03': (50, 50, 60, 80),
#              '03': (40, 40, 50, 70),
#              '05': (40, 60, 70, 70)}
#
# crops = {'01': (20, 40, 50, 50),
#          '02': (40, 35, 65, 65),
#          '05': (35, 50, 65, 65)}
#
# cell_numbers = {'01': 78,
#                 '02': 60,
#                 '03': 39,
#                 '04': 45,
#                 '05': 58}
#
# exclude_cells = {'01': [5, 8, 24, 26, 37, 56, 57, 58, 59, 61, 62],
#                  '02': [3, 19, 39],
#                  '03': [3, 1, 11, 18, 23, 28, 31],
#                  '04': [],
#                  '05': [2]}
#
# average_oracle_correlation = {'01': 0.6465418,
#                               '02': float('nan'),
#                               '03': float('nan'),
#                               '04': 0.38977978,
#                               '05': float('nan'), }
#
# explainable_variance = {'01': 0.43,
#                         '02': float('nan'),
#                         '03': float('nan'),
#                         '04': 0.2036676,
#                         '05': float('nan')}
#
# cell_oracle_correlations = {
#     '01': [0.8268581, 0.36021528, 0.35586053, 0.73004735, 0.73335195, 0.7085799,
#            0.6904433, 0.5711028, 0.562644, 0.74564266, 0.85244346, 0.6833525,
#            0.7082064, 0.44161227, 0.64343905, 0.46024236, 0.26771104, 0.5506245,
#            0.755251, 0.75516325, 0.6760011, 0.7209638, 0.4580723, 0.41117093,
#            0.75559247, 0.56964266, 0.7589324, 0.80191094, 0.77101, 0.79176503,
#            0.7446992, 0.72869694, 0.7842388, 0.7175237, 0.7451092, 0.5305798,
#            0.34554762, 0.74851835, 0.60060954, 0.37297744, 0.8152386, 0.64005136,
#            0.71788895, 0.710045, 0.53110725, 0.702015, 0.5977911, 0.7817108,
#            0.56831706, 0.75285345, 0.71268564, 0.36178556, 0.46432495, 0.7947421,
#            0.7282064, 0.5039327, 0.76973945, 0.79846513, 0.7466562, 0.6229012,
#            0.6885522, 0.64947444, 0.647055, 0.85014594, 0.41293722, 0.7279322,
#            0.8006044, 0.49066082, 0.73248744, 0.8459257, 0.623574, 0.76262355,
#            0.4544705, 0.4916767, 0.7352828, 0.5877189, 0.5385386, 0.6337836, ],
#     '02': None,
#     '03': None,
#     '04': [3.1836751e-01, 4.2340425e-01, 4.1641393e-01, 3.0366927e-03,
#            3.3957568e-01, 3.2244363e-01, 2.3166926e-01, 6.8096083e-01,
#            1.9954826e-01, 4.7723624e-01, 6.6223818e-01, 5.4484040e-01,
#            5.1294696e-01, 5.9680444e-01, 5.9403175e-01, 4.5221710e-01,
#            3.7790790e-01, 3.5176957e-01, 5.6190932e-01, -3.3755906e-04,
#            5.0994951e-01, 4.4420955e-01, 4.1492608e-01, 2.5848833e-01,
#            3.8724306e-01, 5.1291758e-01, 6.2596697e-01, 2.4538453e-01,
#            5.2674752e-01, 3.5987711e-01, 2.1304947e-01, 3.5664430e-01,
#            3.0604351e-01, 7.0098168e-01, 7.9631425e-02, 3.6381027e-01,
#            4.5746455e-01, 5.8541429e-01, 2.1367650e-01, 3.0915162e-01,
#            3.4938815e-01, 4.9059519e-01, 1.0995749e-02, 3.9510617e-01,
#            3.5544458e-01],
#     '05': None}
#
# cell_explainable_variance = {
#     '01': [0.68209563, 0.12984769, 0.16809089, 0.51903341, 0.53525742, 0.49899674,
#            0.47791554, 0.27336239, 0.31897035, 0.55265945, 0.72098464, 0.45742531,
#            0.49789193, 0.20184636, 0.41352869, 0.21520548, 0.07124622, 0.29360015,
#            0.56780419, 0.55464122, 0.45474972, 0.51778592, 0.2131389, 0.16166001,
#            0.57087676, 0.33168974, 0.57862813, 0.64152003, 0.59182242, 0.62615412,
#            0.54095571, 0.52848297, 0.57873155, 0.50991558, 0.55294264, 0.28146467,
#            0.1293177, 0.55590285, 0.38732149, 0.1499299, 0.66110406, 0.40529187,
#            0.51451485, 0.50341264, 0.25923459, 0.4914635, 0.3569573, 0.61039545,
#            0.32608601, 0.56562359, 0.49394717, 0.13491025, 0.2185508, 0.62910433,
#            0.51613554, 0.25676767, 0.58773303, 0.63230189, 0.55187731, 0.38407795,
#            0.47030632, 0.41616832, 0.41617062, 0.72006875, 0.17166408, 0.52456513,
#            0.63877514, 0.27724551, 0.5332427, 0.71267333, 0.36741564, 0.57314221,
#            0.20449585, 0.2461801, 0.53819728, 0.33384105, 0.33995077, 0.40177686],
#     '02': None,
#     '03': None,
#     '04': [1.40062951e-01, 1.86956199e-01, 1.75430680e-01, 7.58176608e-04,
#            1.94844681e-01, 1.16392127e-01, 6.40509468e-02, 4.55103870e-01,
#            4.36813111e-02, 2.43602374e-01, 4.49483304e-01, 3.06631264e-01,
#            2.65162644e-01, 4.06519438e-01, 3.53546985e-01, 1.98238822e-01,
#            1.54488337e-01, 1.44034351e-01, 3.26214988e-01, -1.38910594e-04,
#            3.61415337e-01, 2.01969518e-01, 2.38619914e-01, 8.39490082e-02,
#            1.54037983e-01, 3.51437586e-01, 3.97761293e-01, 1.14424152e-01,
#            2.76069556e-01, 2.67963894e-01, 8.81427589e-02, 1.86689984e-01,
#            1.25914146e-01, 4.84424550e-01, 1.30066401e-02, 1.57263858e-01,
#            2.12076867e-01, 3.45538884e-01, 7.81074915e-02, 1.02022654e-01,
#            1.31143333e-01, 2.40547642e-01, 2.00947538e-03, 1.62351217e-01,
#            1.63089948e-01],
#     '05': None}


def get_exclude_cells_based_on_explainable_variance_threshold(
    retina_index, config=global_config, threshold=None
):
    if threshold is None or (threshold <= 0):
        threshold = -10
    reliabilities = config["cell_explainable_variance"][f"0{retina_index+1}"]
    # already_excluded_cells = config['exclude_cells'][f'0{retina_index+1}']
    newly_excluded_cells = [
        x for x in range(len(reliabilities)) if reliabilities[x] < threshold
    ]
    return list(set(newly_excluded_cells))


def get_exclude_cells_based_on_correlation_threshold(
    retina_index, config=global_config, threshold=None
):
    if (threshold is None) or (threshold <= 0):
        threshold = -10
    reliabilities = config["cell_oracle_correlations"][f"0{str(retina_index + 1)}"]
    # already_excluded_cells = config['exclude_cells'][f'0{str(retina_index + 1)}']
    newly_excluded_cells = [
        x for x in range(len(reliabilities)) if reliabilities[x] < threshold
    ]
    return list(set(newly_excluded_cells))


def get_exclude_cells_based_on_thresholds(
    retina_index,
    config=global_config,
    explainable_variance_threshold=None,
    correlation_threshold=None,
):
    ev_excluded = get_exclude_cells_based_on_explainable_variance_threshold(
        retina_index, config, explainable_variance_threshold
    )
    oc_excluded = get_exclude_cells_based_on_correlation_threshold(
        retina_index, config, correlation_threshold
    )
    return list(set(ev_excluded + oc_excluded))


def get_cell_numbers_after_crop(
    retina_index,
    config=global_config,
    correlation_threshold=None,
    explained_variance_threshold=None,
):
    already_excluded_cells = config["exclude_cells"][f"0{retina_index + 1}"]
    excluded_from_corr = get_exclude_cells_based_on_correlation_threshold(
        retina_index, config, correlation_threshold
    )
    excluded_from_ev = get_exclude_cells_based_on_explainable_variance_threshold(
        retina_index, config, explained_variance_threshold
    )
    excluded_cells = list(
        set(excluded_from_ev + excluded_from_corr + already_excluded_cells)
    )
    return config["cell_numbers"][f"0{retina_index+1}"] - len(excluded_cells)


def get_cell_names(
    retina_index,
    config=global_config,
    correlation_threshold=None,
    explained_variance_threshold=None,
):
    if 'exclude_cells' in config.keys():
        already_excluded_cells = config["exclude_cells"][f"0{retina_index + 1}"]
    else:
        already_exclude_cells = []
    excluded_from_corr = get_exclude_cells_based_on_correlation_threshold(
        retina_index, config, correlation_threshold
    )
    excluded_from_ev = get_exclude_cells_based_on_explainable_variance_threshold(
        retina_index, config, explained_variance_threshold
    )
    cell_names = [
        x
        for x in range(config["cell_numbers"][f"0{retina_index+1}"])
        if x not in excluded_from_corr
        and x not in excluded_from_ev
        and x not in already_excluded_cells
    ]
    return cell_names


def get_cell_numbers(retina_index, config=global_config):
    return config["cell_numbers"][f"0{retina_index+1}"]


def get_exclude_cells(retina_index, config=global_config):
    exclude_cells = config["exclude_cells"]
    return exclude_cells[f"0{retina_index+1}"]


def get_all_cell_numbers_after_crop(config=global_config):
    cell_numbers = config["cell_numbers"]
    exclude_cells = config["exclude_cells"]
    cell_numbers_after_crop = {
        "01": cell_numbers["01"] - len(exclude_cells["01"]),
        "02": cell_numbers["02"] - len(exclude_cells["02"]),
        "03": cell_numbers["03"] - len(exclude_cells["03"]),
        "04": cell_numbers["04"] - len(exclude_cells["04"]),
        "05": cell_numbers["05"] - len(exclude_cells["05"]),
    }
    return cell_numbers_after_crop


def set_random_seed(seed, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    # if deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
