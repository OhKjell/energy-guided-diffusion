import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.global_functions import model_seed


def make_seed_compatible(directory):
    # for cell in range(78):
    # whole_dir = f'{directory}/cell_{cell}/'
    whole_dir = directory
    model_names = os.listdir(whole_dir)
    for model_name in model_names:
        model_dirs = os.listdir(os.path.join(whole_dir, model_name))
        for md in model_dirs:
            if md != "config":
                dir_files = os.listdir(os.path.join(whole_dir, model_name, md))
                Path(
                    os.path.join(whole_dir, model_name, md, f"seed_{model_seed}")
                ).mkdir(exist_ok=True)
                for file in dir_files:
                    if "seed" not in file:
                        source = os.path.join(whole_dir, model_name, md, file)
                        dest = os.path.join(
                            whole_dir, model_name, md, f"seed_{model_seed}", file
                        )
                        os.rename(source, dest)


def move_trial_files_in_dir(directory):
    files = os.listdir(directory)
    files = [file for file in files if file.startswith("trial")]

    for file in tqdm(files):
        if ".npy" in file:
            if os.path.exists(f'{directory}/{file.split(".")[0]}/all_images.npy'):
                continue
            else:
                Path(f'{directory}/{file.split(".")[0]}/').mkdir(exist_ok=True)
                array = np.load(f"{directory}/{file}")
                array = np.moveaxis(array, 0, 2)
                Path(
                    f'{directory}/{file.split(".")[0][:-3]}_{file.split(".")[0][-3:]}'
                ).mkdir(exist_ok=True)
                np.save(
                    f'{directory}/{file.split(".")[0][:-3]}_{file.split(".")[0][-3:]}/all_images.npy',
                    array,
                )
        else:
            new_file = f"{directory}/{file}/all_images.npy"
            if os.path.exists(new_file):
                array = np.load(new_file)
                array = np.moveaxis(array, 0, 2)
                Path(f"{directory}/{file[:-3]}_{file[-3:]}").mkdir(exist_ok=True)
                np.save(f"{directory}/{file[:-3]}_{file[-3:]}/all_images.npy", array)


if __name__ == "__main__":
    dirs = [
        # '/usr/users/vystrcilova/retinal_circuit_modeling/models/ln_models/salamander/retina1',
        "/usr/users/vystrcilova/retinal_circuit_modeling/models/basic_ev_0.15_cnn/salamader/retina1/cell_None/readout_isotropic/gmp_0"
        # '/usr/users/vystrcilova/retinal_circuit_modeling/models/factorized_ev_0.15_cnn/retina1',
        # '/usr/users/vystrcilova/retinal_circuit_modeling/models/factorized_ev_0.15_cnn/retina4',
    ]
    for directory in dirs:
        make_seed_compatible(directory)
    # move_trial_files_in_dir('/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/non_repeating_stimuli/')
