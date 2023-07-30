from utils.global_functions import home
import yaml
import matplotlib.pyplot as plt
from nnfabrik import builder
import torch
import numpy as np
import os

from dynamic.training.regularizers import TimeLaplaceL23d

def get_model_and_dataloader_for_nm_new(
    directory,
    filename,
    model_fn="models.BasicEncoder.build_trained",
    device="cpu",
    data_dir=None,
    test=False,
    seed=None,
    config_dict=None,
    data_type="salamander",
    dataloader_config=None,
    stimulus_seed=None,
    fixation_file=None,
    num_of_trials_to_use=None,
    stimulus_dir=None
):
    with open(f"/{directory}/{filename}/config/config.yaml", "r") as config_file:
        config = yaml.unsafe_load(config_file)
    model_config = config["model_config"]
    model_config["readout_type"] = "isotropic"

    if "config" not in model_config.keys():
        model_config["config"] = config_dict
    else:
        config_dict = model_config["config"]
    #dataset_fn = "datasets.frame_movie_loader"
    #config["base_path"] = home_dir

    model_config["base_path"] = f"{home}"
    # if test:
    # dataloader_config['time_chunk_size'] = 1

    # dataloader_config['cell_index'] = cell_index

    # dataloader_config
    # dataloader_config['path'] = 'vystrcilova/retinal_circuit_modeling/data'c

    # model_fn = 'models.multi_retina_regular_cnn_model'
    dataloaders_fake = dict()
    if "readout" in model_config.keys():
        del model_config["readout"]
    if data_dir is None:
        model_config["data_dir"] = home
    else:
        model_config["data_dir"] = data_dir
    model_config["padding"] = 0
    model_fn = eval(model_fn)
    model = builder.get_model(
        model_fn,
        model_config={
            "model_dir": directory,
            "model_name": filename,
            "data_dir": home,
            "device": device,
        },
        dataloaders=dataloaders_fake,
        seed=seed,
    )
    model_config[
        "core.temporal_regulazirer.laplace.filter"
    ] = TimeLaplaceL23d().laplace.filter
    # model.load_state_dict(model_dict)
    model = model.double()
    return model
