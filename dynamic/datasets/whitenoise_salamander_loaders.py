import os
from abc import ABC
import yaml
import math
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from utils.global_functions import (
    set_random_seed,
    get_exclude_cells_based_on_correlation_threshold,
    get_exclude_cells_based_on_explainable_variance_threshold,
)
import numpy as np
import pickle
from datasets.TrialWiseDataset import TrialWiseDataset


def get_trial_wise_validation_split(train_image_path, train_frac, seed=None):
    """
    Splits the trials containing images into train and validation set.
    It ensures that in every session, the same train and validation data are used.
    For each session, only the trial indices the session contains are considered.

    :param train_image_path: Path to directory containing a .npy file with images for every trial
    :param train_frac: Fraction of trials to be used as training set
    :param seed: random seed

    :return: Two arrays, containing trial IDs of all trials, split into train and validation
    """
    num_of_trials = len(
        [file for file in os.listdir(train_image_path) if "trial" in file]
    )
    if seed is not None:
        set_random_seed(seed)
    else:
        set_random_seed(1000)
    train_idx, val_idx = np.split(
        np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)]
    )

    assert not np.any(
        np.isin(train_idx, val_idx)
    ), "train_set and val_set are overlapping sets"

    print(f"train idx: {train_idx}")
    print(f"val idx: {val_idx}")
    return train_idx, val_idx


def average_repeated_stimuli_responses(repeated_responses: np.ndarray):
    # assert len(repeated_responses.shape) == 3  # Repeated stimuli should have
    # shape num_of_cells x number of repeated images x number of repetitions of repeated stimuli
    return np.mean(repeated_responses, axis=-1)


def get_dataloader(
    response_dict,
    path,
    indices,
    test,
    batch_size,
    shuffle=False,
    use_cache=True,
    cache_maxsize=20,
    movie_like=False,
    num_of_frames=None,
    device="cpu",
    crop=50,
    subsample=20,
    conv3d=False,
    time_chunk_size=1,
    overlapping=False,
    num_of_layers=None,
    temporal_dilation=1,
    hidden_temporal_dilation=1,
    num_of_hidden_frames=None,
    extra_frame=0,
):
    data = TrialWiseDataset(
        response_dict,
        path,
        "images",
        "responses",
        indices=indices,
        use_cache=use_cache,
        test=test,
        cache_maxsize=cache_maxsize,
        crop=crop,
        subsample=subsample,
        movie_like=movie_like,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=num_of_hidden_frames,
        num_of_layers=num_of_layers,
        device=device,
        conv3d=conv3d,
        time_chunk_size=time_chunk_size,
        single_prediction=overlapping,
        temporal_dilation=temporal_dilation,
        hidden_temporal_dilation=hidden_temporal_dilation,
        extra_frame=extra_frame,
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_dataloader_dims(dataloaders):
    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]
    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    # in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields
    in_name, out_name = "images", "responses"
    n_neurons_dict = {}
    in_shapes_dict = {}
    input_channels = []
    for key in dataloaders.keys():
        n_neurons_dict[key] = dataloaders[key].dataset.n_neurons
        v = dataloaders[key].dataset.__getitem__(0)
        in_shapes_dict[key] = v[0].shape
        input_channels.append(v[0].shape[0])

    # session_shape_dict = get_dims_for_loader_dict(dataloaders)
    # n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    # in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    # input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    return n_neurons_dict, in_shapes_dict, input_channels


def white_noise_loader(
    config,
    neuronal_data_dir,
    train_image_path,
    test_image_path,
    batch_size=10,
    seed=None,
    train_frac=0.8,
    subsample=1,
    crop=20,
    use_cache=True,
    num_of_trials_to_use=None,
    movie_like=False,
    num_of_frames=None,
    num_of_hidden_frames=None,
    temporal_dilation=1,
    hidden_temporal_dilation=1,
    cell_index=0,
    retina_index=None,
    device="cpu",
    conv3d=False,
    time_chunk_size=None,
    overlapping=False,
    num_of_layers=None,
    cell_indices_out_of_range=True,
    explainable_variance_threshold=None,
    oracle_correlation_threshold=None,
    normalize_responses=False,
    retina_specific_crops=True,
    extra_frame=0,
):
    """
    :param config: The config dictionary for the given dataset
    :param neuronal_data_dir: Directory containing response data for each session in a separate file
    and a config file for the given dataset named config
    :param train_image_path: Directory containing non-repeating input images used for training & validation.
                             Each trial is expected to have its own directory with one file containing all its images
    :param test_image_path: Directory containing repeating input images to be used for testing.
                            It is expected to only contain one file with all images in an .npy array.
    :param batch_size: batch size
    :param seed: random seed to be used for train/validation split
    :param train_frac: specifies the fraction of trials to be used for training
    :param subsample: specifies if the input image should be sub-sampled
    :param crop: specifies if and how the input image should be cropped
    :param use_cache: if True, the whole np.array for a specific trial is loaded
    :param num_of_trials_to_use: The maximum number of trials to use for training and validation
    :param movie_like: time is represented as channels, intended for ln models
    :param num_of_frames: Number of sequential frames to be returned on one getitem call
    :param cell_index: Index of cell on which we are training. If 'all' then all cells in dataset are used.
    :param retina_index: Index of the session on which we are training. If None, all sessions are used in a shared core
    :param device: 'cuda' or 'cpu'
    :param conv3d: Time is represented as a 3rd dimension
    :param time_chunk_size: The number of time-points that are supposed to be predicted from one chunk
    :param overlapping: Whether two sequential items produced by the dataloader overlap. If false,
                        you are only using 1/num_of_frames of the dataset. Only has impact if conv3d is true.
    :param num_of_layers: the number of (convolutional) layers the model we are training has.
    :param cell_indices_out_of_range: whether to remove cells with receptive fields outside of the crop
                                      if True, cell indices have to be provided in exlude_cells dict
                                      with session_id keys
    :param extra_frame: the number of frames you want to predict into the future

    :return: train, validation and test dataloaders for all (specified) sessions

    """

    # with open(os.path.join(neuronal_data_dir, 'config.yaml'), 'rb') as f:
    #     config = yaml.unsafe_load(f)

    dataloaders = {"train": {}, "validation": {}, "test": {}}

    all_train_ids, all_validation_ids = get_trial_wise_validation_split(
        train_image_path=train_image_path, train_frac=train_frac, seed=seed
    )
    # print('config: ', config)
    files = config["files"]
    # files = ['cell_data_01_NC.mat.pickle', 'cell_data_02_NC.mat.pickle', 'cell_data_03_NC.mat.pickle',
    #          'cell_data_04_NC.mat.pickle', 'cell_data_05_NC.mat.pickle']

    # TODO: Allow for a subset of retinas to be trained on, not just one or all
    if retina_index is not None:
        files = [files[retina_index]]

    for i, file in enumerate(sorted(files)):
        with open(os.path.join(neuronal_data_dir, file), "rb") as pkl:
            neural_data = pickle.load(pkl)
        index = int(file.split("_")[2]) - 1 if file.split("_")[2].isdigit() else 0

        session_id = str(index + 1).zfill(2)
        if cell_indices_out_of_range:
            cell_indices_out_of_range_list = config["exclude_cells"][session_id]
        else:
            cell_indices_out_of_range_list = []
        if (explainable_variance_threshold is not None) and (
            explainable_variance_threshold > 0
        ):
            ev_cell_indices_out_of_range_list = (
                get_exclude_cells_based_on_explainable_variance_threshold(
                    config=config,
                    retina_index=index,
                    threshold=explainable_variance_threshold,
                )
            )
        else:
            ev_cell_indices_out_of_range_list = []
        if (oracle_correlation_threshold is not None) and (
            oracle_correlation_threshold > 0
        ):
            oc_cell_indices_out_of_range_list = (
                get_exclude_cells_based_on_correlation_threshold(
                    retina_index=index,
                    config=config,
                    threshold=explainable_variance_threshold,
                )
            )
        else:
            oc_cell_indices_out_of_range_list = []
        test_responses = neural_data["test_responses"]
        test_responses = average_repeated_stimuli_responses(test_responses)
        cell_indices_out_of_range_list = list(
            set(
                cell_indices_out_of_range_list
                + list(
                    set(
                        oc_cell_indices_out_of_range_list
                        + ev_cell_indices_out_of_range_list
                    )
                )
            )
        )

        if cell_index is not None:
            train_responses = neural_data["train_responses"][
                cell_index : cell_index + 1, :, :
            ]
            test_responses = test_responses[cell_index : cell_index + 1]
        else:
            train_responses = neural_data["train_responses"]
            train_responses = np.delete(
                train_responses, cell_indices_out_of_range_list, axis=0
            )
            test_responses = np.delete(
                test_responses, cell_indices_out_of_range_list, axis=0
            )

        if normalize_responses:
            std = np.std(train_responses, axis=(1, 2))
            std = np.where(std > 0.1, std, 0.1)
            std_train = np.repeat(
                np.repeat(std.reshape(-1, 1), train_responses.shape[1], axis=1).reshape(
                    -1, train_responses.shape[1], 1
                ),
                train_responses.shape[2],
                axis=2,
            )
            std_test = np.repeat(std.reshape(-1, 1), test_responses.shape[1], axis=1)
            train_responses = np.divide(train_responses, std_train)
            test_responses = np.divide(test_responses, std_test)
        print("train responses shape: ", train_responses.shape)
        train_ids = np.isin(
            all_train_ids,
            np.arange(min(train_responses.shape[2], num_of_trials_to_use)),
        )
        train_ids = all_train_ids[train_ids]
        assert len(train_ids) > 0
        print("training trials: ", len(train_ids), train_ids)
        valid_ids = np.isin(
            all_validation_ids,
            np.arange(min(train_responses.shape[2], num_of_trials_to_use)),
        )
        valid_ids = all_validation_ids[valid_ids]
        assert len(valid_ids) > 0
        print("validation trials: ", len(valid_ids), valid_ids)
        print("getting loaders")

        # print('for STA calculation putting all trials in train')
        # train_ids = np.asarray(list(train_ids) + list(valid_ids))

        if retina_specific_crops:
            crop = config["big_crops"][session_id]
        train_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            path=train_image_path,
            indices=train_ids,
            test=False,
            batch_size=batch_size,
            use_cache=use_cache,
            cache_maxsize=2,
            movie_like=movie_like,
            num_of_frames=num_of_frames,
            num_of_hidden_frames=num_of_hidden_frames,
            device=device,
            conv3d=conv3d,
            crop=crop,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            overlapping=overlapping,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation,
            extra_frame=extra_frame,
        )

        valid_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            path=train_image_path,
            indices=valid_ids,
            test=False,
            batch_size=batch_size,
            use_cache=use_cache,
            cache_maxsize=1,
            movie_like=movie_like,
            num_of_frames=num_of_frames,
            num_of_hidden_frames=num_of_hidden_frames,
            device=device,
            conv3d=conv3d,
            crop=crop,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            overlapping=overlapping,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation,
            extra_frame=extra_frame,
        )

        test_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            path=test_image_path,
            indices=train_ids,
            test=True,
            batch_size=batch_size,
            use_cache=use_cache,
            cache_maxsize=20,
            movie_like=movie_like,
            num_of_frames=num_of_frames,
            num_of_hidden_frames=num_of_hidden_frames,
            device=device,
            conv3d=conv3d,
            crop=crop,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            overlapping=overlapping,
            num_of_layers=num_of_layers,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation,
            extra_frame=extra_frame,
        )

        dataloaders["train"][session_id] = train_loader
        dataloaders["validation"][session_id] = valid_loader
        dataloaders["test"][session_id] = test_loader

    return dataloaders
