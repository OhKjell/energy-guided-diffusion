from dynamic.datasets.FrameWiseDataset import FrameWiseDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
from dynamic.datasets.natural_stimuli.create_dataset import get_img
from dynamic.datasets.whitenoise_salamander_loaders import average_repeated_stimuli_responses
from dynamic.utils.global_functions import (
    get_exclude_cells_based_on_explainable_variance_threshold,
    get_exclude_cells_based_on_correlation_threshold,
    set_random_seed,
)


def get_dataloader(
    response_dict,
    path,
    indices,
    test,
    frames,
    batch_size,
    fixations,
    temporal_dilation=1,
    shuffle=False,
    num_of_frames=15,
    num_of_hidden_frames=None,
    device="cpu",
    crop=0,
    subsample=1,
    time_chunk_size=1,
    num_of_layers=1,
    hidden_temporal_dilation=1,
    padding=200,
    full_img_w=1000,
    full_img_h=800,
    img_h=600,
    img_w=800,
):
    data = FrameWiseDataset(
        response_dict,
        path,
        "images",
        "responses",
        frames=frames,
        fixations=fixations,
        indices=indices,
        transforms=[],
        test=test,
        crop=crop,
        temporal_dilation=temporal_dilation,
        hidden_temporal_dilation=hidden_temporal_dilation,
        subsample=subsample,
        device=device,
        time_chunk_size=time_chunk_size,
        num_of_layers=num_of_layers,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=num_of_hidden_frames,
        padding=padding,
        full_img_h=full_img_h,
        full_img_w=full_img_w,
        img_h=img_h,
        img_w=img_w,
    )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def process_fixations(fixations, flip_imgs=False, select_flip=None):
    if not flip_imgs:
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0]),
            }
            for f in fixations
        ]
    else:
        print('FLIPPING THE OTHER WAY AROUND!')
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0] == '0'),
            }
            for f in fixations
    ]
    if select_flip is not None:
        fixations = [x for x in fixations if x['flip'] == select_flip]
    return fixations


def get_trial_wise_validation_split(
    train_responses, train_frac, seed=None, final_training=False, hard_coded=None
):
    if hard_coded is not None:
        print(f'hard coded train trials:{hard_coded[0]}')
        print(f'hard coded validation trials: {hard_coded[1]}')
        return hard_coded[0], hard_coded[1]
    num_of_trials = train_responses.shape[-1]
    if seed is not None:
        set_random_seed(seed)
    else:
        set_random_seed(1000)
    train_idx, val_idx = np.split(
        np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)]
    )
    # in the marmoset dataset for seed 2022, the last ~3000 frames are the same as the first ~3000 frames, therefore,
    # they need to be in the same dataset part in the marmoset dataset

    # for seed 2023, it's the same, therefore, they need to be in
    # the same dataset part, this then holds for trials 10 and 20 if all trials are used or 0 and 10 if only the 2023
    # seed trials are used. This however is by ok with the default seed so we don't worry about it now
    # TODO: might worry about it later
    while ((0 in train_idx) and (9 in val_idx)) or (
        (9 in train_idx) and (0 in val_idx)
    ):
        train_idx, val_idx = np.split(
            np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)]
        )

    if final_training:
        train_idx = list(train_idx) + list(val_idx)
    else:
        assert not np.any(
            np.isin(train_idx, val_idx)
        ), "train_set and val_set are overlapping sets"

    print(f"train idx: {train_idx}")
    print(f"val idx: {val_idx}")
    return train_idx, val_idx


def load_frames(basepath, img_dir_name, frame_file, full_img_w, full_img_h):
    # images = os.listdir(f'{basepath}/data/marmoset_data/{img_dir_name}/')
    images = os.listdir(img_dir_name)
    images = [frame for frame in images if frame_file in frame]
    all_frames = np.zeros((len(images), full_img_w, full_img_h), dtype=np.float16)
    i = 0
    for img_file in tqdm(sorted(images)):
        # img = get_img(img=img, directory=f'{basepath}/data/marmoset_data/{img_dir_name}/', show=False)
        # img = np.load(f'{basepath}/data/marmoset_data/{img_dir_name}/{img_file}')
        img = np.load(f"{img_dir_name}/{img_file}")

        all_frames[i] = img / 255
        i += 1
    # all_frames = np.moveaxis(all_frames, 0, 2)
    print("normalizing")
    return all_frames


def frame_movie_loader(
    config: dict,
    neuronal_data_dir,
    basepath,
    all_image_path,
    batch_size=16,
    seed=None,
    train_frac=0.8,
    subsample=1,
    crop=0,
    num_of_trials_to_use=None,
    start_using_trial=0,
    num_of_frames=None,
    num_of_hidden_frames=None,
    temporal_dilation=1,
    hidden_temporal_dilation=1,
    cell_index=None,
    retina_index=None,
    device="cpu",
    time_chunk_size=None,
    num_of_layers=None,
    cell_indices_out_of_range=True,
    explainable_variance_threshold=None,
    oracle_correlation_threshold=None,
    normalize_responses=False,
    frame_file="_img_",
    img_dir_name="stimuli",
    full_img_w=1400,
    full_img_h=1200,
    final_training=False,
    padding=200,
    retina_specific_crops=True,
    stimulus_seed=2021,
    hard_coded=None,
    flip_imgs=False,
    select_flip=None

):
    dataloaders = {"train": {}, "validation": {}, "test": {}}
    if retina_index is None:
        with open(
                f"{basepath}/{config['fixation_file']['01']}", "r"
        ) as file:
            fixation_file = file.readlines()
            fixations = process_fixations(fixation_file, flip_imgs=flip_imgs)
    else:
        with open(
            f"{basepath}/{config['fixation_file'][str((retina_index + 1)).zfill(2)]}", "r"
        ) as file:
            fixation_file = file.readlines()
            fixations = process_fixations(fixation_file, flip_imgs=flip_imgs)

    files = config["files"]
    img_h = full_img_h - 3 * padding
    img_w = full_img_w - 3 * padding
    # TODO: Allow for a subset of retinas to be trained on, not just one or all
    if retina_index is not None:
        files = [files[retina_index]]
    frames = None
    for i, file in enumerate(files):
        with open(os.path.join(neuronal_data_dir, file), "rb") as pkl:
            neural_data = pickle.load(pkl)

        session_id = str(file.split("_")[2])
        session_id_int = int(session_id) -1
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
                    retina_index=session_id_int if retina_index is None else retina_index,
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
                    retina_index=session_id_int if retina_index is None else retina_index,
                    config=config,
                    threshold=explainable_variance_threshold,
                )
            )
        else:
            oc_cell_indices_out_of_range_list = []
        test_responses = neural_data["test_responses"]

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
        num_trials = train_responses.shape[2]
        if 'seeds' in neural_data.keys():
            seed_info = neural_data["seeds"]
            if (stimulus_seed in seed_info):
                trials_assigned_to_seed = neural_data["trial_separation"][stimulus_seed]
                train_responses = train_responses[:, :, trials_assigned_to_seed]

                test_responses = test_responses[:, :, trials_assigned_to_seed]
            elif (retina_index == 0) or (session_id[1] == '1'):
                test_responses = test_responses[:, :, :10]

        test_responses = average_repeated_stimuli_responses(test_responses)

        all_train_ids, all_validation_ids = get_trial_wise_validation_split(
            train_responses=train_responses,
            train_frac=train_frac,
            seed=seed,
            final_training=final_training,
            hard_coded=hard_coded
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
        if hard_coded is None:

            train_ids = np.isin(
            all_train_ids,
            np.arange(start_using_trial, min(train_responses.shape[2], num_of_trials_to_use+start_using_trial)),
        )
            train_ids = np.asarray(all_train_ids)[train_ids]
        else:
            train_ids = [x for x in all_train_ids if x < min(num_trials, num_of_trials_to_use+start_using_trial)]
        assert len(train_ids) > 0
        print("training trials: ", len(train_ids), train_ids)
        if hard_coded is None:
            valid_ids = np.isin(
            all_validation_ids,
            np.arange(start_using_trial, min(train_responses.shape[2], num_of_trials_to_use+start_using_trial)),
        )
            valid_ids = all_validation_ids[valid_ids]
        else:
            valid_ids = [x for x in all_validation_ids if x < min(num_trials, num_of_trials_to_use+start_using_trial)]
        #     assert len(valid_ids) > 0
        print("validation trials: ", len(valid_ids), valid_ids)
        print("getting loaders")
        # print('FOR STA PURPOSES ALL TRIALS ARE TRAIN TRIALS')
        # train_ids = np.array(list(valid_ids) + list(train_ids))

        if frames is None:
            frames = load_frames(
            basepath=basepath,
            img_dir_name=img_dir_name,
            frame_file=frame_file,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
        )

        if retina_specific_crops:
            crop = config["big_crops"][session_id]
        train_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            fixations=fixations,
            path=all_image_path,
            indices=train_ids,
            test=False,
            batch_size=batch_size,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=False,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        valid_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            path=all_image_path,
            indices=valid_ids,
            test=False,
            batch_size=batch_size,
            fixations=fixations,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=False,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        test_loader = get_dataloader(
            {"train_responses": train_responses, "test_responses": test_responses},
            fixations=fixations,
            path=all_image_path,
            indices=train_ids,
            test=True,
            batch_size=batch_size,
            num_of_frames=num_of_frames,
            device=device,
            crop=crop,
            shuffle=False,
            subsample=subsample,
            time_chunk_size=time_chunk_size,
            num_of_layers=num_of_layers,
            frames=frames,
            num_of_hidden_frames=num_of_hidden_frames,
            padding=padding,
            full_img_h=full_img_h,
            full_img_w=full_img_w,
            img_h=img_h,
            img_w=img_w,
            temporal_dilation=temporal_dilation,
            hidden_temporal_dilation=hidden_temporal_dilation
        )

        dataloaders["train"][session_id] = train_loader
        dataloaders["validation"][session_id] = valid_loader
        dataloaders["test"][session_id] = test_loader

    return dataloaders
