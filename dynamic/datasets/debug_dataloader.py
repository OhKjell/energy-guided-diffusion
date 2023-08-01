from dynamic.utils.global_functions import dataset_seed
from nnfabrik import builder
import os
from dynamic.utils.global_functions import big_crops
import numpy as np

basepath = "/Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling/"


def get_cnn_dataloader(
    neuronal_data_path,
    training_img_dir,
    test_img_dir,
    batch_size,
    crop,
    subsample,
    num_of_trials,
    num_of_frames,
    cell,
    retina_index,
    normalize_response,
    conv3d=True,
    layers=1,
    overlapping=True,
):
    dataset_fn = "datasets.white_noise_loader"
    dataset_config = dict(
        neuronal_data_dir=neuronal_data_path,
        train_image_path=training_img_dir,
        test_image_path=test_img_dir,
        batch_size=batch_size,
        crop=crop,
        subsample=subsample,
        seed=dataset_seed,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=False,
        num_of_frames=num_of_frames,
        cell_index=None,
        retina_index=retina_index,
        conv3d=conv3d,
        num_of_layers=layers,
        overlapping=False,
        time_chunk_size=150,
    )
    dataloaders = builder.get_data(dataset_fn, dataset_config)
    return dataloaders


def get_ln_model_dataloader(
    neuronal_data_path,
    training_img_dir,
    test_img_dir,
    batch_size,
    crop,
    subsample,
    num_of_trials,
    num_of_channels,
    cell,
    retina_index,
    normalize_response,
):
    dataset_fn = "datasets.white_noise_loader"
    dataset_config = dict(
        neuronal_data_dir=neuronal_data_path,
        train_image_path=training_img_dir,
        test_image_path=test_img_dir,
        batch_size=batch_size,
        crop=crop,
        subsample=subsample,
        seed=dataset_seed,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=True,
        num_of_frames=num_of_channels,
        cell_index=cell,
        retina_index=retina_index,
        normalize_responses=normalize_response,
    )
    dataloaders = builder.get_data(dataset_fn, dataset_config)
    return dataloaders


def compare_dataloaders(cnn_loader, ln_loader, cell):
    print(
        "train length - cnn:",
        cnn_loader["train"]["01"].dataset._len,
        "ln:",
        ln_loader["train"]["01"].dataset._len,
    )
    print(
        "validation length - cnn:",
        cnn_loader["validation"]["01"].dataset._len,
        "ln:",
        ln_loader["validation"]["01"].dataset._len,
    )
    print(
        "test length - cnn:",
        cnn_loader["test"]["01"].dataset._len,
        "ln:",
        ln_loader["test"]["01"].dataset._len,
    )

    for (ln_images, ln_responses), (cnn_images, cnn_responses) in zip(
        ln_loader["validation"][str(retina_index + 1).zfill(2)],
        cnn_loader["validation"][str(retina_index + 1).zfill(2)],
    ):
        print(ln_images.shape, ln_responses.shape)
        # ln_images = transform_according_to_cnn_loader(ln_images, cnn_loader)
        # print(ln_images.shape, ln_responses.shape)
        # print(cnn_images.shape, cnn_responses.shape)
        # for i, (x,y) in enumerate(zip(ln_responses, cnn_responses[:, 0])):
        #     print(f'ln_response: {x}, cnn response: {y}')
        #     assert (ln_images[i] == cnn_images[i, 0]).all()
        # print('')


def transform_according_to_cnn_loader(images, cnn_loader):
    dataset = cnn_loader["validation"]["01"].dataset
    _, num_of_images, h, w = images.shape
    images = images[
        :,
        :,
        dataset.crop[0] : h - dataset.crop[1] : dataset.subsample,
        dataset.crop[2] : w - dataset.crop[3] : dataset.subsample,
    ]
    return images


if __name__ == "__main__":
    neuronal_data_path = os.path.join(basepath, "data/responses/")
    training_img_dir = os.path.join(basepath, "data/non_repeating_stimuli/")
    test_img_dir = os.path.join(basepath, "data/repeating_stimuli/")
    batch_size = 10
    retina_index = 0
    crop_ln = 0
    crop_cnn = big_crops[str(retina_index + 1).zfill(2)]
    subsample = 1
    num_of_trials = 250
    num_of_frames = 15
    cell_index = 0
    normalize_responses = False

    cnn_loader = get_cnn_dataloader(
        neuronal_data_path,
        training_img_dir,
        test_img_dir,
        batch_size,
        crop=crop_cnn,
        subsample=subsample,
        num_of_trials=num_of_trials,
        num_of_frames=num_of_frames,
        cell=cell_index,
        retina_index=retina_index,
        normalize_response=normalize_responses,
    )

    ln_loader = get_ln_model_dataloader(
        neuronal_data_path,
        training_img_dir,
        test_img_dir,
        batch_size,
        crop=crop_ln,
        subsample=subsample,
        num_of_trials=num_of_trials,
        num_of_channels=num_of_frames,
        cell=cell_index,
        retina_index=retina_index,
        normalize_response=normalize_responses,
    )
    compare_dataloaders(cnn_loader, ln_loader, cell=cell_index)
