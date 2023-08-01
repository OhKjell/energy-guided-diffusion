import os, yaml, torch
import numpy as np
from nnfabrik import builder
from tqdm import tqdm
import matplotlib.pyplot as plt
from dynamic.datasets.stas import show_sta
from dynamic.utils.global_functions import home, dataset_seed


def estimate_sta(dataloaders, num_of_frames, rf_size, dataset_config, device='cuda'):
    stas = torch.zeros(
        (
            dataloaders["train"][
                str(dataset_config["retina_index"] + 1).zfill(2)
            ].dataset.n_neurons,
            num_of_frames,
            rf_size[0],
            rf_size[1],
        ),
        dtype=torch.float16,
        device=device,
    )
    spike_counts = torch.zeros(
        (
            dataloaders["train"][
                str(dataset_config["retina_index"] + 1).zfill(2)
            ].dataset.n_neurons
        ),
        device=device,
    )
    for i, (images, responses) in enumerate(
        tqdm(dataloaders["train"][str(dataset_config["retina_index"] + 1).zfill(2)])
    ):
        responses = responses.transpose(1, 2).to(device).contiguous()
        responses = torch.flatten(responses, start_dim=0, end_dim=1)
        images = images.to(device)
        # continue
        for j, response in enumerate(responses.squeeze()):
            if response > 0:
                stas[j] = torch.add(stas[j], (response * images).squeeze())
            # if i % 1000 == 0:
            # print(j, 'max_sta', torch.max(stas[j]))
            # print(j, 'min_sta', torch.min(stas[j]))
            # print('spike_counts', spike_counts[j])
            # print()
        spike_counts = torch.add(spike_counts, responses.squeeze())

    normed_stas = np.zeros(stas.shape)
    print("spike counts: ", spike_counts)
    for i in range(stas.shape[0]):
        stas[i] = stas[i] / spike_counts[i]
        print(i, "max_sta", torch.max(stas[i]))
        print(i, "min_sta", torch.min(stas[i]))
        normed_stas[i] = stas[i].cpu().numpy() / np.linalg.norm(stas[i].cpu().numpy())
    # stas = np.asarray(stas.detach()
    for cell in range(
        dataloaders["train"][
            str(dataset_config["retina_index"] + 1).zfill(2)
        ].dataset.n_neurons
    ):
        # show_sta(stas[cell], cell)
        print(f"saving cell {cell}")
        # show_sta(normed_stas[cell].cpu().numpy(), cell, vmin=np.min(normed_stas[cell]), vmax=np.max(normed_stas[cell]))
        np.save(
            f"/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/stas/cell_data_02_WN_stas_{num_of_frames}_cell_{cell}.npy",
            stas[cell].cpu().numpy(),
        )


def rename_stas(sta_dir, sub_string):
    files = os.listdir(sta_dir)
    files = [x for x in files if sub_string in x]
    cell_names = []
    for file in tqdm(files):
        sta = np.load(os.path.join(sta_dir, file))
        # new_sta = sta-np.mean(sta)
        # plt.imshow(sta[15], cmap='gray')
        # plt.show()
        new_sta = np.flip(sta, axis=1)
        # plt.imshow(new_sta[15], cmap='gray')
        # plt.show()
        new_file_name_list = file.split("_")
        # new_file_name = '_'.join(new_file_name_list[:5]) + f'_{new_file_name_list[-1].split(".")[0]}_cell_{new_file_name_list[-2]}.npy'
        new_file_name = f"{'_'.join(new_file_name_list[:-2])}_flipped_{'_'.join(new_file_name_list[-2:])}"
        np.save(os.path.join(sta_dir, new_file_name), new_sta)
        # os.remove(os.path.join(sta_dir, file))


if __name__ == "__main__":
    # rename_stas(
    #     "/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/stas",
    #     "01_NC_stas_cell",
    # )
    # exit()
    """
    cuda = torch.cuda.is_available()
    if cuda:
        device = "cuda"
    else:
        device = "cpu"

    basepath = home

    with open(
        f"{basepath}/data/marmoset_data/responses/config_s4_2023.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)

    neuronal_data_path = os.path.join(basepath, config_dict["response_path"])
    img_dir = os.path.join("/user/vystrcilova/", config_dict["image_path"])

    retina_index = 0
    time_chunk = 1
    layers = 1
    explainable_variance_threshold = None

    batch_size = 1
    subsample = 1
    num_of_trials = 10
    num_of_frames = 25
    cell_index = "all"

    dataset_fn = "datasets.frame_movie_loader"
    dataset_config = dict(
        config=config_dict,
        basepath=basepath,
        img_dir_name=img_dir,
        neuronal_data_dir=neuronal_data_path,
        all_image_path=img_dir,
        batch_size=batch_size,
        seed=None,
        train_frac=0.8,
        subsample=subsample,
        crop=0,
        num_of_trials_to_use=num_of_trials,
        num_of_frames=num_of_frames,
        cell_index=None if cell_index == "all" else int(cell_index),
        retina_index=retina_index,
        device=device,
        time_chunk_size=time_chunk,
        num_of_layers=layers,
        explainable_variance_threshold=None,
        oracle_correlation_threshold=None,
        normalize_responses=False,
        final_training=True,
        full_img_h=300,
        full_img_w=350,
        padding=50,
        cell_indices_out_of_range=False,
        stimulus_seed=2023,
        retina_specific_crops=False,
    )

    dataloaders = builder.get_data(dataset_fn, dataset_config)
    estimate_sta(
        dataloaders=dataloaders, num_of_frames=num_of_frames, rf_size=(150, 200), dataset_config=dataset_config
    )"""
    dataset = 'marmoset_data'
    config_file = 'config_05'
    num_of_trials = 10
    num_of_frames = 40
    cell_index = 'all'
    retina_index = 1
    directory_prefix = ''

    with open(
            f"{home}/data/{dataset}/responses/{config_file}.yaml", "rb"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)


    dataset_fn = "datasets.white_noise_loader"
    dataset_config = dict(
        neuronal_data_dir=config_dict['neuronal_data_path'],
        train_image_path=config_dict['training_img_dir'],
        test_image_path=config_dict['test_img_dir'],
        batch_size=1,
        crop=0,
        temporal_dilation=1,
        hidden_temporal_dilation=1,
        subsample=1,
        seed=dataset_seed,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=False,
        num_of_frames=num_of_frames,
        num_of_hidden_frames=2,
        cell_index=None if cell_index == "all" else int(cell_index),
        retina_index=retina_index,
        conv3d=True,
        time_chunk_size=1,
        overlapping=False,
        num_of_layers=1,
        explainable_variance_threshold=None,
        oracle_correlation_threshold=None,
        normalize_responses=0,
        config=config_dict,
    )
    dataloaders = builder.get_data(dataset_fn, dataset_config)
    estimate_sta(dataloaders, 40, (150, 200), dataset_config=dataset_config)

