import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
import torch
from datasets.stas import plot_all_stas, visualize_weights
from meis.MEI import get_mei_area
from utils.global_functions import get_cell_names, home


class LoggedEMEI:
    def __init__(self, mei_dir, filename, mei_file, e_mei_norm_value, cell_name, seed, epoch):
        self.mei_dir = mei_dir
        self.filename = filename
        self.e_mei_norm = e_mei_norm_value
        self.cell_name = cell_name
        self.seed = seed
        self.epoch = epoch
        self.mei_file = mei_file
        self.mei, self.activation = None, None
        self.load_mei()

    def load_mei(self):
        self.mei = get_logged_array(os.path.join(self.mei_dir, self.filename, f"cell_{self.cell_name}",
                                                 self.mei_file, f'seed_{self.seed}'), epoch=self.epoch)
        self.activation = get_logged_array(os.path.join(self.mei_dir, self.filename, f"cell_{self.cell_name}",
                                                        self.mei_file, f'seed_{self.seed}'), epoch=self.epoch,
                                           array_type='activation')


class LoggedSMEI:
    def __init__(self, mei_dir, filename, mei_file, e_mei_norm_value, cell_name, seed, epoch,
                 s_mei_norm_value, s_mei_file):
        self.e_mei = LoggedEMEI(mei_dir=mei_dir, filename=filename, mei_file=mei_file, e_mei_norm_value=e_mei_norm_value,
                               cell_name=cell_name, seed=seed, epoch=epoch)
        self.s_mei_norm = s_mei_norm_value
        self.s_mei_file = s_mei_file
        self.load_s_mei()

    def load_s_mei(self):
        self.activation_sup_sur = get_logged_array(
            os.path.join(self.e_mei.mei_dir, self.e_mei.filename,
                         f"cell_{self.e_mei.cell_name}",
                         self.e_mei.mei_file, f'seed_{self.e_mei.seed}',
                         'suppressive_surround', self.s_mei_file),
            epoch=self.e_mei.epoch, array_type='activation')
        self.complete_mei = get_logged_array(os.path.join(self.e_mei.mei_dir, self.e_mei.filename,
                         f"cell_{self.e_mei.cell_name}",
                         self.e_mei.mei_file, f'seed_{self.e_mei.seed}',
                         'suppressive_surround', self.s_mei_file),
            epoch=self.e_mei.epoch, array_type='whole_mei')
        self.masked_surround = get_logged_array(os.path.join(self.e_mei.mei_dir, self.e_mei.filename,
                         f"cell_{self.e_mei.cell_name}",
                         self.e_mei.mei_file, f'seed_{self.e_mei.seed}',
                         'suppressive_surround', self.s_mei_file),
            epoch=self.e_mei.epoch, array_type='masked_surround')
        self.surround = get_logged_array(os.path.join(self.e_mei.mei_dir, self.e_mei.filename,
                         f"cell_{self.e_mei.cell_name}",
                         self.e_mei.mei_file, f'seed_{self.e_mei.seed}',
                         'suppressive_surround', self.s_mei_file),
            epoch=self.e_mei.epoch, array_type='state')


def get_logged_mei_arrays(cell_meis, cell_index, cell_names, mei_dir, filename, string='lr_3', seed=None, epoch=-1,
                          norms=None):
    if seed is None:
        seed = [8]
    cell_name = cell_names[cell_index]
    epoch=-1
    mei_files = os.listdir(os.path.join(mei_dir, filename, f"cell_{cell_name}"))
    for mei_file in mei_files:
        if string in mei_file:
            mei_norm = mei_file.split('_')[4]
            if (mei_norm not in norms) and (norms is not None):
                continue
            if cell_index not in cell_meis.keys():
                cell_meis[cell_index] = {}
            logged_e_mei = LoggedEMEI(mei_dir=mei_dir,
                                      filename=filename,
                                      cell_name=cell_name,
                                      mei_file=mei_file,
                                      seed=seed,
                                      epoch=epoch,
                                      e_mei_norm_value=mei_norm,
                                     )
            cell_meis[cell_index][mei_norm] = logged_e_mei
    return cell_meis


def get_logged_suppressive_surrounds(cell_ss, cell_index, cell_names, mei_dir, filename, string='lr_3', ss_string='lr_3', epoch=-1, seed=None,
                                     mei_norms=None, s_mei_norms=None):
    if seed is None:
        seed = [8]
    cell_name = cell_names[cell_index]

    mei_files = os.listdir(os.path.join(mei_dir, filename, f"cell_{cell_name}"))
    for mei_file in mei_files:
        if string in mei_file:
            #             print(os.path.join(mei_dir, filename, f"cell_{cell_name}", mei_file, f'seed_{seed}', 'suppressive_surround'))
            mei_norm = float(mei_file.split('_')[4])
            if (f'{mei_norm:.1f}' not in mei_norms) and (mei_norms is not None):
                continue
            if os.path.isdir(os.path.join(mei_dir, filename, f"cell_{cell_name}", mei_file, f'seed_{seed}',
                                          'suppressive_surround')):
                sup_sur_files = [file for file in os.listdir(
                    os.path.join(mei_dir, filename, f"cell_{cell_name}", mei_file, f'seed_{seed}',
                                 'suppressive_surround')
                    ) if 'lr_' in file]
                cell_ss[f'{mei_norm:.1f}'] = {}
                print(mei_norm)
                for sup_sur_file in sup_sur_files:
                    print(sup_sur_file)
                    s_mei_norm = float(sup_sur_file.split('_')[3])
                    if (f'{s_mei_norm:.1f}' not in s_mei_norms) and (s_mei_norms is not None):
                        continue
                    mask_cut = sup_sur_file.split('_')[-1]
                    smei = LoggedSMEI(mei_dir=mei_dir,
                                      filename=filename,
                                      mei_file=mei_file,
                                      e_mei_norm_value=mei_norm,
                                      cell_name=cell_name,
                                      seed=seed,
                                      epoch=epoch,
                                      s_mei_norm_value=s_mei_norm,
                                      s_mei_file=sup_sur_file
                                      )
                    if f'{s_mei_norm:.1f}' not in cell_ss[f'{mei_norm:.1f}'].keys():
                        cell_ss[f'{mei_norm:.1f}'][f'{s_mei_norm:.1f}'] = {}
                    cell_ss[f'{mei_norm:.1f}'][f'{s_mei_norm:.1f}'][mask_cut] = smei
    return cell_ss


def get_model_activations(model, input_frames):
    input_frames = torch.tensor(input_frames)
    input_frames = input_frames.to('cuda').double()
    activations = model(input_frames)
    return activations


def get_logged_array(dir, epoch, array_type="state"):
    print(f"getting {array_type} for epoch {epoch}")
    logs = f"{dir}/{array_type}/logs.pkl"

    with open(logs, "rb") as f:
        try:
            states = pickle.load(f)
        except:
            return None
    if epoch < 0:
        print("returtning state for epoch", states["times"][epoch])
        return states["values"][epoch]
    if epoch not in states["times"]:
        print(f"Epoch {epoch} not recorded")
        return None
    else:
        state_index = states["times"].index(epoch)
        state = states["values"][state_index]
        return state


def make_mei_video(dir, epochs: list, mask_border=False, cmap="gray"):
    for epoch in epochs:
        mei = get_logged_array(dir, epoch)
        if mask_border:
            mask_border = get_mask_border(dir=dir, epoch=epoch)
        else:
            mask_border = None
        save_mei_video(
            epoch=epoch,
            current_state=mei,
            log_dir=dir,
            mask_border=mask_border,
            cmap=cmap,
        )


def save_mei_video(
    epoch,
    current_state,
    log_dir,
    mask_border=None,
    colormap="gray",
    prefix="mei",
    vmin=-1,
    vmax=1,
):
    Path(os.path.join(log_dir, "videos")).mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    max_value = np.max(np.abs(np.array(current_state)))

    if vmin is None:
        vmin = -1 * max_value
        vmax = max_value
    print(f"max value {prefix}: {max_value}")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    frames = visualize_weights(
        current_state[0, 0],
        ax,
        vmin=vmin,
        vmax=vmax,
        weight_index=0,
        mask_border=mask_border,
        cmap=colormap,
    )
    # plt.colorbar()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
    anim = animation.ArtistAnimation(
        fig, frames, interval=400, blit=True, repeat_delay=1000
    )
    anim.save(os.path.join(log_dir, "videos", f"{prefix}_e{epoch}.mp4"))


def plot_meis(mei):
    fig, ax = plt.subplots(mei.shape[1], mei.shape[0], figsize=(18, 25))
    for i in range(mei.shape[1]):
        for j in range(mei.shape[0]):
            ax[i, j].imshow(mei[j, i], vmin=-1, vmax=1, cmap="gray")
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])
            ax[0, j].set_title(f"Cell {j}", size="15")
    plt.axis("off")
    plt.savefig(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/tmp/meis.png",
        dpi=300,
    )
    plt.show()


def save_all_mei_videos(
    retina_index_str,
    hash,
    seed,
    lr,
    std,
    num_of_preds,
    file,
    cell_names,
    epoch,
    cmap="gray",
):
    meis = None
    for i, cell in enumerate(cell_names[:62]):
        mei = get_logged_array(
            os.path.join(
                home,
                "meis",
                "data",
                "salamander",
                f"retina{retina_index_str[1]}",
                file,
                f"cell_{cell}",
                f"{hash}_seed_{seed}_lr_{lr}_std_{std}_np_{num_of_preds}",
            ),
            epoch,
        )
        if meis is None:
            meis = np.zeros((len(cell_names),) + mei.shape[2:])
        if mei is not None:
            meis[i] = mei
    max_value = np.max(np.abs(np.array(meis)))
    # plot_meis(meis[:15, -15:], )
    return meis
    # plot_all_stas(retina_index_str=retina_index_str, saving_file=saving_file, cell_rfs=meis, cell_names=cell_names,
    #               cells=len(cell_names),
    #               vmin=None, vmax=None, cmap=cmap)


def get_mask_border(dir, epoch):
    mei = get_logged_array(dir, epoch)
    mask = get_mei_area(mei)
    border = np.array(1 * mask, dtype=np.int)
    border = np.diff(border)
    # np.imshow(border)
    return border


def get_all_cell_activations(
    retina_index_str, hash, seed, file, std, num_of_preds, lr, cell_names, epoch=-1
):
    activations = {}
    if num_of_preds is not None:
        np_str = f"_np_{num_of_preds}"
    else:
        np_str = ""
    for i, cell in enumerate(cell_names):
        activations[cell] = get_logged_array(
            os.path.join(
                home,
                "meis",
                "data",
                "salamander",
                f"retina{retina_index_str[1]}",
                file,
                f"cell_{cell}",
                f"{hash}_seed_{seed}_lr_{lr}_std_{std}{np_str}",
            ),
            epoch,
            array_type="activation",
        )
    return activations


def get_cross_activations(seed_lists, different_seed_meis, models, cell_index):
    cross_activations = {}
    for seed_list in [seed_lists]:
        cross_activations[seed_list] = {}
    for seed_1 in seed_lists:
        print('seed 1:', seed_1)
        for seed_2 in seed_lists:
            print('seed 2', seed_2)
            mei = different_seed_meis[seed_1][cell_index]['5'].mei
            seed_2_models = [int(x) if ' ' in seed_2 else int(seed_2) for x in seed_2.split(' ')]
            activations = []
            for seed in seed_2_models:
                model = models[seed][1]
                activations.append(get_model_activations(model, mei)[:, cell_index].item())
            activations = np.mean(activations)
            cross_activations[seed_1][seed_2] = activations
    return cross_activations

def get_cross_activations_for_all_cells(cell_names, different_seed_meis, models, seed_lists):
    cell_cross_activations = {}
    for cell in range(len(cell_names)):
        cell_cross_activations[cell] = get_cross_activations(seed_lists=seed_lists, different_seed_meis=different_seed_meis,
                                                             models=models, cell_index=cell)



if __name__ == "__main__":
    retina_index = 0
    hash = "2hQ7vp3FO7lE4"
    # hash = '4Dq3qS1uH2Zp'
    seed = [128, 1024, 42, 2048, 64, 256, 8]
    lr = 10
    num_of_preds = 1
    std = 0.025
    file = "lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x11x11_hk_25x7x7_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1"
    cell_names = get_cell_names(
        retina_index=retina_index,
        correlation_threshold=0,
        explained_variance_threshold=0.15,
    )
    save_all_mei_videos(
        retina_index_str=f"0{retina_index + 1}",
        saving_file=f"/usr/users/vystrcilova/retinal_circuit_modeling/meis/data/salamander/retina{retina_index + 1}/{file}/all_meis_{hash}_std_{std}_np_{num_of_preds}",
        epoch=-1,
        cell_names=cell_names,
        hash=hash,
        seed=seed,
        lr=lr,
        std=std,
        num_of_preds=num_of_preds,
    )
    exit()
    for cell_name in cell_names:
        dir = f"/usr/users/vystrcilova/retinal_circuit_modeling/meis/data/salamander/retina{retina_index + 1}/{file}/cell_{cell_name}"

        epochs = [0, 500, 2500, 5000]
        epochs = [5000]

        # get_mask_border(dir=dir, epoch=epochs[-1])
        make_mei_video(dir, epochs=epochs, mask_border=True)
        print(f"saved videos for cell {cell_name}")
