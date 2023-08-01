from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import yaml
from dynamic.datasets.stas import visualize_weights

# from models.model_visualizations import visualize_all_gaussian_readout, home
import matplotlib.animation as animation
import seaborn as sns

from dynamic.models.helper_functions import get_seed_model_versions
from dynamic.utils.global_functions import home

sns.set()


def visualized_factorized_filters(
    model, layer, visualization_dir, max_corr, max_corr_epoch
):
    layer_str = "" if layer == 0 else f"_{layer}"
    spatial_kernel = model.core.features[layer][0].weight.detach().cpu().numpy()
    temporal_kernel = model.core.features[layer][1].weight.detach().cpu().numpy()

    fig, ax = plt.subplots(
        # temporal_kernel.shape[0] + 1, spatial_kernel.shape[0], figsize=(30, 30)
    2, spatial_kernel.shape[0], figsize = (26, 4)
    )
    timed_frames = [[] for _ in range(temporal_kernel.shape[2])]
    max_value = np.max(np.abs(spatial_kernel))
    for kernel in range(spatial_kernel.shape[0]):
        ax[0, kernel].imshow(
            spatial_kernel[kernel, 0, 0], cmap="gray", vmin=-1.2*max_value, vmax=1.2*max_value
        )
        ax[0, kernel].set_title(f"Spatial kernel {kernel}")

    for out_channel in range(temporal_kernel.shape[0]):
        for in_channel in range(spatial_kernel.shape[0]):
            ax[1, in_channel].plot(np.arange(temporal_kernel.shape[2]), temporal_kernel[out_channel, in_channel, :, 0, 0])
            ax[1, in_channel].set_title(f'Temporal kernels for spatial kernel {in_channel}')
            # weights = np.array(
            #     [
            #         spatial_kernel[in_channel, 0, 0]
            #         * temporal_kernel[out_channel, 0, time, 0, 0]
            #         for time in range(temporal_kernel.shape[2])
            #     ]
            # )
            # frames = visualize_weights(
            #     weights,
            #     ax[out_channel + 1, in_channel],
            #     vmin=-0.1,
            #     vmax=0.1,
            #     weight_index=0,
            # )
            # ax[out_channel + 1, in_channel].set_title(
            #     f"S {in_channel} x T {out_channel}"
            # )
            # ax[out_channel + 1, in_channel].axis("off")
            # for i, frame in enumerate(frames):
            #     timed_frames[i].append(frame[0])
    fig.suptitle(
        f"Layer {layer} (Max corr {max_corr:.2f} in epoch {max_corr_epoch})",
        va="bottom",
        size="xx-large",
    )
    plt.rcParams["axes.grid"] = False
    fig.tight_layout()
    Path(f"{visualization_dir}/layer_{layer}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{visualization_dir}/layer_{layer}/conv_filters.png", dpi=300, bbox_inches='tight')
    plt.show()
    # anim = animation.ArtistAnimation(
    #     fig, timed_frames, interval=400, blit=True, repeat_delay=1000
    # )
    # anim.save(f"{visualization_dir}/layer_{layer}/conv_filters.mp4")


def visualize_temporal_kernels(model, visualization_dir, layer):
    layer_str = "" if layer == 0 else f"_{layer}"
    temporal_kernel = model[
        f"core.features.layer{layer}.conv_temporal{layer_str}.weight"
    ]
    i = 0
    fig, ax = plt.subplots(temporal_kernel.shape[0], figsize=(16, 48))
    im = None
    for out_channel in range(temporal_kernel.shape[0]):
        im = ax[out_channel].imshow(
            temporal_kernel[out_channel, :, :, 0, 0],
            cmap="coolwarm",
            vmin=-0.7,
            vmax=0.7,
        )
        ax[out_channel].set_title(f"Out channel {out_channel}")
        ax[out_channel].set_xticks([])
        ax[out_channel].set_yticks([x for x in range(temporal_kernel.shape[1])])
        ax[out_channel].set_yticklabels(
            [f"In {x}" for x in range(temporal_kernel.shape[1])]
        )
    # plt.colorbar(im, ax=ax, aspect=7)
    plt.tight_layout()
    # plt.tight_layout()
    # fig, ax = plt.subplots(temporal_kernel.shape[1], 1, figsize=(10, 20))
    # for in_channel in range(temporal_kernel.shape[1]):
    #     ax[i].imshow(np.reshape(temporal_kernel[out_channel, in_channel, :, 0, 0], (1, -1)), cmap='coolwarm')
    #     ax[i].set_ylabel(f'Out {out_channel} x In {in_channel}', rotation=90)
    #     i += 1
    plt.savefig(
        f"{home}/{visualization_dir}/layer_{layer}/temporal_filters.png",
        bbox_to_anchor="tight",
    )
    plt.show()


def visualize_factorized_cnn(filename, directory, model_fn, seed, layers, data_dir, data_type, nm):
    models, seeds = get_seed_model_versions(
            model_name=filename,
            model_dir=directory,
            model_fn=model_fn,
            device="cuda",
            data_dir=data_dir,
            seeds=seed,
            config_dict=None,
            data_type=data_type,
            nm=nm
        )

    for model, seed in zip(models, seeds):
        visualization_dir = os.path.join(directory, filename, "visualizations", f'seed_{seed}')
        # correlation = np.load(os.path.join(home, model_file, 'stats', 'correlations.npy'))
        correlation = np.load(os.path.join(directory, filename, "stats", f'seed_{seed}', "correlations.npy"))

        max_corr = max(correlation)
        best_epoch = np.argmax(correlation)
        # visualize_readout(model, visualization_dir, vmin=vmin, vmax=vmax)
        # visualize_all_gaussian_readout(model, visualization_dir, readout_index=retina_index+1,
        #                               retina_index=retina_index, spatial_str='_spatial',
        #                                correlation_threshold=oracle_correlation_threshold,
        #                                explainable_variance_threshold=explainable_variance_threshold)
        for layer in range(layers):
            visualized_factorized_filters(
                model, layer, visualization_dir, max_corr, best_epoch
            )
            # visualize_temporal_kernels(model, visualization_dir, layer=layer)


if __name__ == "__main__":
    retina_index = 1
    data_type = 'marmoset'
    # nm smooth
    # filename = 'lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_16_tr_10_ik_27x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_48.0000_gt_0.0740_l1_0.0230_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90'
    #wn smooth
    # filename = 'lr_0.0073_l_4_ch_[8, 16, 32, 64]_t_25_bs_32_tr_250_ik_25x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_48.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
    # nm no smooth
    # filename = 'lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_16_tr_10_ik_27x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_0.0000_gt_0.0740_l1_0.0230_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90'
    # wn non-smooth
    filename = 'lr_0.0073_l_4_ch_[8, 16, 32, 64]_t_25_bs_32_tr_250_ik_25x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_0.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'

    directory = f'{home}/models/factorized_ev_0.15_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/'
    visualize_factorized_cnn(filename=filename, directory=directory, model_fn='models.FactorizedEncoder.build_trained',
                             seed=[8, 16, 64, 128, 256], data_dir='/user/vystrcilova/', layers=1, data_type=data_type, nm=False

    )
