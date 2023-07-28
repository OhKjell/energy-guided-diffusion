from models.model_visualizations import plot_curves, visualize_mutli_channel_cnn
import os
import numpy as np

from utils.global_functions import home


def visualize_cnn_readout_and_filters(
    file,
    best_corr_threshold,
    retina_index=0,
    layers=1,
    cnn_prefix="basic",
    seed_str=None,
    data_type="salamander",
    img_h=150,
    img_w=200,
):
    if seed_str is None:
        seed_str = "8"
    print("starting", file)
    correlation_file = os.path.join(
        directory, file, "stats", f"seed_{seed_str}", f"{curve}.npy"
    )
    if os.path.isfile(correlation_file):
        print("considering", file)
        correlations = os.path.join(correlation_file)
        correlations = np.load(correlations)
        if len(correlations) == 0:
            return
        if np.max(correlations) > best_corr_threshold:
            if "_" in file:
                print("plotting", file, f"with corr: {np.max(correlations)}")
                visualize_mutli_channel_cnn(
                    f"models/{cnn_prefix}_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/{file}/weights/seed_{seed_str}/best_model.m",
                    layers=layers,
                    vmin=-0.7,
                    vmax=0.7,
                    corr_threshold=0.0,
                    readout_index=retina_index + 1,
                    img_h=img_h,
                    img_w=img_w,
                )


if __name__ == "__main__":
    retina_index = 0
    all_correlations = {}
    retina_specific_correlations = {}
    plot_curves(f'{home}/models/multiretinal_factorized_ev_0.15_cnn/salamander/retinaall/cell_None/readout_isotropic/gmp_0/',
            'correlations',  file_suffix=f'_retinaall', max_lenght=1000, best_corr_threshold=0., seed=None,)

    for retina in range(1,2):
        # files = ['lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x(17, 17)x(17, 17)_hk_25x(11, 11)x(11, 11)_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1']
        # print(f'retina {retina}')
        """plot_curves(
            f'{home}/models/factorized_ev_0.15_cnn/marmoset/retina{retina+1}/cell_None/readout_isotropic/gmp_0/',
            'correlations',  file_suffix=f'_retina{retina+1}', max_lenght=1000, best_corr_threshold=0., seed=None)
        plot_curves(
            f'{home}/models/factorized_ev_0.15_cnn/marmoset/retina0{retina + 1}/cell_None/readout_isotropic/gmp_0/',
            'correlations', file_suffix=f'_retina{retina + 1}', max_lenght=1000, best_corr_threshold=0., seed=None)"""

        print(f"retina {retina}")
        plot_curves(
            f'{home}/models/factorized_4_ev_0.15_cnn/marmoset/retina0{retina+1}/cell_None/readout_isotropic/gmp_0/',
            'correlations',  file_suffix=f'_retina{retina+1}', max_lenght=1000, best_corr_threshold=0., seed=None,
        file_substring='')
        print()
        exit()
        # print('salamander')
        file_correlations = plot_curves(
            f"{home}/models/factorized_ev_0.15_cnn/salamander/retina{retina + 1}/cell_None/readout_isotropic/gmp_0/",
            "correlations",
            file_suffix=f"_retina{retina + 1}",
            max_lenght=1000,
            best_corr_threshold=0.0,
            seed=None,
            file_substring="l_",
        )
        retina_specific_correlations[retina] = {}
        for k, v in file_correlations.items():
            if k[0] in all_correlations.keys():
                all_correlations[k[0]].append(v[0])
            else:
                all_correlations[k[0]] = [v[0]]
            retina_specific_correlations[retina][k[0]] = v[0]
    all_correlations = {
        k: v
        for k, v in sorted(
            all_correlations.items(), key=lambda item: np.mean(item[1]), reverse=True
        )
    }
    print("")
    for k, v in all_correlations.items():
        if len(v) == 5:
            print(k, np.mean(v), v)
    print()
    print()
    print()
    best_avg = []
    for retina_index in range(5):
        print(f"best 5 for retina {retina_index}")
        retina_specific_correlations[retina_index] = {
            k: v
            for k, v in sorted(
                retina_specific_correlations[retina_index].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for i, (k, v) in enumerate(retina_specific_correlations[retina_index].items()):
            if i <= 5:
                print(k, v)
            if i == 0:
                best_avg.append(v)
    print(f"best mean: {np.mean(best_avg)}, {best_avg}")

    exit()
    directory = f"{home}/models/factorized_4_cnn/marmoset/retina{retina_index+1}/cell_None/readout_isotropic/gmp_0/"
    curve = "correlations"
    files = None
    files = [
        "lr_0.0090_l_1_ch_16_t_15_bs_16_tr_10_ik_15x(20, 20)x(20, 20)_g_1.3300_gt_0.0030_l1_0.5000_l2_0.0000_sg_0.2_p_0_bn_1_s_1norm_0_fn_1_2023"
    ]
    best_corr_threshold = 0.0
    if files is None:
        files = os.listdir(directory)
    retina_index = 0
    for file in files:
        # if ('l_3' in file) and ('ch_8' in file):
        visualize_cnn_readout_and_filters(
            file,
            best_corr_threshold,
            retina_index=retina_index,
            layers=1,
            cnn_prefix="factorized_4",
            data_type="marmoset",
            img_h=150,
            img_w=200,
            seed_str=8,
        )
