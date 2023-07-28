import argparse
import os
import random
import string

import torch
from torch import optim
from mei.objectives import EvaluationObjective, PostProcessedInputObjective
from mei.stoppers import NumIterations

import models
import wandb
from models.helper_functions import get_seed_model_versions
from meis.MEI import (ExcitingEnsembleMEI, ExcitingMEI, SuppressiveSurroundMEI,
                      optimize)
from meis.postprocessing import (ChangeStdAndClip, MaxNorm,
                                 PNormConstraintAndClip,
                                 ThresholdCenteredSigmoid, TorchL2NormAndClip)
from meis.preconditions import GaussianBlur3d, MultipleConditions
from meis.stoppers import ActivationIncrease
from meis.tracking import (GradientObjective, LoggingTracker,
                           SuppressiveLoggingTracker)
from meis.visualizer import get_logged_array
from models.helper_functions import get_model_and_dataloader, get_model_temp_reach, get_model_and_dataloader_for_nm
from utils.global_functions import (get_cell_names, global_config, home,
                                    mei_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--cell_index", default=57, help="index of cell")
parser.add_argument("--optimizer", default="SGD", type=str)
parser.add_argument("--layers", default=3, type=int)
parser.add_argument("--std", default=0.05, type=float)
parser.add_argument("--norm_value", default=20, type=float)
parser.add_argument("--init_variance", default=0.05, type=float)
parser.add_argument("--lr", default=3, type=float)
parser.add_argument("--hash", default="marmoset-ensemble", type=str)
parser.add_argument('--postprocessing', default='pnorm', type=str)
parser.add_argument('--log_dir', default='None', type=str)
parser.add_argument('--data_type', default='marmoset', type=str)
parser.add_argument('--retina_index', default=1, type=int)
parser.add_argument('--data_dir', default='/user/vystrcilova/')
parser.add_argument('--nm', default=1, type=int, help='whether or not to use natural movie stimulus')
parser.add_argument('--directory_prefix', default='factorized_4')
parser.add_argument('--filename', default='None')
parser.add_argument('--seed', default='8', type=str)

from time import gmtime, strftime


def run_pnorm_optimization():
    pass


if __name__ == "__main__":
    # norm_values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 15, 20]
    args = parser.parse_args()
    cell_index = args.cell_index
    if cell_index == "None":
        cells = list(range(0, 62))
    else:
        cells = [int(cell_index)]
    if args.log_dir == 'None':
        log_dir = home
    else:
        log_dir = args.log_dir
    retina_index = args.retina_index
    epoch = 100
    # seed = [128, 1024, 42, 64, 256, 8]
    # seed = [16, 8, 2048, 128, 256, 1024]
    # seed = [8]
    if args.seed == 'None':
        seed = None
    else:
        seed = [int(x) for x in args.seed.split(' ')]
    mei_seed = 18
    random.seed(mei_seed)

    # seed = None
    suppress = False
    init_variance = args.init_variance
    # lrs = [1, 10, 100]
    lr = args.lr
    sigma = 0.2
    sigma_temp = 0.1
    std = args.std
    nm = args.nm == 1
    # seed = [16, 128, 64, 8]
    # norm_value = args.norm_value
    data_type = args.data_type
    directory_prefix = args.directory_prefix
    hash = args.hash
    data_dir = args.data_dir
    if data_dir == 'None':
        data_dir = None
    nums_of_predictions = [1]

    # directory = f"{home}/models/basic_ev_0.15_cnn/{data_type}retina0{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    # mei_dir = f"{home}/meis/data/{data_type}/retina{retina_index + 1}"
    # filename = "lr_0.0100_l_1_ch_16_t_15_bs_16_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1"
    # fn = "models.BasicEncoder.build_trained"
    if nm:
        mei_dir_str = 'nm'
    else:
        mei_dir_str = 'wn'
    directory = f"{log_dir}/models/{directory_prefix}_ev_0.15_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
    mei_dir = f"{log_dir}/meis/data/{data_type}/{mei_dir_str}/retina{retina_index + 1}"
    # filename = 'lr_0.0094_l_3_ch_16_t_25_bs_16_tr_250_ik_25x17x17_hk_25x11x11_g_47.0000_gt_1.1453_l1_1.2520_l2_0.0000_sg_0.35_p_0_bn_1_norm_0_fn_1'
    # filename = "lr_0.0094_l_4_ch_64_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_7.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2_hdt_2_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90"
    # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2_hdt_1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
    # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-2-2_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_90_w_100'
    # filename = 'lr_0.0010_l_4_ch_16_t_20_bs_64_tr_250_ik_20x11x11_hk_20x7x7_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1'
    # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2_hdt_2_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
    # filename = 'lr_0.0094_l_3_ch_[8, 16, 32]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2_hdt_1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
    # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_16_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(11, 11)x(11, 11)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-1-1_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_90_w_100'

    if args.filename == 'None':
        # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_32_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(9, 9)x(9, 9)_g_47.0000_gt_1.4530_l1_1.2500_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-1-1_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
        filename = 'lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_16_tr_10_ik_27x(17, 17)x(17, 17)_hk_5x(3, 3)x(3, 3)_g_48.0000_gt_0.0740_l1_0.0230_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-4-6_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90'
        # filename = 'lr_0.0094_l_4_ch_[8, 16, 32, 64]_t_25_bs_32_tr_250_ik_25x(11, 11)x(11, 11)_hk_5x(5, 5)x(5, 5)_g_47.0000_gt_1.4530_l1_1.2520_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-2-2_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
    # marmoset
    # filename = 'lr_0.0054_l_4_ch_[4, 8, 16, 32]_t_27_bs_32_tr_10_ik_27x(12, 12)x(12, 12)_hk_5x(6, 6)x(6, 6)_g_8.0338_gt_0.0739_l1_0.0002_l2_0.0007_sg_0.25_d_2_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_90_w_100'
    #     filename = 'lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_32_tr_10_ik_27x(17, 17)x(17, 17)_hk_5x(3, 3)x(3, 3)_g_48.0338_gt_0.0739_l1_0.0222_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-4-6_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90'
    else:
        filename = args.filename

    fn = "models.FactorizedEncoder.build_trained"

    if isinstance(seed, list) or (seed is None):
        model, seed = get_seed_model_versions(
            model_name=filename,
            model_dir=directory,
            model_fn=fn,
            device="cuda",
            data_dir=data_dir,
            seeds=seed,
            config_dict=None,
            data_type=data_type,
            nm=nm
        )
    else:
        if nm:
            _, model, _ = get_model_and_dataloader(
                directory=directory,
                filename=filename,
                model_fn=fn,
                device="cuda",
                seed=seed,
                config_dict=None,
                data_type=data_type
            )
        else:
            _, model, _ = get_model_and_dataloader_for_nm(
                directory=directory,
                filename=filename,
                model_fn=fn,
                device="cuda",
                seed=seed,
                config_dict=None,
                data_type=data_type)
    norm_values = [5, 5.5, 6, 2, 2.5, 3, 3.5, 4, 4.5, 6.5, 7, 7.5, 8, 8.5, 9, 10, 12, 15, 20]
    for norm_value in norm_values:

        post_processing_dict = {"p": 2, "norm_value": norm_value}
        pps = [(PNormConstraintAndClip, post_processing_dict)]

        for cell_index in cells:
            cell_index = int(cell_index)
            for pp in pps:
                for num_of_predictions in nums_of_predictions:
                    # time = strftime()
                    # hash = ''.join([random.choice(string.ascii_letters) if x%3 != 0 else str(random.randint(0, 9)) for x in range(13)])
                    #                     for cell in range(model[0].config_dict['n_neurons_dict'][f"0{model[0].config_dict['retina_index'] + 1}"])[cell_index:cell_index+1]:
                    cell = cell_index
                    if args.optimizer == "SGD":
                        optimizer = torch.optim.SGD
                    else:
                        optimizer = torch.optim.Adam
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau
                    scheduler = optim.lr_scheduler.ExponentialLR

                    cell_name = get_cell_names(
                        retina_index=retina_index,
                        correlation_threshold=0,
                        explained_variance_threshold=0.15, config=model[0].config_dict['config']
                    )[cell_index]
                    print(f"cell {cell_name}, index: {cell_index}")
                    # initial_state = get_mei(os.path.join(home, 'meis', 'data',
                    #                                               f"retina{model[0].config_dict['retina_index'] + 1}",
                    #                                               model[0].config_dict['model_name'],
                    #                                               f'cell_{cell_name}',
                    #                                               f'seed_{seed}_lr_{0.1}_s_{2}_st_{2}',
                    #                                               ), epoch=-1)
                    initial_state = None
                    # preconditions = [(GaussianBlur3d, {'sigma': sigma, 'sigma_temp': sigma_temp})]
                    preconditions = []
                    postprocessings = [pp]
                    wandb.init(
                        config={
                            "model_name": model[0].config_dict["model_name"],
                            "cell": cell,
                            'cell_name': cell_name,
                            "lr": lr,
                            "seed": seed,
                            "optimizer": optimizer,
                            "mei_seed": mei_seed,
                            "random_initial": True,
                            "hash": hash,
                            "num_of_predictions": num_of_predictions,
                            "data_type": data_type,
                            "init_variance": init_variance,
                            'sigma': sigma,
                            'sigma_temp': sigma_temp,
                            **post_processing_dict
                        },
                        project=f"e-MEI-{data_type}-{'nm' if nm else 'wn'}",
                        entity="retinal-circuit-modeling", dir=log_dir
                    )
                    preconditions_names = [x[0] for x in preconditions]
                    postprocessings_names = [str(p[0]) for p in postprocessings]
                    wandb.config.update({"postprocessing": postprocessings_names})
                    wandb.config.update({"preconditions": preconditions_names})

                    for precondition in preconditions:
                        wandb.config.update(precondition[1])
                    for postprocessing in postprocessings:
                        wandb.config.update(postprocessing[1])

                    print(f"lr: {lr}", f"optimizer: {optimizer}")

                    mei = ExcitingEnsembleMEI(
                        models=model,
                        cell_index=cell_index,
                        input_shape=(
                            1,
                            1,
                            get_model_temp_reach(model[0].config_dict),
                            # 37,
                             80, 90,
                           # model[0].config_dict["img_h"],
                           #  model[0].config_dict["img_w"],
                        ),
                        learning_rate=lr,
                        device="cuda",
                        preconditions=preconditions,
                        postprocessing=[pp],
                        seed=mei_seed,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        initial_input=initial_state,
                        num_of_predictions=num_of_predictions,
                        init_variance=init_variance,
                        contrast=norm_value,
                    )
                    stopper = ActivationIncrease(
                        initial_activation=-1, minimal_change=0.00001, max_steps=10000, patience=1000
                    )
                    tracker = LoggingTracker(
                        log_dir=os.path.join(
                            log_dir,
                            "meis",
                            "data",
                            data_type,
                            f"retina{model[0].config_dict['retina_index'] + 1}",
                            model[0].config_dict["model_name"],
                            f"cell_{cell_name}",
                            f"{hash}_lr_{lr}_norm_{norm_value}_np_{num_of_predictions}_iv_{init_variance}",
                            f"seed_{seed}",
                        ),
                        seed=seed,
                        wandb_log=True,
                        log_frequency=2,
                        activation=EvaluationObjective(2),
                        state=PostProcessedInputObjective(100),
                        p_grad=GradientObjective(100),
                        video_log_frequency=100
                    )
                    optimize(
                        mei, stopper=stopper, tracker=tracker, save_tracks_interval=100
                    )
                    wandb.finish()
