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
from meis.MEI import (ExcitingEnsembleMEI, ExcitingMEI, SuppressiveSurroundMEI, SuppressiveSurroundEnsembleMEI,
                      optimize)
from meis.postprocessing import (ChangeStdAndClip, MaxNorm,
                                 PNormConstraintAndClip,
                                 ThresholdCenteredSigmoid, TorchL2NormAndClip)
from meis.preconditions import GaussianBlur3d, MultipleConditions
from meis.stoppers import ActivationIncrease
from meis.tracking import (GradientObjective, LoggingTracker,
                           SuppressiveLoggingTracker, WholeMeiObjective, MaskedSurroundObjective)
from meis.visualizer import get_logged_array
from models.helper_functions import get_model_and_dataloader, get_model_temp_reach, get_model_and_dataloader_for_nm
from utils.global_functions import (get_cell_names, global_config, home,
                                    mei_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--cell_index", default=5, help="index of cell")
parser.add_argument("--optimizer", default="SGD", type=str)
parser.add_argument("--layers", default=3, type=int)
parser.add_argument("--std", default=0.05, type=float)
parser.add_argument("--norm_scaler", default=2, type=float)
parser.add_argument("--init_variance", default=0.05, type=float)
parser.add_argument("--lr", default=1, type=float)
parser.add_argument("--e_mei_hash", default="hlrn-no-smoothing", type=str)
parser.add_argument('--postprocessing', default='pnorm', type=str)
parser.add_argument('--log_dir', default='/scratch/usr/nibmvyst/wandb/', type=str)
parser.add_argument('--data_type', default='marmoset', type=str)
parser.add_argument('--retina_index', default=1, type=int)
parser.add_argument('--nm', default=0, type=int, help='whether or not to use natural movie stimulus')
parser.add_argument('--mei_dir', default='None')
parser.add_argument('--data_dir', default='/user/vystrcilova/')
parser.add_argument('--mask_cutoff', default=0.75, type=float)
parser.add_argument('--e_mei_norm', default=2, type=int)
parser.add_argument('--e_mei_lr', default=2.0, type=float)
parser.add_argument('--e_mei_np', default=1, type=int, help='num_of_predictions for exciting mei')
parser.add_argument('--e_mei_iv', default=0.05, type=float, help='init variance for exciting mei')
parser.add_argument('--e_mei_seed', default='8 16', type=str, help='seed(s) for exciting mei')


if __name__ == "__main__":
    args = parser.parse_args()
    cell_index = int(args.cell_index)
    if cell_index == "None":
        cells = list(range(0, 70))
    else:
        cells = [int(cell_index)]

    retina_index = args.retina_index
    epoch = -1

    if args.e_mei_seed == 'None':
        seed = None
    else:
        seed = [int(x) for x in args.e_mei_seed.split(' ')]
    e_hash = args.e_mei_hash
    e_init_variance = args.e_mei_iv
    e_np = args.e_mei_np
    e_lr = args.e_mei_lr
    e_norm_value = args.e_mei_norm

    mei_seed = 18
    random.seed(mei_seed)
    suppress = True

    # lrs = [1, 10, 100]
    lr = args.lr
    sigma = 0.2
    sigma_temp = 0.1
    std = args.std
    nm = args.nm == 1
    mask_cutoff = args.mask_cutoff
    data_type = args.data_type
    nums_of_predictions = e_np
    norm_scaler = args.norm_scaler
    log_dir = args.log_dir

    data_dir = args.data_dir
    if data_dir == 'None':
        data_dir = None

    if args.layers == 1:
        directory = f"{home}/models/basic_ev_0.15_cnn/{data_type}retina0{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
        mei_dir = f"{home}/meis/data/{data_type}/retina{retina_index + 1}"
        filename = "lr_0.0100_l_1_ch_16_t_15_bs_16_tr_250_ik_15x15x15_g_47.0000_gt_0.0300_l1_0.0100_l2_0.0000_sg_0.15_p_0_bn_1_norm_0_fn_1"
        fn = "models.BasicEncoder.build_trained"
    else:
        directory = f"{home}/models/factorized_ev_0.15_cnn/{data_type}/retina{retina_index + 1}/cell_None/readout_isotropic/gmp_0/"
        mei_dir = f"{home}/meis/data/{data_type}/retina{retina_index + 1}"
        # filename = 'lr_0.0060_l_4_ch_[8, 16, 32, 64]_t_27_bs_32_tr_10_ik_27x(17, 17)x(17, 17)_hk_5x(3, 3)x(3, 3)_g_48.0338_gt_0.0739_l1_0.0222_l2_0.0000_sg_0.25_d_1_dt_1_hd_2-4-6_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_0_h_80_w_90'
        filename = 'lr_0.0073_l_4_ch_[8, 16, 32, 64]_t_25_bs_32_tr_250_ik_25x(21, 21)x(21, 21)_hk_5x(5, 5)x(5, 5)_g_2.0000_gt_1.4530_l1_1.3000_l2_0.0000_sg_0.25_d_1_dt_1_hd_1-2-3_hdt_1-1-1_p_0_bn_1_s_1norm_0_fn_1_h_80_w_90'
        fn = "models.FactorizedEncoder.build_trained"

    if isinstance(seed, list) or (seed is None):
        model, seed = get_seed_model_versions(
            model_name=filename,
            model_dir=directory,
            data_dir=data_dir,
            model_fn=fn,
            device="cuda",
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
                data_dir='/user/vystrcilova/',
                filename=filename,
                model_fn=fn,
                device="cuda",
                seed=seed,
                config_dict=None,
                data_type=data_type)

    for norm_scaler in [1.5, 1.75, 2, 2.25, 2.5, 3, 4, 5, 10, 20]:
        norm_value = e_norm_value*norm_scaler
        if args.mei_dir == 'None':
            mei_settings = f'{e_hash}_lr_{e_lr}_norm_{e_norm_value}_np_{nums_of_predictions}_iv_{e_init_variance}/seed_{seed}/'
        else:
            mei_settings = args.mei_dir

        post_processing_dict = {"p": 2, "norm_value": norm_value}
        pps = [(PNormConstraintAndClip, post_processing_dict)]

        sup_sur_dir = f'lr_{lr}_norm_{norm_value}_iv_{e_init_variance}_maskcut_{mask_cutoff}/'


        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD
        else:
            optimizer = torch.optim.Adam

        cell_name = get_cell_names(
            retina_index=retina_index,
            correlation_threshold=0,
            explained_variance_threshold=0.15, config=model[0].config_dict['config']
        )[cell_index]

        scheduler = optim.lr_scheduler.ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ExponentialLR

        mei = get_logged_array(
            os.path.join(mei_dir, filename, f"cell_{cell_name}", mei_settings), epoch=epoch
        )
        preconditions = [(GaussianBlur3d, dict(sigma=sigma, sigma_temp=sigma_temp))]
        postprocessing = []
        wandb.init(
            config={
                "model_name": model[0].config_dict["model_name"],
                "cell": cell_index,
                "cell_name": cell_name,
                "lr": lr,
                "optimizer": optimizer,
                "sigma": sigma,
                "sigma_temp": sigma_temp,
                "seed": seed,
                'nm': nm,
                'norm_value': norm_value,
                'mei_settings': mei_settings,
                'mask_cutoff': mask_cutoff,
                'norm_scaler': norm_scaler,
                'e_mei_norm': e_norm_value,
                'e_mei_iv': e_init_variance,
                'e_mei_num_of_predictions': e_np,
                'num_of_predictions': nums_of_predictions,
                'e_lr': e_lr,
                'e_hash': e_hash,
                'sup_sur_dir': sup_sur_dir

            },
            project=f"s-MEI-{data_type}-nm",
            entity="retinal-circuit-modeling",
            dir=log_dir
        )
        suppressive_mei = SuppressiveSurroundEnsembleMEI(
            models=model,
            cell_index=cell_index,
            exciting_mei=mei,
            input_shape=(
                1,
                1,
                get_model_temp_reach(model[0].config_dict),
                # 37,
                80, 90,
                # model[0].config_dict["img_h"],
                #  model[0].config_dict["img_w"],
            ),
            preconditions=preconditions,
            mask_cutoff=mask_cutoff,
            postprocessing=pps,
            learning_rate=lr,
            device="cuda",
            optimizer=optimizer
        )

        stopper = ActivationIncrease(initial_activation=-1, patience=1000, minimal_change=0.00001, max_steps=10000)
        tracker = SuppressiveLoggingTracker(
            log_dir=os.path.join(
                log_dir,
                "meis",
                "data",
                data_type,
                f"retina{model[0].config_dict['retina_index'] + 1}",
                model[0].config_dict["model_name"],
                f"cell_{cell_name}", mei_settings,
                "suppressive_surround", sup_sur_dir
            ),
            seed=seed,
            log_frequency=2,
            wandb_log=True,
            mei=suppressive_mei,
            activation=EvaluationObjective(1),
            state=PostProcessedInputObjective(100),
            p_grad=GradientObjective(100),
            whole_mei=WholeMeiObjective(100, mei=suppressive_mei),
            masked_surround=MaskedSurroundObjective(100, mei=suppressive_mei),
            video_log_frequency=250
        )
        optimize(suppressive_mei, stopper=stopper, tracker=tracker)
        wandb.finish()

