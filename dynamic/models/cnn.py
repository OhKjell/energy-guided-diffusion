import copy
import os
from pathlib import Path

import torch
import numpy as np
import yaml
from neuralpredictors.layers.readouts import FullGaussian2d, MultipleFullGaussian2d
# from dynamic.datasets.stas import (
#     get_rf_center_grid,
#     recalculate_positions_after_convs,
#     normalize_source_grid,
#     calculate_pca_on_cropped_stas,
# )
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from torch import nn
from torch.nn import functional as F

# from dynamic.datasets.whitenoise_salamander_loaders import get_dataloader_dims
# from dynamic.evaluations.factorized_cnn_visualization import (
#     visualized_factorized_filters,
#     visualize_temporal_kernels,
# )
# from dynamic.models.model_visualizations import (
#     visualize_all_gaussian_readout,
#     visualize_filters,
# )
from dynamic.models.readouts import (
    initialize_full_gaussian_readout,
    initialize_multiple_full_gaussian_readouts,
)
from dynamic.utils.global_functions import (
    set_random_seed,
    #get_exclude_cells_based_on_thresholds,
    home,
    model_seed,
    #global_config,
    get_exclude_cells,
)

from models.cores import Basic3dCore, Factorized3dCore


class Encoder(nn.Module):
    def __init__(self, core, readout, readout_nonlinearity, elu_xshift, elu_yshift, l1):
        super().__init__()
        self.core = core
        self.readout = readout
        self.l1 = l1
        if readout_nonlinearity == "adaptive_elu":
            self.nonlinearity = core.nonlinearities[readout_nonlinearity](
                elu_xshift, elu_yshift
            )
        else:
            self.nonlinearity = core.nonlinearities[readout_nonlinearity]()
        self.visualization_dir = None

    def forward(self, x, data_key=None):
        out_core = self.core(x)
        out_core = torch.transpose(out_core, 1, 2)
        # print('outcore shape', out_core.shape)
        out_core = out_core.reshape(((-1,) + out_core.size()[2:]))
        # print('outcore shape', out_core.shape)
        # out_core = out_core.squeeze()
        if data_key is None:
            readout_out = self.readout(out_core)
        else:
            readout_out = self.readout[data_key](out_core)
        out = self.nonlinearity(readout_out)
        return out

    @staticmethod
    def build_trained(dataloaders, model_dir, model_name, data_dir, device="cpu"):
        # get parameters from config file
        with open(f"{model_dir}/{model_name}/config/config.yaml", "r") as config_file:
            config = yaml.unsafe_load(config_file)
            model_config = config["model_config"]
            if 'config' in config.keys():
                for key in dataloaders['train'].keys():
                    config['config']['big_crops'][key] = dataloaders['train'][key].dataset.crop
            # config['size'] = (1, 150 - sum(config['dataloader_config']['crop'][:2]), 200-sum(config['dataloader_config']['crop'][2:]))
            # config['img_h'] = 150 - sum(config['dataloader_config']['crop'][:2])
            # config['img_w'] = 200 - sum(config['dataloader_config']['crop'][2:])
            config['size'] = (1, 80, 90)
            config['img_h'] = 80
            config['img_w'] = 90
            if "config" not in model_config.keys():
                model_config["config"] = global_config
            if "spatial_dilation" not in model_config.keys():
                model_config["spatial_dilation"] = 1
                model_config["temporal_dilation"] = 1
            if "hidden_spatial_dilation" not in model_config.keys():
                model_config["hidden_spatial_dilation"] = 1
                model_config["hidden_temporal_dilation"] = 1
            data_config = config["dataloader_config"]
            n_neurons_dict, in_shapes_dict, input_channels = get_dataloader_dims(
                dataloaders
            )
            del config["model_config"]
            del config["dataloader_config"]

            config_dict = {
                **model_config,
                **data_config,
                **config,
                "base_path": data_dir,
                "initialize_source_grid": True,
                "n_neurons_dict": n_neurons_dict,
                "in_shapes_dict": in_shapes_dict,
                "input_channels": input_channels[0],
                "model_dir": model_dir,
                "model_name": model_name,
            }
        return config_dict

    def get_max_corr(self):
        correlation = np.load(
            os.path.join(
                home,
                self.config_dict["model_dir"],
                self.config_dict["model_name"],
                "stats",
                "correlations.npy",
            )
        )
        best_epoch = np.argmax(correlation)
        max_corr = max(correlation)
        return best_epoch, max_corr

    def visualize_readout(self):
        self.visualization_dir = os.path.join(
            self.config_dict["model_dir"],
            self.config_dict["model_name"],
            "visualizations",
        )

        visualize_all_gaussian_readout(
            self,
            self.visualization_dir,
            readout_index=self.config_dict["retina_index"] + 1,
            retina_index=self.config_dict["retina_index"],
            correlation_threshold=self.config_dict["oracle_correlation_threshold"],
            explainable_variance_threshold=self.config_dict[
                "explainable_variance_threshold"
            ],
        )

    def regularizer(self, data_key=None):
        if data_key is None:
            return self.core.regularizer() + (
                self.l1 * self.readout.feature_l1(average=False),
            )
        else:
            return self.core.regularizer() + (
                self.l1 * self.readout[data_key].feature_l1(average=False),
            )  # +l1*torch.norm(self.readout.spatial, 1) + l1*torch.norm(self.readout.features, 1)


class BasicEncoder(Encoder):
    def __init__(
        self, config_dict: dict, dataloaders: dict, trained: bool, seed: None
    ) -> None:
        config_dict["padding"] = 0
        core_dict = dict(
            input_channels=config_dict["input_channels"],
            hidden_channels=config_dict["hidden_channels"],
            input_kernel=config_dict["input_kern"],
            hidden_kernel=config_dict["hidden_kern"],
            stride=config_dict["stride"],
            layers=config_dict["layers"],
            gamma_input_spatial=config_dict["gamma_input"],
            gamma_input_temporal=config_dict["gamma_temporal"],
            bias=config_dict["bias"],
            hidden_nonlinearities=config_dict["core_nonlinearity"],
            spatial_dilation=config_dict["spatial_dilation"],
            temporal_dilation=config_dict["temporal_dilation"],
            x_shift=config_dict["elu_xshift"],
            y_shift=config_dict["elu_yshift"],
            batch_norm=config_dict["batch_norm"],
            laplace_padding=config_dict["laplace_padding"],
            input_regularizer=config_dict["input_regularizer"],
            padding=config_dict["padding"],
            final_nonlin=config_dict["final_nonlinearity"],
            independent_bn_bias=config_dict["independent_bn_bias"],
        )

        readout_dict = dict(
            retina_index=config_dict["retina_index"],
            data_dir=config_dict["base_path"],
            oracle_correlation_threshold=config_dict["oracle_correlation_threshold"],
            explainable_variance_threshold=config_dict[
                "explainable_variance_threshold"
            ],
            readout_bias=config_dict["readout_bias"],
            in_shapes_dict=config_dict["in_shapes_dict"],
            n_neurons_dict=config_dict["n_neurons_dict"],
            init_mu_range=config_dict["init_mu_range"],
            init_sigma=config_dict["init_sigma"],
            use_grid_mean_predictor=config_dict["use_grid_mean_predictor"],
            initialize_source_grid=config_dict["initialize_source_grid"],
            dataloaders=dataloaders,
            readout_type=config_dict["readout_type"],
            config=config_dict["config"],
            img_h=config_dict["img_h"],
            img_w=config_dict["img_w"],
            subsample=config_dict["subsample"],
            cell_index=config_dict["cell_index"],
        )
        core = Basic3dCore(**core_dict)
        readout = initialize_full_gaussian_readout(**readout_dict, core=core)
        if seed is None:
            self.seed = model_seed
        else:
            self.seed = seed
        self.trained = trained
        self.config_dict = config_dict
        super().__init__(
            core,
            readout,
            config_dict["readout_nonlinearity"],
            config_dict["elu_xshift"],
            config_dict["elu_yshift"],
            config_dict["l1"],
        )

    @staticmethod
    def build_trained(
        dataloaders, model_dir, model_name, data_dir, device="cpu", seed=None
    ):
        # get parameters from config file
        config_dict = Encoder.build_trained(
            dataloaders=dataloaders,
            model_dir=model_dir,
            model_name=model_name,
            device=device,
            data_dir=data_dir,
        )
        model = BasicEncoder(
            config_dict=config_dict, dataloaders=dataloaders, trained=True, seed=seed
        )
        state_dict = torch.load(
            f"{model_dir}/{model_name}/weights/seed_{model.seed}/best_model.m",
            map_location=torch.device(device),
        )["model"]
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    @staticmethod
    def build_initial(dataloaders, config_dict, seed=None):
        n_neurons_dict, in_shapes_dict, input_channels = get_dataloader_dims(
            dataloaders
        )
        config_dict = {
            **config_dict,
            "n_neurons_dict": n_neurons_dict,
            "in_shapes_dict": in_shapes_dict,
            "input_channels": input_channels[0],
        }
        set_random_seed(seed)
        model = BasicEncoder(
            config_dict=config_dict, dataloaders=dataloaders, trained=False, seed=seed
        )
        return model

    def visualize(self, v_min, v_max):
        # TODO: implement in parent
        visualization_dir = f"{self.config_dict['model_dir']}/{self.config_dict['model_name']}/visualizations/seed_{self.seed}/"
        Path(visualization_dir).mkdir(exist_ok=True, parents=True)
        self.visualize_readout()
        max_corr, best_epoch = self.get_max_corr()
        for layer in range(self.config_dict["layers"]):
            visualize_filters(
                self,
                layer,
                visualization_dir,
                max_corr=max_corr,
                max_corr_epoch=best_epoch,
                v_max=v_max,
                v_min=v_min,
            )


class FactorizedEncoder(Encoder):
    def __init__(
        self, config_dict: dict, dataloaders: dict, seed: None, trained: bool
    ) -> None:
        config_dict["padding"] = 0
        core_dict = dict(
            input_channels=config_dict["input_channels"],
            hidden_channels=config_dict["hidden_channels"],
            spatial_input_kernel=config_dict["spatial_input_kern"],
            spatial_hidden_kernel=config_dict["spatial_hidden_kern"],
            temporal_input_kernel=config_dict["temporal_input_kern"],
            temporal_hidden_kernel=config_dict["temporal_hidden_kern"],
            stride=config_dict["stride"],
            layers=config_dict["layers"],
            gamma_input_spatial=config_dict["gamma_input"],
            gamma_input_temporal=config_dict["gamma_temporal"],
            spatial_dilation=config_dict["spatial_dilation"],
            temporal_dilation=config_dict["temporal_dilation"],
            hidden_spatial_dilation=config_dict["hidden_spatial_dilation"],
            hidden_temporal_dilation=config_dict["hidden_temporal_dilation"],
            bias=config_dict["bias"],
            hidden_nonlinearities=config_dict["core_nonlinearity"],
            x_shift=config_dict["elu_xshift"],
            y_shift=config_dict["elu_yshift"],
            batch_norm=config_dict["batch_norm"],
            laplace_padding=config_dict["laplace_padding"],
            input_regularizer=config_dict["input_regularizer"],
            padding=config_dict["padding"],
            final_nonlin=config_dict["final_nonlinearity"],
            independent_bn_bias=config_dict["independent_bn_bias"],
        )

        readout_dict = dict(
            retina_index=config_dict["retina_index"],
            data_dir=config_dict["base_path"],
            oracle_correlation_threshold=config_dict["oracle_correlation_threshold"],
            explainable_variance_threshold=config_dict[
                "explainable_variance_threshold"
            ],
            readout_bias=config_dict["readout_bias"],
            in_shapes_dict=config_dict["in_shapes_dict"],
            n_neurons_dict=config_dict["n_neurons_dict"],
            init_mu_range=config_dict["init_mu_range"],
            init_sigma=config_dict["init_sigma"],
            use_grid_mean_predictor=config_dict["use_grid_mean_predictor"],
            initialize_source_grid=config_dict["initialize_source_grid"],
            dataloaders=dataloaders,
            readout_type=config_dict["readout_type"],
            config=config_dict["config"],
            img_h=config_dict["img_h"],
            img_w=config_dict["img_w"],
            subsample=config_dict["subsample"],
        )
        core = Factorized3dCore(**core_dict)
        self.config_dict = config_dict
        if seed is None:
            self.seed = model_seed
        else:
            self.seed = seed
        self.trained = trained
        readout = initialize_full_gaussian_readout(**readout_dict, core=core)
        super().__init__(
            core,
            readout,
            config_dict["readout_nonlinearity"],
            config_dict["elu_xshift"],
            config_dict["elu_yshift"],
            config_dict["l1"],
        )

    @staticmethod
    def build_trained(
        dataloaders, model_dir, model_name, data_dir, seed=None, device="cpu"
    ):
        # get parameters from config file
        config_dict = Encoder.build_trained(
            dataloaders=dataloaders,
            model_dir=model_dir,
            model_name=model_name,
            device=device,
            data_dir=data_dir,
        )
        model = FactorizedEncoder(
            config_dict=config_dict, dataloaders=dataloaders, seed=seed, trained=True
        )
        print(seed)
        state_dict = torch.load(
            f"{model_dir}/{model_name}/weights/seed_{model.seed}/best_model.m",
            map_location=torch.device(device),
        )["model"]
        # if 'readout._mu' in state_dict.keys():
        #     state_dict['readout.mu'] = state_dict['readout._mu']
        #     state_dict['readout.features'] = state_dict['readout._features']
        #     del state_dict['readout._features']
        #     del state_dict['readout._mu']

        model.load_state_dict(state_dict)
        model.config_dict = config_dict
        model.to(device)
        return model

    @staticmethod
    def build_initial(dataloaders, config_dict, seed=None):
        n_neurons_dict, in_shapes_dict, input_channels = get_dataloader_dims(
            dataloaders
        )

        config_dict = {
            **config_dict,
            "n_neurons_dict": n_neurons_dict,
            "in_shapes_dict": in_shapes_dict,
            "input_channels": input_channels[0],
        }
        set_random_seed(seed)
        model = FactorizedEncoder(
            config_dict=config_dict, dataloaders=dataloaders, seed=seed, trained=False
        )
        model.config_dict = config_dict
        return model

    def visualize(self):
        visualization_dir = f"{self.config_dict['model_dir']}/{self.config_dict['model_name']}/visualizations/seed_{self.seed}/"
        Path(visualization_dir).mkdir(exist_ok=True, parents=True)
        self.visualize_readout()
        max_corr, best_epoch = self.get_max_corr()
        visualize_all_gaussian_readout(
            self,
            visualization_dir,
            readout_index=self.config_dict["retina_index"] + 1,
            retina_index=self.config_dict["retina_index"],
            correlation_threshold=self.config_dict["oracle_correlation_threshold"],
            explainable_variance_threshold=self.config_dict[
                "explainable_variance_threshold"
            ],
        )

        for layer in range(self.config_dict["layers"]):
            visualized_factorized_filters(
                self,
                layer,
                visualization_dir,
                max_corr=max_corr,
                max_corr_epoch=best_epoch,
            )
            visualize_temporal_kernels(self, visualization_dir, layer=layer)


class MultiRetinalFactorizedEncoder(Encoder):
    def __init__(
        self, config_dict: dict, dataloaders: dict, seed: None, trained: bool
    ) -> None:
        config_dict["padding"] = 0
        core_dict = dict(
            input_channels=config_dict["input_channels"],
            hidden_channels=config_dict["hidden_channels"],
            spatial_input_kernel=config_dict["spatial_input_kern"],
            spatial_hidden_kernel=config_dict["spatial_hidden_kern"],
            temporal_input_kernel=config_dict["temporal_input_kern"],
            temporal_hidden_kernel=config_dict["temporal_hidden_kern"],
            stride=config_dict["stride"],
            layers=config_dict["layers"],
            gamma_input_spatial=config_dict["gamma_input"],
            gamma_input_temporal=config_dict["gamma_temporal"],
            spatial_dilation=config_dict["spatial_dilation"],
            temporal_dilation=config_dict["temporal_dilation"],
            hidden_spatial_dilation=config_dict["hidden_spatial_dilation"],
            hidden_temporal_dilation=config_dict["hidden_temporal_dilation"],
            bias=config_dict["bias"],
            hidden_nonlinearities=config_dict["core_nonlinearity"],
            x_shift=config_dict["elu_xshift"],
            y_shift=config_dict["elu_yshift"],
            batch_norm=config_dict["batch_norm"],
            laplace_padding=config_dict["laplace_padding"],
            input_regularizer=config_dict["input_regularizer"],
            padding=config_dict["padding"],
            final_nonlin=config_dict["final_nonlinearity"],
            independent_bn_bias=config_dict["independent_bn_bias"],
        )

        readout_dict = dict(
            data_keys=dataloaders["train"].keys(),
            data_dir=config_dict["base_path"],
            oracle_correlation_threshold=config_dict["oracle_correlation_threshold"],
            explainable_variance_threshold=config_dict[
                "explainable_variance_threshold"
            ],
            readout_bias=config_dict["readout_bias"],
            in_shapes_dict=config_dict["in_shapes_dict"],
            n_neurons_dict=config_dict["n_neurons_dict"],
            init_mu_range=config_dict["init_mu_range"],
            init_sigma=config_dict["init_sigma"],
            use_grid_mean_predictor=config_dict["use_grid_mean_predictor"],
            initialize_source_grid=config_dict["initialize_source_grid"],
            dataloaders=dataloaders,
            readout_type=config_dict["readout_type"],
            config=config_dict["config"],
            img_h=config_dict["img_h"],
            img_w=config_dict["img_w"],
            subsample=config_dict["subsample"],
        )

        core = Factorized3dCore(**core_dict)
        self.config_dict = config_dict
        if seed is None:
            self.seed = model_seed
        else:
            self.seed = seed
        self.trained = trained

        readout = initialize_multiple_full_gaussian_readouts(**readout_dict, core=core)
        super().__init__(
            core,
            readout,
            config_dict["readout_nonlinearity"],
            config_dict["elu_xshift"],
            config_dict["elu_yshift"],
            config_dict["l1"],
        )

    @staticmethod
    def build_trained(
        dataloaders, model_dir, model_name, data_dir, seed=None, device="cpu"
    ):
        # get parameters from config file
        config_dict = Encoder.build_trained(
            dataloaders=dataloaders,
            model_dir=model_dir,
            model_name=model_name,
            device=device,
            data_dir=data_dir,
        )
        model = MultiRetinalFactorizedEncoder(
            config_dict=config_dict, dataloaders=dataloaders, seed=seed, trained=True
        )
        print(seed)
        state_dict = torch.load(
            f"{model_dir}/{model_name}/weights/seed_{model.seed}/best_model.m",
            map_location=torch.device(device),
        )["model"]
        model.load_state_dict(state_dict)
        model.config_dict = config_dict
        model.to(device)
        return model

    @staticmethod
    def build_initial(dataloaders, config_dict, seed=None):
        n_neurons_dict, in_shapes_dict, input_channels = get_dataloader_dims(
            dataloaders
        )

        config_dict = {
            **config_dict,
            "n_neurons_dict": n_neurons_dict,
            "in_shapes_dict": in_shapes_dict,
            "input_channels": input_channels[0],
        }
        set_random_seed(seed)
        model = MultiRetinalFactorizedEncoder(
            config_dict=config_dict, dataloaders=dataloaders, seed=seed, trained=False
        )
        model.config_dict = config_dict
        return model


def sta_model(
    dataloaders,
    hidden_channels,
    input_kern,
    hidden_kern,
    seed=None,
    l1=0.01,
    gamma_input=0.0,
    readout="fc",
    core_nonlinearity="elu",
    final_nonlinearity="relu",
    elu_xshift=0.0,
    elu_yshift=1,
    bias=True,
    batch_norm=False,
    readout_bias=False,
    momentum=0.9,
    padding=True,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_mu_range=0.3,
    init_sigma=0.35,
    retina_index=0,
    data_dir=None,
    use_grid_mean_predictor=False,
    initialize_source_grid=True,
    train_core=False,
    subsample=1,
    train_readout=True,
    correlation_threshold=None,
    explainable_variance_threshold=None,
    cell_index=None,
    nonlin=True,
    config=None,
):
    """
    A model consisting of a 1 layer Basic3dCore in ./models and FullGaussianReadout neuralpredictors.layers.readouts.
    The convolutional filters of the core are initialized as the first hidden_channels principal components when PCA is
    done on the STAs of recorded neurons. The readout source_grid is initialized at the x,y pixel position with the
    highest variance for each of the neurons and the feature vectors are initialized as projections (called loadings but
    are not really loadings).

    :param dataloaders:
    :param hidden_channels: Number of channels in the convolutional layer. Corresponds to the number of principal components that should be used.
    :param seed: random seed
    :param retina_index:
    :param data_dir:
    :param train_core: bool, specifies, whether core should be trained after initial prediction
    :param train_readout: bool, specifies whether the readout should be trained after initial prediction
    :param correlation_threshold:
    :param explainable_variance_threshold:
    :param cell_index: Specifies for which cell the model is built. If None, all cells in dataloader are considered
    :param nonlin: bool, specifies whether to use a non-linearity after readout output
    :return:
    """
    if config is None:
        config = global_config
    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]
    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    n_neurons_dict, in_shapes_dict, input_channels = get_dataloader_dims(dataloaders)
    # n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    # in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    # input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    set_random_seed(seed)

    class Encoder(nn.Module):
        def __init__(self, core, readout, mean):
            super().__init__()
            self.core = core
            self.readout = readout
            self.mean = mean
            if self.mean is not None:
                self.conv = nn.Conv3d(
                    1, 1, kernel_size=self.core.input_kernel, bias=False
                )
                self.conv.weight.data = torch.tensor(
                    self.mean.reshape((1, 1) + self.core.input_kernel)
                )
                self.conv.requires_grad_(False)
            if final_nonlinearity == "adaptive_elu":
                self.nonlinearity = core.nonlinearities[final_nonlinearity](
                    elu_xshift, elu_yshift
                )
            else:
                self.nonlinearity = core.nonlinearities[final_nonlinearity]()

        def forward(self, x, data_key=None):
            out_core = self.core(x)
            out_core = torch.transpose(out_core, 1, 2)
            out_core = out_core.reshape(((-1,) + out_core.size()[2:]))

            readout_out = self.readout(out_core)
            mean = self.conv(x).detach().cpu()
            out = readout_out + torch.tensor(
                [
                    mean[
                        :,
                        0,
                        :,
                        self.readout.mu_not_normalized[i, 1].astype(int),
                        self.readout.mu_not_normalized[i, 0].astype(int),
                    ]
                    .numpy()
                    .flatten()
                    for i in range(self.readout.mu_not_normalized.shape[0])
                ]
            ).T.to("cuda")

            if nonlin:
                out = self.nonlinearity(out)
            return out

        def regularizer(self, data_key=None):
            return self.core.regularizer() + (l1 * readout.feature_l1(average=False),)

    true_rf_fields, loadings, components = calculate_pca_on_cropped_stas(
        input_kern[0],
        retina_index=retina_index,
        data_dir=data_dir,
        plot=True,
        rf_size=(input_kern[1], input_kern[1]),
    )
    loadings = loadings.T
    if cell_index is None:
        excluded_cells = list(
            set(
                get_exclude_cells_based_on_thresholds(
                    retina_index=retina_index,
                    config=config,
                    explainable_variance_threshold=explainable_variance_threshold,
                    correlation_threshold=correlation_threshold,
                )
                + get_exclude_cells(config=config, retina_index=retina_index)
            )
        )
        loadings = np.delete(loadings, excluded_cells, axis=1)
    else:
        loadings = loadings[:, cell_index].reshape((-1, 1))

    loadings = torch.tensor(loadings).unsqueeze(0).unsqueeze(2)

    core = Basic3dCore(
        input_channels=input_channels[0],
        hidden_channels=hidden_channels,
        input_kernel=input_kern,
        hidden_kernel=hidden_kern,
        layers=1,
        gamma_input_spatial=gamma_input,
        bias=bias,
        hidden_nonlinearities=core_nonlinearity,
        batch_norm=False,
        batch_norm_scale=False,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        padding=padding,
        momentum=momentum,
        final_nonlin=False,
    )

    for filter_index in range(core.features[0].conv.weight.shape[0]):
        core.features[0].conv.weight.data[
            filter_index, :, :, :, :
        ] = torch.nn.Parameter(
            torch.tensor(
                components.components_[filter_index].reshape(
                    core.features[0].conv.weight.shape[1:]
                )
            ).double()
        )

    source_grid = get_rf_center_grid(
        retina_index=retina_index,
        crop=config["big_crops"][f"0{retina_index + 1}"],
        data_dir=data_dir,
        correlation_threshold=correlation_threshold,
        explainable_varinace_threshold=explainable_variance_threshold,
        config=config,
        subsample=subsample,
    )

    readout = FullGaussian2d(
        in_shape=(
            (core.hidden_channels[-1],)
            + core.get_output_shape(list(in_shapes_dict.values())[0])[2:]
        ),
        outdims=list(n_neurons_dict.values())[0],
        bias=readout_bias,
        batch_sample=True,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma,
        grid_mean_predictor={
            "hidden_layers": 0,
            "hidden_features": 30,
            "final_tanh": False,
        }
        if use_grid_mean_predictor
        else None,
        source_grid=source_grid if use_grid_mean_predictor else None,
        gauss_type="isotropic",
    )

    if not use_grid_mean_predictor and initialize_source_grid:
        source_grid = recalculate_positions_after_convs(
            source_grid,
            core.get_kernels(),
            img_h=150 - sum(config["big_crops"][f"0{retina_index + 1}"][:2]),
            img_w=200 - sum(config["big_crops"][f"0{retina_index + 1}"][2:]),
        )
        mu_not_normalized = source_grid.copy()
        readout.mu_not_normalized = source_grid.copy()
        source_grid = normalize_source_grid(
            source_grid, core.get_output_shape(list(in_shapes_dict.values())[0])[2:]
        )

        for i in range(source_grid.shape[0]):
            readout.mu_not_normalized[i] = (
                mu_not_normalized[i, 1],
                mu_not_normalized[i, 0],
            )
            source_grid[i] = source_grid[i][1], source_grid[i][0]
        source_grid = torch.unsqueeze(torch.Tensor(source_grid), dim=0)
        source_grid = torch.unsqueeze(source_grid, dim=2)

        if cell_index is not None:
            readout._mu.data = torch.nn.Parameter(
                source_grid[0, cell_index].unsqueeze(0)
            )
            readout.mu_not_normalized = readout.mu_not_normalized[
                cell_index : cell_index + 1
            ]
        else:
            readout._mu.data = torch.nn.Parameter(source_grid)

        excluded_cells = list(
            set(
                get_exclude_cells_based_on_thresholds(
                    retina_index=retina_index,
                    config=config,
                    explainable_variance_threshold=explainable_variance_threshold,
                    correlation_threshold=correlation_threshold,
                )
                + get_exclude_cells(config=config, retina_index=retina_index)
            )
        )
        # loadings = np.delete(loadings, excluded_cells, axis=-1)
        readout._features = torch.nn.Parameter(
            loadings[:, : core.out_channels, :, :].contiguous()
        ).double()

    if readout_bias:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout.bias.data = torch.tensor([0.0])

    model = Encoder(core, readout, mean=components.mean_)
    for name, param in model.named_parameters():
        print(name, param.shape)
        if "core" in name:
            param.requires_grad = train_core
        if "readout" in name:
            param.requires_grad = train_readout

    return model

