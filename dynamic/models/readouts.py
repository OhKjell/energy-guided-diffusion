import torch
from neuralpredictors.layers.readouts import FullGaussian2d
from torch.nn import ModuleDict

from datasets.stas import (
    get_rf_center_grid,
    recalculate_positions_after_convs,
    normalize_source_grid,
)
from utils.global_functions import global_config


class InitializedMultipleFullGaussian2d(ModuleDict):
    def __init__(
        self,
        data_keys,
        data_dir,
        oracle_correlation_threshold,
        explainable_variance_threshold,
        core,
        in_shapes_dict,
        n_neurons_dict,
        readout_bias,
        init_mu_range,
        init_sigma,
        use_grid_mean_predictor,
        initialize_source_grid,
        dataloaders,
        readout_type="isotropic",
        config=None,
        cell_index=None,
        img_h=150,
        img_w=200,
        subsample=1,
    ):
        super().__init__()
        self.data_keys = data_keys
        for i, data_key in enumerate(data_keys):
            readout = initialize_full_gaussian_readout(
                retina_index=int(data_key) - 1,
                data_dir=data_dir,
                oracle_correlation_threshold=oracle_correlation_threshold,
                explainable_variance_threshold=explainable_variance_threshold,
                core=core,
                in_shapes_dict=in_shapes_dict,
                n_neurons_dict=n_neurons_dict,
                readout_bias=readout_bias,
                init_sigma=init_sigma,
                init_mu_range=init_mu_range,
                use_grid_mean_predictor=use_grid_mean_predictor,
                initialize_source_grid=initialize_source_grid,
                dataloaders=dataloaders,
                readout_type=readout_type,
                config=config,
                cell_index=cell_index,
                img_h=img_h,
                img_w=img_w,
                subsample=subsample,
            )
            self.add_module(data_key, readout)


def initialize_full_gaussian_readout(
    retina_index,
    data_dir,
    oracle_correlation_threshold,
    explainable_variance_threshold,
    core,
    in_shapes_dict,
    n_neurons_dict,
    readout_bias,
    init_mu_range,
    init_sigma,
    use_grid_mean_predictor,
    initialize_source_grid,
    dataloaders,
    readout_type="isotropic",
    config=None,
    cell_index=None,
    img_h=150,
    img_w=200,
    subsample=1,
):
    """

    :param retina_index: specifies which retina (data_key) corresponds to the given readout
    :param data_dir:
    :param oracle_correlation_threshold: if not None or 0, selects cells with higher oracle correlation then specified thresholds
    :param explainable_variance_threshold: if not None or 0, selects cells with higher explainable variance then specified thresholds
                                           TODO: for 0 actually this is not true as the explainable variance threshold can be negative
    :param core: The core object used with the readout
    :param in_shapes_dict: dict containing 'data_key': input shape entries
    :param n_neurons_dict: dict containing 'data_key': n_neurons entries
    :param readout_bias: whether readout should use bias
    :param init_mu_range: the mean value for the distribution deciding how source_grid values are initialized
    :param init_sigma: the std value for the distribution deciding how source_grid values are initialized
    :param use_grid_mean_predictor: whether the grid predictor should be a fully connected network
    :param initialize_source_grid: whether to
    :param dataloaders: a dictionary of dataloaders in the format of {"data_key": dataloader, ...}
    :param readout_type: what type of convariance matrix the readou should use
    :return: Initialized readout with source grid locations at the pixel of biggest variance of STA of a given cell

    when the use_grid_mean_predictor is set to false and initialize_source_grid is true, the mu and sigma values are
    not considered and the source_grid is initialized with values based on STAs of recorded cells
    """
    if config is None:
        config = global_config

    source_grid = get_rf_center_grid(
        retina_index=retina_index,
        crop=config["big_crops"][f"0{retina_index + 1}"],
        data_dir=data_dir,
        correlation_threshold=oracle_correlation_threshold,
        explainable_varinace_threshold=explainable_variance_threshold,
        config=config,
        subsample=subsample,
    )
    readout = FullGaussian2d(
        in_shape=(
            (core.hidden_channels[-1],)
            + core.get_output_shape(list(in_shapes_dict.values())[0])[2:]
        ),
        outdims=list(n_neurons_dict.values())[retina_index]
        if len(list(n_neurons_dict.values())) > 1
        else list(n_neurons_dict.values())[0],
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
        source_grid=source_grid,
        gauss_type=readout_type,
    )

    if not use_grid_mean_predictor and initialize_source_grid:
        source_grid = get_rf_center_grid(
            retina_index=retina_index,
            crop=config["big_crops"][f"0{retina_index + 1}"],
            data_dir=data_dir,
            correlation_threshold=oracle_correlation_threshold,
            explainable_varinace_threshold=explainable_variance_threshold,
            config=config,
            subsample=subsample,
        )

        # initializing readout locations with sta pixel of the biggest variance
        source_grid = recalculate_positions_after_convs(
            source_grid,
            core.get_kernels(),
            img_h=img_h / subsample
            - sum(config["big_crops"][f"0{retina_index + 1}"][:2]),
            img_w=img_w / subsample
            - sum(config["big_crops"][f"0{retina_index + 1}"][2:]),
        )
        source_grid = normalize_source_grid(
            source_grid, core.get_output_shape(list(in_shapes_dict.values())[0])[2:]
        )
        if cell_index is not None:
            cell_index = int(cell_index)
            source_grid = source_grid[cell_index : cell_index + 1, :]
        for i in range(source_grid.shape[0]):
            source_grid[i] = source_grid[i][1], source_grid[i][0]
        source_grid = torch.unsqueeze(torch.Tensor(source_grid), dim=0)
        source_grid = torch.unsqueeze(source_grid, dim=2)
        readout._mu.data = torch.nn.Parameter(source_grid)

    if readout_bias:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout.bias.data = targets.mean(0)
    return readout


def initialize_multiple_full_gaussian_readouts(
    data_keys,
    data_dir,
    oracle_correlation_threshold,
    explainable_variance_threshold,
    core,
    in_shapes_dict,
    n_neurons_dict,
    readout_bias,
    init_mu_range,
    init_sigma,
    use_grid_mean_predictor,
    initialize_source_grid,
    dataloaders,
    readout_type="isotropic",
    config=None,
    cell_index=None,
    img_h=150,
    img_w=200,
    subsample=1,
):
    readout = InitializedMultipleFullGaussian2d(
        data_keys=data_keys,
        data_dir=data_dir,
        oracle_correlation_threshold=oracle_correlation_threshold,
        explainable_variance_threshold=explainable_variance_threshold,
        core=core,
        in_shapes_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        readout_bias=readout_bias,
        init_sigma=init_sigma,
        init_mu_range=init_mu_range,
        use_grid_mean_predictor=use_grid_mean_predictor,
        initialize_source_grid=initialize_source_grid,
        dataloaders=dataloaders,
        readout_type=readout_type,
        config=config,
        cell_index=cell_index,
        img_h=img_h,
        img_w=img_w,
        subsample=subsample,
    )
    return readout
