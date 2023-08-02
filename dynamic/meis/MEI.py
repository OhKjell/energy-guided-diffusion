import os
from pathlib import Path

import numpy as np
import torch
import wandb
from dynamic.mei.optimization import MEI
from numpy.linalg import cholesky, inv
from scipy.stats import multivariate_normal
from torch import nn

from dynamic.models.helper_functions import get_center_coordinates
from dynamic.meis.postprocessing import CenteredSigmoid
from dynamic.meis.preconditions import (GaussianBlur3d, MultipleConditions,
                                SuppressGradient)
from dynamic.meis.readouts import SingleCellMEIEncoder
from dynamic.utils.global_functions import get_cell_names, home


class ExcitingMEI(MEI):
    def __init__(
        self,
        model,
        cell_index,
        input_shape,
        learning_rate,
        device,
        optimizer,
        scheduler,
        precondition: list = None,
        postprocessing: list = None,
        seed=None,
        initial_input=None,
        num_of_predictions: int = 1,
        init_variance=0.1,
        contrast=0.1,
    ):
        self.model = model.to(device)
        self.seed = seed
        self.input_shape = input_shape
        if num_of_predictions > 1:
            self.input_shape = (
                (self.input_shape[:-3])
                + (self.input_shape[-3] + num_of_predictions - 1,)
                + self.input_shape[-2:]
            )
        # if isinstance(self.model, SingleCellMEIEncoder):
        #     self.cell_index = 0
        # else:
        self.cell_index = cell_index
        self.center_location, self.size = self.get_center_coordinates_and_size()
        self.size = int(self.size)
        self.contrast = contrast
        self.cell_name = get_cell_names(
            retina_index=model.config_dict["retina_index"],
            correlation_threshold=model.config_dict["oracle_correlation_threshold"],
            explained_variance_threshold=model.config_dict[
                "explainable_variance_threshold"
            ], config=model.config_dict['config']
        )[cell_index]
        self.device = device
        self.init_variance = init_variance
        initial_state = self.initialize(initial_input=initial_input).to(device)
        optimizer = optimizer(params=[initial_state], lr=learning_rate)
        self.scheduler = scheduler(optimizer=optimizer, gamma=0.75, verbose=True, )
        initial_state.requires_grad_(True)

        self.precondition = MultipleConditions(precondition)
        self.postprocessing = MultipleConditions(postprocessing)

        super().__init__(
            initial=initial_state,
            func=self.objective,
            optimizer=optimizer,
            transform=self.transform,
            regularization=self.regularize,
            precondition=self.precondition,
            postprocessing=self.postprocessing,
        )

    def objective(self, input):
        return torch.mean(self.model(input)[:, self.cell_index])


    def regularize(self, input, iteration):
        return torch.tensor([0]).to(self.device)

    def transform(self, input, iteration):
        return input

    def initialize(self, initial_input=None):
        torch.manual_seed(self.seed)
        if initial_input is None:
            dist = torch.distributions.Normal(0, self.init_variance)
            # mei_area = dist.sample((self.input_shape[2], self.size, self.size))
            input = dist.sample(self.input_shape)
            # mei_area = torch.nn.functional.normalize(mei_area, dim=(0, 1, 2), p=2)
            # input = torch.zeros(*self.input_shape)
            # input[
            #     :,
            #     :,
            #     :,
            #     self.center_location[1]
            #     - int(self.size / 2) : self.center_location[1]
            #     + int(self.size / 2)
            #     + 1,
            #     self.center_location[0]
            #     - int(self.size / 2) : self.center_location[0]
            #     + int(self.size / 2)
            #     + 1,
            # ] = mei_area
        else:
            input = initial_input
        return input.double()

    def freeze_model(self):
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(False)

    def get_center_coordinates_and_size(self):
        return get_center_coordinates(self.model, self.cell_index)


class ExcitingEnsembleMEI(ExcitingMEI):
    def __init__(
        self,
        models,
        cell_index,
        input_shape,
        optimizer,
        scheduler,
        preconditions: list,
        postprocessing: list,
        learning_rate,
        device,
        seed=None,
        initial_input=None,
        num_of_predictions=1,
        init_variance=0.1,
        contrast=0.1,
    ):
        super().__init__(
            model=models[0],
            cell_index=cell_index,
            input_shape=input_shape,
            learning_rate=learning_rate,
            device=device,
            precondition=preconditions,
            postprocessing=postprocessing,
            seed=seed,
            initial_input=initial_input,
            optimizer=optimizer,
            scheduler=scheduler,
            num_of_predictions=num_of_predictions,
            init_variance=init_variance,
            contrast=contrast,
        )
        self.model = [model.to(device) for model in models]

    def objective(self, input):
        activation = None
        for model in self.model:
            if activation is None:
                activation = torch.mean(model(input)[:, self.cell_index])
            else:
                activation = activation + torch.mean(model(input)[:, self.cell_index])

        return torch.div(activation, len(self.model))

    def freeze_model(self):
        for model in self.model:
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(False)


class SuppressiveSurroundMEI(MEI):
    def __init__(
        self,
        model,
        cell_index,
        exciting_mei,
        preconditions,
        postprocessing,
        learning_rate,
        device,
        optimizer,
        input_shape,
        mask_cutoff
    ):
        self.model = model
        self.input_shape = input_shape
        if isinstance(self.model, SingleCellMEIEncoder):
            self.cell_index = 0
        else:
            self.cell_index = cell_index
        self.surround = self.initialize()
        self.cell_name = get_cell_names(
            retina_index=self.model.config_dict["retina_index"],
            correlation_threshold=self.model.config_dict[
                "oracle_correlation_threshold"
            ],
            explained_variance_threshold=self.model.config_dict[
                "explainable_variance_threshold"
            ], config=model.config_dict['config']
        )[cell_index]
        self.exciting_mei = exciting_mei.clone().cpu()
        self.envelope = get_mei_area(self.exciting_mei, mask_cutoff)
        # initial_state = self.initialize(initial_state)

        self.device = device
        surround = self.initialize().to(device)
        optimizer = optimizer(params=[surround], lr=learning_rate)
        surround.requires_grad_(True)

        self.precondition = MultipleConditions(preconditions)
        self.postprocessing = MultipleConditions(postprocessing)

        super().__init__(
            initial=surround,
            func=self.objective,
            optimizer=optimizer,
            transform=self.transform,
            regularization=self.regularize,
            precondition=self.precondition,
            postprocessing=self.postprocessing,
        )

    def objective(self, input):
        return -torch.mean(
            self.model.double()(
                self.exciting_mei.to("cuda")
                + (1 - torch.tensor(self.envelope).to("cuda")) * input
            )[:, self.cell_index]
        )

    def transform(self, input, iteration):
        return input

    def initialize(self):
        input = torch.DoubleTensor(*self.input_shape).uniform_(-0.1, 0.1)
        return input

    def regularize(self, input, iteration):
        return torch.tensor([0]).to(self.device)

    def freeze_model(self):
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(False)


class SuppressiveSurroundEnsembleMEI(SuppressiveSurroundMEI):
    def __init__(
        self,
        models,
        cell_index,
        input_shape,
        exciting_mei,
        optimizer,
        preconditions: list,
        postprocessing: list,
        learning_rate,
        device,
        mask_cutoff

    ):
        super().__init__(
            model=models[0],
            cell_index=cell_index,
            learning_rate=learning_rate,
            exciting_mei=exciting_mei,
            device=device,
            input_shape=input_shape,
            preconditions=preconditions,
            postprocessing=postprocessing,
            optimizer=optimizer,
            mask_cutoff=mask_cutoff

        )
        self.model = [model.to(device) for model in models]

    def freeze_model(self):
        for model in self.model:
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(False)

    def objective(self, input):
        activation = None
        for model in self.model:
            if activation is None:
                activation = -torch.mean(
                    model.double()(
                        self.exciting_mei.to("cuda")
                        + (1 - torch.tensor(self.envelope).to("cuda")) * input
                    )[:, self.cell_index]
                )
            else:
                activation = activation + -torch.mean(
                    model.double()(
                        self.exciting_mei.to("cuda")
                        + (1 - torch.tensor(self.envelope).to("cuda")) * input
                    )[:, self.cell_index]
                )

        return activation


def optimize(mei, stopper, tracker, save_tracks_interval=50):
    mei.freeze_model()
    max_activation = -1

    while True:
        current_state = mei.step()
        # wandb.log({'learning_rate': mei.optimizer.param_groups[0]['lr']}, step=mei.i_iteration)
        # if max_activation < current_state.evaluation:
        #     max_activation = current_state.evaluation
        # else:
        #     for i, g in enumerate(mei.optimizer.param_groups):
        #         if i == 0:
        #             print(f'lowering learning rate from {g["lr"]} to {g["lr"]*0.5}')
        #         g['lr'] = g['lr']*0.5

        stop, output = stopper(current_state)
        # if stopper.steps_wo_change > 20:
        #     for i, g in enumerate(mei.optimizer.param_groups):
        #         if i == 0:
        #             print(f'increasing learning rate from {g["lr"]} to {g["lr"] * 100}')
        #         g['lr'] = g['lr'] * 1.5
        #         stopper.steps_wo_change = 0
        current_state.stopper_output = output
        tracker.track(current_state)
        if (mei.i_iteration % save_tracks_interval) == 0:
            tracker.save_tracks(current_state=current_state)
        if stop:
            tracker.save_tracks(current_state=current_state, end=True)
            break
    return current_state.evaluation, current_state.post_processed_input


def optimize_like_golan(mei, stopper, tracker, lrs: list = None):
    if lrs is None:
        lrs = [1, 10, 100]
    mei.freeze_model()
    for lr in lrs:
        for g in mei.optimizer.param_groups:
            g["lr"] = lr
        optimize(mei, stopper, tracker)


def gauss2d(vx, vy, mu, cov):
    input_shape = vx.shape
    mu_x, mu_y = mu
    v = np.stack([vx.ravel() - mu_x, vy.ravel() - mu_y])
    cinv = inv(cholesky(cov))
    y = cinv @ v
    g = np.exp(-0.5 * (y * y).sum(axis=0))
    return g.reshape(input_shape)


def fit_gaussian(img):
    vx, vy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    rect = (img - img.mean()) ** 2
    pdf = rect / rect.sum()
    mu_x = (vx * pdf).sum()
    mu_y = (vy * pdf).sum()

    cov_xy = (vx * vy * pdf).sum() - mu_x * mu_y
    cov_xx = (vx**2 * pdf).sum() - mu_x**2
    cov_yy = (vy**2 * pdf).sum() - mu_y**2

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    g = gauss2d(vx, vy, (mu_x, mu_y), cov)
    mu = (mu_x, mu_y)
    return mu, cov, np.sqrt(g.reshape(img.shape))


def get_mei_area(initial_state, mask_cutoff):
    initial_state2d = (
        np.sum(np.abs(np.array(initial_state[0, 0])), axis=0) / initial_state.shape[2]
    )
    mu, cov, gaussian = fit_gaussian(initial_state2d)
    # mei_border = 1.5*np.sqrt(np.max(cov))
    # Y, X = np.ogrid[:initial_state2d.shape[-2], :initial_state2d.shape[-1]]
    # var = (mu, cov)
    mask = gaussian
    mask = np.where(mask > mask_cutoff, 1, mask/mask_cutoff)
    return mask
