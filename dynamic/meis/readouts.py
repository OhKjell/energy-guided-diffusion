import math

import torch
import torch.nn.functional as F
from torch import nn

from dynamic.datasets.stas import (calculate_position_before_convs,
                           crop_around_receptive_field, normalize_source_grid,
                           unnormalize_source_grid)


class SingleCellMEIEncoder(nn.Module):
    def __init__(self, model, cell_index):
        super().__init__()
        self.initial_model = model
        self.config_dict = model.config_dict
        self.core = model.core
        self.cell_index = cell_index
        self.feature_vector = nn.Parameter(
            model.readout._features[:, :, :, cell_index : cell_index + 1],
            requires_grad=False,
        )
        self.learned_position = model.readout.grid[0, cell_index, 0, :]
        self.h = model.config_dict["img_h"]
        self.w = model.config_dict["img_w"]
        self.core_output_shape = model.core.get_output_shape(
            model.config_dict["in_shapes_dict"][
                f"0{model.config_dict['retina_index'] +1 }"
            ]
        )
        self.input_position, size = self.calculate_input_center_coordinate()
        self.size = (int(size), int(size))

    def forward(self, x):
        x = self.crop_around_center_coordinate(x)
        core_out = self.core(x)
        out_core = torch.transpose(core_out, 1, 2)
        out_core = out_core.reshape(((-1,) + out_core.size()[2:]))
        readout_out = (out_core * self.feature_vector).sum(1).squeeze(-1)
        out = self.initial_model.nonlinearity(readout_out)
        return out

    def crop_around_center_coordinate(self, img):
        images = crop_around_receptive_field(
            self.input_position, img, self.size, self.h, self.w
        )
        return images

    def calculate_input_center_coordinate(self):
        self.learned_position = unnormalize_source_grid(
            self.learned_position.detach().cpu(), self.core_output_shape
        )
        # TODO: Problematic calculation because the readout does bilinear interpolation but
        # I have to get some coordinates to crop around which is hard to interpolate and I am not sure how to do that
        # so I round it to the nearest coordinate

        # sampled_value = F.grid_sample(input=input.double(), grid=self.initial_model.readout.grid[:, 0, :, :].detach().cpu(),
        #                               align_corners=self.initial_model.readout.align_corners)[0,0,0,0].item()

        input_position, size = calculate_position_before_convs(
            center=self.learned_position,
            kernel_sizes=[self.config_dict["spatial_input_kern"][0]]
            + [
                self.config_dict["spatial_hidden_kern"][0]
                for _ in range(self.initial_model.config_dict["layers"] - 1)
            ],
            feature_h=self.core_output_shape[-2],
            feature_w=self.core_output_shape[-1],
            img_h=self.h,
            img_w=self.w,
            plot=True,
        )

        return input_position, size
