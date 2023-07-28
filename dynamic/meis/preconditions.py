import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import signal


class GaussianBlur3d:
    def __init__(self, sigma, sigma_temp, truncate: int = 4, pad_mode: str = "reflect"):
        self.spatial_blur = GaussianBlurSpatial(
            sigma=sigma, truncate=truncate, pad_mode=pad_mode
        )
        self.temporal_blur = GaussianBlurTemporal(
            sigma=sigma_temp, truncate=truncate, pad_mode=pad_mode
        )

    def __call__(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        x = self.spatial_blur(x, iteration)
        x = self.temporal_blur(x, iteration)
        return x


class MultipleConditions:
    def __init__(self, conditions):
        """

        Args:
            conditions: tuple consisting of callable and a dict with parameters for the callable
        """
        self.preconditions = []

        for precondition in conditions:
            self.preconditions.append(precondition[0](**precondition[1]))

    def __call__(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        for p in self.preconditions:
            x = p(x, iteration)
        return x


class GaussianBlurSpatial:
    """Blur image or gradients with a Gaussian window.
    Adapted from the code of Max F. Burg in https://github.com/MaxFBurg/controversial-stimuli/blob/c0c0c64ecc09dcfd56201bd27061c469df3217ba/controversialstimuli/utility/pixel_image.py#L9
    """

    def __init__(self, sigma: float, truncate: float = 4, pad_mode: str = "reflect"):
        """Blur image or gradients with a Gaussian window. Initialize callable class.
        Args:
            sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring
            truncate (float, optional): Gaussian window is truncated after this number of standard deviations to each
                side. Defaults to 4.
            pad_mode (str, optional): Images / gradients are padded by `truncate * sigma` to each side by using this
                padding mode. Valid uses are 'constant', 'reflect' and 'replicate'. Defaults to "reflect".
        """
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.truncate = truncate
        self.pad_mode = pad_mode

        # Define 1-dim kernels to use for blurring
        self.y_halfsize = max(int(round(self.sigma[0] * self.truncate)), 1)
        self.y_gaussian = signal.gaussian(2 * self.y_halfsize + 1, std=self.sigma[0])

        self.x_halfsize = max(int(round(self.sigma[1] * self.truncate)), 1)
        self.x_gaussian = signal.gaussian(2 * self.x_halfsize + 1, std=self.sigma[1])

    def __call__(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        num_channels = x.shape[1]

        self.y_gaussian = torch.as_tensor(
            self.y_gaussian, device=x.device, dtype=x.dtype
        )
        self.x_gaussian = torch.as_tensor(
            self.x_gaussian, device=x.device, dtype=x.dtype
        )

        # Blur
        padded_x = None
        blurred_x = None
        for step in range(x.shape[2]):
            step_padded_x = F.pad(
                x[:, :, step],
                pad=(
                    self.x_halfsize,
                    self.x_halfsize,
                    self.y_halfsize,
                    self.y_halfsize,
                ),
                mode=self.pad_mode,
            )

            step_blurred_x = F.conv2d(
                step_padded_x,
                self.y_gaussian.repeat(num_channels, num_channels, 1)[..., None],
                groups=num_channels,
            )
            # after repeat: shape (num_channels, num_channels, height=num_gauss_window_points). Then, add width dimension of 1

            step_blurred_x = F.conv2d(
                step_blurred_x,
                self.x_gaussian.repeat(num_channels, num_channels, 1, 1),
                groups=num_channels,
            )
            step_padded_x = torch.unsqueeze(step_padded_x, 2)
            step_blurred_x = torch.unsqueeze(step_blurred_x, 2)

            if padded_x is None:
                padded_x = step_padded_x
                blurred_x = step_blurred_x
            else:
                padded_x = torch.cat((padded_x, step_padded_x), 2)
                blurred_x = torch.cat((blurred_x, step_blurred_x), 2)

        final_x = blurred_x / (
            self.y_gaussian.sum() * self.x_gaussian.sum()
        )  # normalize

        return final_x.double()


class GaussianBlurTemporal:
    def __init__(self, sigma: float, truncate: float = 4, pad_mode: str = "reflect"):
        """Blur image or gradients with a Gaussian window. Initialize callable class.
        Args:
            sigma (float): Standard deviation in x used for the gaussian blurring
            truncate (float, optional): Gaussian window is truncated after this number of standard deviations to each
                side. Defaults to 4.
            pad_mode (str, optional): Images / gradients are padded by `truncate * sigma` to each side by using this
                padding mode. Valid uses are 'constant', 'reflect' and 'replicate'. Defaults to "reflect".
        """
        self.sigma = sigma
        self.truncate = truncate
        self.pad_mode = pad_mode

        self.x_halfsize = max(int(round(self.sigma * self.truncate)), 1)
        self.x_gaussian = signal.gaussian(2 * self.x_halfsize + 1, std=self.sigma)

    def __call__(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        num_channels = x.shape[1]

        self.x_gaussian = torch.as_tensor(
            self.x_gaussian, device=x.device, dtype=x.dtype
        )
        blurred_x = torch.zeros(x.shape, device=x.device)

        # Blur

        for i in range(x.shape[-2]):
            for j in range(x.shape[-1]):
                step_padded_x = F.pad(
                    x[:, :, :, i, j],
                    pad=(self.x_halfsize, self.x_halfsize),
                    mode=self.pad_mode,
                )
                step_blurred_x = F.conv1d(
                    step_padded_x,
                    self.x_gaussian.repeat(num_channels, num_channels, 1),
                    groups=num_channels,
                )
                # after repeat: shape (num_channels, num_channels, height=num_gauss_window_points). Then, add width dimension of 1

                # step_blurred_x = F.conv1d(step_blurred_x, self.x_gaussian.repeat(num_channels, num_channels, 1, 1),
                #                           groups=num_channels)
                # step_padded_x = step_padded_x.unsqueeze(-1).unsqueeze(-1)
                # step_blurred_x = step_blurred_x.unsqueeze(-1).unsqueeze(-1)

                blurred_x[:, :, :, i, j] = step_blurred_x

        final_x = blurred_x / (self.x_gaussian.sum())  # normalize

        return final_x.double()


class SuppressGradient:
    def __init__(self, mask):
        self.mask = mask

    def __call__(
        self, x: torch.Tensor, surround: torch.Tensor, iteration: int
    ) -> torch.Tensor:
        # plt.imshow(x[0, 0, 10].clone().cpu())
        # plt.show()

        # plt.imshow(x[0, 0, 10].clone().cpu())
        # plt.show()
        return x + (1 - self.mask) * surround


class GradientMask:
    def __init__(self, center_coordinates, mask_size, input_shape):
        self.center_coordinates = center_coordinates
        self.mask_size = mask_size
        Y, X = np.ogrid[: input_shape.shape[-2], : input_shape.shape[-1]]
        dist_from_rf_location = np.max(
            np.abs(X - int(center_coordinates[0])),
            np.abs(Y - int(center_coordinates[1])),
        )

        self.mask = torch.BoolTensor(dist_from_rf_location >= self.mask_size)

    def __call__(self, img: torch.Tensor, iteration):
        img = img.masked_fill(self.mask, 0)
        return img
