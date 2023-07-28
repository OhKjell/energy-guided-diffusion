import numpy as np
import torch


class ChangeStd:
    def __init__(self, std: float = 1) -> None:
        """Change image pixel value such that the new image has the standard deviation given by std.
        Args:
            std (float): Desired standard deviation of the image
        """
        self.std = std

    def __call__(self, img: torch.Tensor, iteration) -> torch.Tensor:
        img_std = torch.std(img.view(img.shape[0], -1), dim=-1, unbiased=False)[0]
        fixed_std_img = img * (self.std / (img_std + 1e-9))
        return fixed_std_img


class ChangeStdAndClip:
    def __init__(
        self,
        std: torch.float64 = 1.0,
        max_value: torch.float64 = 1.0,
        min_value: torch.float64 = -1.0,
        contrast: torch.float64 = 1,
    ) -> None:
        """Change image pixel value such that the new image has the standard deviation given by std.
        Args:
            std (float): Desired standard deviation of the image
        """
        self.std = std
        self.min_value = min_value
        self.max_value = max_value
        self.contrast = contrast

    def __call__(self, img: torch.Tensor, iteration) -> torch.Tensor:
        img_std = torch.std(img.view(img.shape[0], -1), dim=-1, unbiased=False)[0]
        if img_std > self.std:
            fixed_std_img = img * (self.std / (img_std + 1e-9))
        else:
            fixed_std_img = img

        fixed_std_img = torch.where(
            fixed_std_img > self.max_value, self.max_value, fixed_std_img
        )
        fixed_std_img = torch.where(
            fixed_std_img < self.min_value, self.min_value, fixed_std_img
        )
        # return img
        return fixed_std_img

    # def __str__(self):
    #     return f'std{self.std}_and_clip'


class PNormConstraint:
    def __init__(self, norm_value: float, p: int = 1):
        self.p = p
        self.max_value = norm_value

    def __call__(self, img: torch.Tensor, iteration):
        norm_value = torch.pow(torch.sum(torch.pow(torch.abs(img), self.p)), 1 / self.p)
        if norm_value > self.max_value:
            normalized_img = img * (self.max_value / (norm_value + 1e-12))
        else:
            normalized_img = img
        return normalized_img


class MaxNorm:
    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, img: torch.Tensor, iteration):
        max_value = torch.max(torch.abs(img))
        img = img / max_value * self.contrast
        return img


class TorchL2NormAndClip:
    def __init__(
        self,
        norm_value: torch.float64 = 30.0,
        p: int = 1,
        max_pixel_value: torch.float64 = 1.0,
        min_pixel_value: torch.float64 = -1.0,
    ):
        self.max_value = norm_value
        self.max_pixel_value = max_pixel_value
        self.min_pixel_value = min_pixel_value
        self.p = p

    def __call__(self, img: torch.Tensor, iteration):
        # if iteration > 10:
        # img_shape = img.shape
        img = img / torch.norm(img, p=self.p) * self.max_value
        # img = torch.nn.functional.normalize(torch.flatten(img, start_dim=2), p=self.p, dim=(2))*self.max_value
        # img = img.unflatten(dim=2, sizes=img_shape[2:])

        normalized_img = img.double()
        normalized_img = torch.where(
            normalized_img > self.max_pixel_value, self.max_pixel_value, normalized_img
        )
        normalized_img = torch.where(
            normalized_img < self.min_pixel_value, self.min_pixel_value, normalized_img
        )
        return img


class PNormConstraintAndClip:
    def __init__(
        self,
        norm_value: torch.float64 = 30.0,
        p: int = 1,
        max_pixel_value: torch.float64 = 1.0,
        min_pixel_value: torch.float64 = -1.0,
    ):
        self.max_value = norm_value
        self.max_pixel_value = max_pixel_value
        self.min_pixel_value = min_pixel_value
        self.p = p

    def __call__(self, img: torch.Tensor, iteration):
        norm_value = torch.pow(torch.sum(torch.pow(torch.abs(img), self.p)), 1 / self.p)
        if norm_value > self.max_value:
            normalized_img = img * (self.max_value / (norm_value + 1e-12))
        else:
            normalized_img = img
        normalized_img = normalized_img.double()
        normalized_img = torch.where(
            normalized_img > self.max_pixel_value, self.max_pixel_value, normalized_img
        )
        normalized_img = torch.where(
            normalized_img < self.min_pixel_value, self.min_pixel_value, normalized_img
        )
        return normalized_img

    def __str__(self):
        return f"p{self.p}_norm_and_clip"


class CenteredSigmoid:
    def __init__(self, amplitude: float = 1):
        self.amplitude = amplitude

    def __call__(self, img: torch.Tensor, iteration):
        img = self.amplitude * 2 * (torch.sigmoid(img) - 0.5)
        centered_img = self.amplitude * 2 * (torch.sigmoid(img) - 0.5)
        return centered_img


class ThresholdCenteredSigmoid:
    def __init__(self, amplitude: float = 1, threshold=None):
        self.amplitude = amplitude
        if threshold is None:
            self.threshold = amplitude

    def __call__(self, img: torch.Tensor, iteration):
        # if self.threshold < torch.max(torch.abs(img)):
        centered_img = self.amplitude * 2 * (torch.sigmoid(img) - 0.5)
        return centered_img
        # else:
        #     return img


class MinMaxScaler:
    def __init__(self, min_amplitude: float = -1, max_amplitude: float = 1.0):
        self.amin = min_amplitude
        self.amax = max_amplitude

    def __call__(self, img: torch.Tensor, iteration):
        img = 2 * ((img - torch.min(img)) / (torch.max(img) - torch.min(img))) - 1
        return img


class ThresholdMinMaxScaler:
    def __init__(
        self, min_amplitude: float = -1, max_amplitude: float = 1.0, threshold=None
    ):
        self.amin = min_amplitude
        self.amax = max_amplitude
        if threshold is None:
            self.threshold = max_amplitude

    def __call__(self, img: torch.Tensor, iteration):
        if torch.max(torch.abs(img)) > self.threshold:
            img = 2 * ((img - torch.min(img)) / (torch.max(img) - torch.min(img))) - 1
        return img
