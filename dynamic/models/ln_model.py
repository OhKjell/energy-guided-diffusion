import scipy
from torch import nn
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from neuralpredictors.regularizers import LaplaceL2norm
import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dynamic.utils.global_functions import model_seed, set_random_seed
from dynamic.datasets.stas import crop_around_receptive_field
from dynamic.training.regularizers import l1_regularization

import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
            self,
            input_shape,
            num_of_neurons,
            num_of_frames,
            l1,
            gamma,
            do_nonlin=True,
            seed=None,
            do_sta=False,
            svd=False,
            fancy_nonlin=False,
            sta=None,
            learn_filter=True,
    ):
        super(Model, self).__init__()
        input_shape = (num_of_frames,) + input_shape
        self.conv1 = nn.Conv3d(1, num_of_neurons, kernel_size=input_shape, stride=1)
        self.init_conv(self.conv1)
        print("weight min", np.min(self.conv1.weight.detach().cpu().numpy()))
        print("weight max", np.max(self.conv1.weight.detach().cpu().numpy()))
        print("weight sum", np.sum(np.abs(self.conv1.weight.detach().cpu().numpy())))

        self.laplace = LaplaceL2norm()
        self.l1 = l1
        self.gamma = gamma
        self.do_sta = do_sta
        self.sta = sta

        if self.do_sta:
            init_with_sta(self.sta, layer=self.conv1, learnable=learn_filter)

        self.do_nonlin = do_nonlin
        if fancy_nonlin:
            self.nonlinearity = parametrized_softplus()
        else:
            self.nonlinearity = F.softplus
        if seed is None:
            self.seed = model_seed
        else:
            self.seed = seed
        set_random_seed(self.seed)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data, gain=0.005)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, data_key=None):
        x = self.conv1(x)
        x = torch.squeeze(x)
        if self.do_nonlin:
            x = self.nonlinearity(x)
        # x = torch.unsqueeze(x, 1)
        print(x.shape)
        x = torch.squeeze(x, 1).squeeze(-1)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return x

    def regularizer(self, data_key=None):
        return (
            self.gamma
            * torch.sum(
                torch.tensor(
                    [
                        self.laplace(self.conv1.weight[:, :, i])
                        for i in range(self.conv1.weight.shape[2])
                    ]
                )
            ),
            self.l1 * l1_regularization(self),
            torch.tensor([0], device="cuda"),
        )


class FactorizedModel(nn.Module):
    def __init__(
            self,
            input_shape,
            num_of_neurons,
            num_of_channels,
            l1,
            gamma,
            do_nonlin=True,
            seed=None,
            do_sta=False,
            do_svd=False,
            svd=None,
            fancy_nonlin=False,
            sta=None,
            learn_filter=True,
            fit_gaussian=False,
            dog=False,
            sign=None
    ):
        super().__init__()
        self.fit_gaussian = fit_gaussian
        self.dog = dog
        if self.fit_gaussian:
            if self.dog:
                self.conv1 = DoGConv3d(1, 1, input_shape, sign=sign)
            else:
                self.conv1 = GaussianConv3d(1, 1, input_shape)
        else:
            self.conv1 = nn.Conv3d(1, 1, kernel_size=(1,) + input_shape)
        self.conv2 = nn.Conv3d(1, num_of_neurons, kernel_size=(num_of_channels, 1, 1))
        self.init_conv(self.conv1)
        # print("weight min", np.min(self.conv1.weight.detach().cpu().numpy()))
        # print("weight max", np.max(self.conv1.weight.detach().cpu().numpy()))
        # print("weight sum", np.sum(np.abs(self.conv1.conv.weight.detach().cpu().numpy())))

        self.laplace = LaplaceL2norm()
        self.l1 = l1
        self.gamma = gamma
        self.do_sta = do_sta
        self.do_svd = do_svd
        self.svd = svd
        self.sta = sta

        if self.do_sta:
            if self.do_svd:
                init_with_sta(
                    np.expand_dims(self.svd[0], 0),
                    layer=self.conv1,
                    learnable=learn_filter,
                )
                init_with_sta(
                    np.expand_dims(np.expand_dims(self.svd[1], -1), -1),
                    layer=self.conv2,
                    learnable=learn_filter,
                )
            else:
                init_with_sta(
                    np.expand_dims(self.sta[0], 0),
                    layer=self.conv1,
                    learnable=learn_filter,
                )
                init_with_sta(
                    np.expand_dims(np.expand_dims(self.sta[1], -1), -1),
                    layer=self.conv2,
                    learnable=learn_filter,
                )

        self.do_nonlin = do_nonlin
        if fancy_nonlin:
            self.nonlinearity = parametrized_softplus()
        else:
            self.nonlinearity = F.softplus
        if seed is None:
            self.seed = model_seed
        else:
            self.seed = seed
        set_random_seed(self.seed)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=0.005)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, data_key=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.do_nonlin:
            x = self.nonlinearity(x)
        x = torch.squeeze(x, 1).squeeze(-1)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return x

    def regularizer(self, data_key=None):
        if not self.fit_gaussian:
            return (
                self.gamma * self.laplace(self.conv1.weight[:, :, 0]),
                torch.tensor([0], device="cuda"),
                self.l1 * l1_regularization(self),
            )
        else:
            return (torch.tensor([0], device="cuda"),
                    torch.tensor([0], device="cuda"),
                    torch.tensor([0], device="cuda"))


def init_with_sta(sta, layer, learnable=False, bias=0):
    layer.weight.data = torch.tensor(sta).unsqueeze(0).unsqueeze(0)
    if layer.bias is not None:
        layer.bias.data.fill_(bias)
        layer.bias.requires_grad = learnable
    layer.weight.requires_grad = learnable


class GaussianConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Learnable parameters for mean and sigma
        self.mean_x = nn.Parameter(torch.zeros(1))
        self.mean_y = nn.Parameter(torch.zeros(1))
        self.sigma_x = nn.Parameter(torch.ones(1))
        self.sigma_y = nn.Parameter(torch.ones(1))
        self.sigma_xy = nn.Parameter(torch.zeros(1))
        self.x_coord = torch.linspace(-self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[0]).cuda()
        self.y_coord = torch.linspace(-self.kernel_size[1] // 2, self.kernel_size[1] // 2, self.kernel_size[1]).cuda()

        self.x, self.y = torch.meshgrid(self.x_coord, self.y_coord)

    def forward(self, x):
        # Create the Gaussian kernel
        # self.x.requires_grad = False
        # self.y.requires_grad = False
        kernel = create_gaussian_kernel(self.sigma_x, self.sigma_y, self.sigma_xy, self.x, self.y,
                                        self.mean_x, self.mean_y, self.kernel_size)
        x = F.conv3d(x, kernel, padding=0)
        # print('weight', self.conv.weight.shape)
        # print('x', x.shape)

        return x


def create_gaussian_kernel(sigma_x, sigma_y, sigma_xy, x, y, mean_x, mean_y, kernel_size, amplitude=1.0):
    # Create a grid of (x,y) coordinates
    # print(self.mean_x.device)
    # print(self.x.device)
    # Gaussian kernel
    sigma_xx = torch.exp(sigma_x) ** 2
    sigma_yy = torch.exp(sigma_y) ** 2

    # Covariance matrix
    covariance_matrix = torch.stack([
        torch.stack([sigma_xx, sigma_xy]),
        torch.stack([sigma_xy, sigma_yy])
    ]).squeeze()
    inverse_covariance_matrix = torch.inverse(covariance_matrix)

    # Coordinates
    x_coord = x - mean_x
    y_coord = y - mean_y
    x_coord_flat = x_coord.flatten()
    y_coord_flat = y_coord.flatten()

    # Prepare coordinates for Gaussian computation
    coords = torch.stack([x_coord_flat, y_coord_flat], dim=1)

    # Compute the Gaussian kernel
    gaussian_exponent = -0.5 * (coords @ inverse_covariance_matrix * coords).sum(dim=1)
    gaussian_kernel_flat = amplitude * torch.exp(gaussian_exponent)

    gaussian_kernel = gaussian_kernel_flat.view_as(x)
    # print(gaussian_kernel_flat.shape)

    # gaussian_kernel /= gaussian_kernel.sum()

    # Add channel dimensions
    # print('kernel shape:', gaussian_kernel.view(1, 1,  self.kernel_size[0], self.kernel_size[1]).double().shape)
    return gaussian_kernel.view(1, 1, 1, kernel_size[0], kernel_size[1]).double()


class DoGConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sign):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Learnable parameters for mean and sigma
        self.mean_x = nn.Parameter(torch.zeros(1))
        self.mean_y = nn.Parameter(torch.zeros(1))
        self.sigma_x1 = nn.Parameter(torch.ones(1))
        self.sigma_y1 = nn.Parameter(torch.ones(1))
        self.sigma_xy1 = nn.Parameter(torch.zeros(1))
        self.amplitude1 = nn.Parameter(sign*torch.ones(1))

        self.sigma_x2 = nn.Parameter(2*torch.ones(1))
        self.sigma_y2 = nn.Parameter(2*torch.ones(1))
        self.sigma_xy2 = nn.Parameter(torch.zeros(1))
        self.amplitude2 = nn.Parameter(sign*0.5*torch.ones(1))
        self.lower, self.upper = (-float('inf'), 0) if (sign < 0) else (0., float('inf'))

        self.x_coord = torch.linspace(-self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[0]).cuda()
        self.y_coord = torch.linspace(-self.kernel_size[1] // 2, self.kernel_size[1] // 2, self.kernel_size[1]).cuda()

        self.x, self.y = torch.meshgrid(self.x_coord, self.y_coord)

    def forward(self, x):
        # Create the Gaussian kernel
        # self.x.requires_grad = False
        # self.y.requires_grad = False
        kernel = self.create_dog_kernel()
        x = F.conv3d(x, kernel, padding=0)
        # print('weight', self.conv.weight.shape)
        # print('x', x.shape)

        return x
    def log(self):
        pass

    def create_dog_kernel(self):
        # Create a grid of (x,y) coordinates
        # print(self.mean_x.device)
        # print(self.x.device)
        # Gaussian kernel
        self.amplitude1.data.clamp_(self.lower, self.upper)
        self.amplitude2.data.clamp_(self.lower, self.upper)

        gauss_1 = create_gaussian_kernel(self.sigma_x1, self.sigma_y1, self.sigma_xy1, self.x, self.y,
                                         self.mean_x, self.mean_y, self.kernel_size, amplitude=self.amplitude1)
        gauss_2 = create_gaussian_kernel(self.sigma_x2, self.sigma_y2, self.sigma_xy2, self.x, self.y,
                                         self.mean_x, self.mean_y, self.kernel_size, amplitude=self.amplitude2)
        return gauss_1 - gauss_2


class LearnedSoftPlus(torch.nn.Module):
    def __init__(self, init_beta=1.0, threshold=20):
        super().__init__()
        # keep beta > 0
        self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log())
        self.threshold = 20

    def forward(self, x):
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(beta_x < 20, torch.log1p(beta_x.exp()) / beta, x)


class ParametrizedELU(nn.Module):
    def __init__(self, a=1.0, b=1.0, c=1.0, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Parameter(torch.tensor(a, device=device), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(b, device=device), requires_grad=True)
        self.c = torch.nn.Parameter(torch.tensor(c, device=device), requires_grad=True)

    def forward(self, x):
        return torch.where(x > 0, self.c * x, self.a * (torch.exp(x / self.b) - 1))


class paremetrized_exp(nn.Module):
    """
    Implementation of parametrized exponential activation.
    """

    def __init__(self, alpha=None):
        super().__init__()
        self.weight_clamg = ParamClamp()
        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

        self.weight_clamg.apply(self.alpha)

    def forward(self, x):
        return torch.exp(self.alpha * x)


class ParamClamp(torch.autograd.Function):
    def forward(ctx, input, **kwargs):
        return input.clamp(min=0)

    def backward(ctx, grad_output):
        return grad_output


class parametrized_elu(nn.Module):
    def __init__(self, a=1.0, b=1.0, w=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=True)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=True)

    def forward(self, x):
        return torch.where(x >= 0, self.w * x, self.a * (torch.exp(x / self.b) - 1))


class parametrized_softplus(nn.Module):
    def __init__(self, a=1.0, b=0.0, w=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=True)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=True)

    def forward(self, x):
        return self.a * torch.log(1 + torch.exp(self.w * x + self.b))


class multi_softplus(nn.ModuleDict):
    def __init__(self, num_of_cells, data_keys=None):
        super().__init__()
        self.a, self.b, self.w = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        for data_key in data_keys:
            self.a[data_key] = nn.Parameter(torch.ones(num_of_cells[data_key]), requires_grad=True)
            self.b[data_key] = nn.Parameter(torch.zeros(num_of_cells[data_key]), requires_grad=True)
            self.w[data_key] = nn.Parameter(torch.ones(num_of_cells[data_key]), requires_grad=True)
        self.data_keys = data_keys

    def forward(self, x, data_key):
        if data_key is None:
            data_key = list(self.data_keys)[0]
        a_expanded = self.a[data_key].unsqueeze(0).expand_as(x)
        b_expanded = self.b[data_key].unsqueeze(0).expand_as(x)
        w_expanded = self.w[data_key].unsqueeze(0).expand_as(x)

        # Apply custom parametrized softplus
        output = a_expanded * torch.log(1 + torch.exp(w_expanded * x + b_expanded))

        return output


class IdentityCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("Identity", nn.Identity())

    def forward(self, x):
        return self.features(x)


class LNLayer(nn.Module):
    def __init__(self, num_of_neurons, input_shape, num_of_channels, bias=True):
        super().__init__()
        self.input_shape = input_shape
        assert len(input_shape) == 2
        self.conv = nn.Conv2d(
            num_of_channels, num_of_neurons, kernel_size=input_shape, bias=bias
        )

    def forward(self, x):
        x = self.conv(x)
        # x = torch.squeeze(torch.squeeze(x, dim=2), dim=2)
        return x


class LNReadout(nn.ModuleDict):
    def __init__(
            self,
            core,
            in_shape_dict,
            n_neurons_dict,
            bias,
            smoothing_lambda=0.1,
            l1_lambda=0.001,
            input_regularizer="LaplaceL2Norm",
            laplace_padding=None,
            activation_function="exp",
            movie_like=False,
            num_of_channels=1,
    ):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.smoothing_lambda = smoothing_lambda
        self.activation_function = activation_function
        for k in n_neurons_dict:
            if movie_like:
                in_shape = in_shape_dict[k][1:][-3], in_shape_dict[k][1:][-2]
            else:
                in_shape = in_shape_dict[k][1:][-2], in_shape_dict[k][1:][-1]

            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                LNLayer(
                    num_of_neurons=n_neurons,
                    input_shape=in_shape,
                    bias=bias,
                    num_of_channels=num_of_channels,
                ),
            )

        self._input_weights_regularizer = LaplaceL2norm(padding=laplace_padding)

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def laplace(self, data_key):
        return self._input_weights_regularizer(self[data_key].conv.weight)
        # return self._input_weights_regularizer(self[data_key].linear.weight)

    def regularizer(self, data_key):
        # return self.smoothing_lambda * self.laplace(data_key) + self.l1_lambda * torch.norm(self[data_key].conv.weight,
        #                                                                                     1)
        return self.l1_lambda * torch.norm(self[data_key].conv.weight, 1)


# do not look at!
def linear_nonlinear_model(
        dataloaders,
        seed,
        readout_bias=True,
        l1_lambda=0.0001,
        smoothing_lambda=0.1,
        input_regularizer="LaplaceL2Norm",
        activation_function="exp",
        num_of_channels=1,
):
    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    class Encoder(nn.Module):
        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout
            activation_functions = {
                "relu": nn.ReLU(),
                "softplus": nn.Softplus(),
                "exp": torch.exp,
                "p_exp": paremetrized_exp(),
                "p_softplus": LearnedSoftPlus(),
            }
            assert self.readout.activation_function in activation_functions.keys()
            self.activation_function = activation_functions[
                self.readout.activation_function
            ]

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            # x = self.activation_function(x)

            return x

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key)

    core = IdentityCore()

    readout = LNReadout(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        bias=readout_bias,
        l1_lambda=l1_lambda,
        smoothing_lambda=smoothing_lambda,
        input_regularizer=input_regularizer,
        activation_function=activation_function,
        num_of_channels=num_of_channels,
    )

    model = Encoder(core, readout)

    return model


def fit_nonlinearity(
        dataloaders,
        model_filter,
        device,
        retina_index,
        rf_size,
        max_coordinate,
        h,
        w,
        plot=True,
):
    x_values = torch.tensor([], device=device)
    y_values = torch.tensor([], device=device)
    with torch.no_grad():
        for images, responses in tqdm(
                dataloaders["train"][str(retina_index + 1).zfill(2)]
        ):
            images = images.double().to(device)
            responses = responses.to(device)
            if rf_size != (h, w):
                images = crop_around_receptive_field(
                    max_coordinate=max_coordinate,
                    images=images,
                    rf_size=rf_size,
                    h=h,
                    w=w,
                )

            # x = [(images[i].flatten()*model_filter.flatten().to(device)) for i in range(images.shape[0])]
            x = torch.sum(images * model_filter.to(device), axis=(1, 2, 3))
            x_values = torch.cat((x_values, x))
            # y_values += responses.squeeze().tolist()
            y_values = torch.cat((y_values, responses.squeeze()))

    coefs, _ = scipy.optimize.curve_fit(
        lambda t, a, b: a * numpy.exp(b * t),
        x_values.detach().cpu(),
        y_values.detach().cpu(),
    )
    if plot:
        plt.scatter(x_values, y_values)
        x = np.linspace(min(x_values), max(x_values), 1000)
        y = coefs[0] * np.exp(coefs[1] * x)
        plt.plot(x, y)
        plt.show()
    return coefs