import numpy as np

from utils.global_functions import cuda


def correlation(output, target, eps, dim=0):
    delta_out = output - output.mean(dim, keepdim=True)
    delta_target = target - target.mean(dim, keepdim=True)

    var_out = delta_out.pow(2).mean(dim, keepdim=True)
    var_target = delta_target.pow(2).mean(dim, keepdim=True)

    corrs = (delta_out * delta_target).mean(dim, keepdim=True) / (
        (var_out + eps) * (var_target + eps)
    ).sqrt()
    return corrs


def corr(y1, y2, axis=-1, eps: int = 1e-8, **kwargs) -> np.ndarray:
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).
    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2
    Returns: correlation array
    """

    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    return (y1 * y2).mean(axis=axis, **kwargs)


def variance_of_predictions(outputs):
    if cuda:
        variances = np.var(outputs.cpu().detach().numpy(), axis=0)
    else:
        variances = np.var(outputs.detach().numpy(), axis=0)
    mean_var = np.mean(variances)
    return mean_var


def oracle_corr_conservative(repeated_outputs) -> np.ndarray:
    """
    Compute the corrected oracle correlations per neuron.
    Note that an unequal number of repeats will introduce bias as it distorts assumptions made about the dataset.
    Note that oracle_corr_conservative overestimates the true oracle correlation.
    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).
    Returns:
        array: Corrected oracle correlations per neuron
    """

    var_noise, var_mean = [], []
    for output in repeated_outputs:
        var_noise.append(output.var(axis=0))
        var_mean.append(output.mean(axis=0))
    var_noise = np.mean(np.array(var_noise), axis=0)
    var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def oracle_corr_jackknife(repeated_outputs) -> np.ndarray:
    """
    Compute the oracle correlations per neuron.
    Note that an unequal number of repeats will introduce bias as it distorts assumptions made about the dataset.
    Note that oracle_corr_jackknife underestimates the true oracle correlation.
    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).
    Returns:
        array: Oracle correlations per neuron
    """

    oracles = []
    for outputs in repeated_outputs:
        num_repeats = outputs.shape[0]
        oracle = (outputs.sum(axis=0, keepdims=True) - outputs) / (num_repeats - 1)
        if np.any(np.isnan(oracle)):
            print(
                "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(
                    np.isnan(oracle).mean() * 100
                )
            )
            oracle[np.isnan(oracle)] = 0
        oracles.append(oracle)
    return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)
