import os
from pathlib import Path

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm

from randpy_wn.randpy import ran1


def get_random_stimuli(seed, time_bins, height, width):
    random_number, next_seed = ran1(seed, time_bins * height * width)
    stimulus = np.heaviside(np.array(random_number) - 0.5, 1).astype(int) * 2 - 1
    stimulus = np.reshape(stimulus, (height, width, time_bins), order="F")
    return stimulus, next_seed


def show_array_as_video(array):
    mis_pos = get_mis_pos(array)
    mis_sta = array[:, mis_pos[1], mis_pos[2]]

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(array[0, ...], cmap="gray", vmin=array.min(), vmax=array.max())

    s = plt.Rectangle(
        (mis_pos[2] - 30, mis_pos[1] - 30),
        60,
        60,
        fc=(0.0, 0.0, 0.0, 0.0),
        ec="tab:orange",
    )
    ax.add_patch(s)

    inset_ax = mpl_il.inset_axes(ax, width="25%", height="25%", loc="lower right")
    sta = inset_ax.plot(mis_sta, lw=1)
    vl = inset_ax.axvline(0, color="tab:red", ls="--")
    inset_ax.xaxis.tick_top()
    inset_ax.set(xticklabels=[], yticklabels=[])

    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(array[0, ...])
        vl.set_xdata(0)

    def animate(i):
        im.set_data(array[i, ...])
        vl.set_xdata(0 + i)
        return im

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=array.shape[0], interval=100
    )
    return anim


def load_stimulus_array(
    kind, time_bins, height, width, stimuli_path, trial_id=None, seed=None
):
    if kind == "frozen":
        frozen_noise_path = os.path.join(stimuli_path, "frozen_noise.npz")
        if os.path.exists(frozen_noise_path):
            zip_data = np.load(frozen_noise_path)
            stimuli = zip_data["arr_0"]

            next_seed = None
        else:
            if seed is None:
                frozen_seed = -20000
            else:
                frozen_seed = seed
            stimuli, next_seed = get_random_stimuli(
                frozen_seed, time_bins, height, width
            )
            np.savez_compressed(frozen_noise_path, stimuli)
    elif kind == "running":
        if trial_id is None:
            raise "For running noise, a valid trial id is required!"

        running_noise_path = os.path.join(stimuli_path, "running_stimuli.npy")
        if os.path.exists(running_noise_path):
            mmap_data = np.load(running_noise_path, mmap_mode="r")
            stimuli = mmap_data[trial_id]

            next_seed = None
        else:
            if seed is None:
                running_seed = -10000
            else:
                running_seed = seed
            stimuli, next_seed = get_random_stimuli(
                running_seed, time_bins, height, width
            )
    else:
        raise ValueError(
            f"Invalid kind of stimulus requested: {kind}\nUse either 'frozen' or 'running'."
        )

    return stimuli, next_seed


def get_sta(
    response_array,
    cell,
    noise_kind,
    height,
    width,
    kernel_size,
    stimuli_path,
    vmin=-1,
    vmax=1,
):
    time_bins = response_array.shape[2]

    summed_stimuli = np.zeros((height, width, kernel_size))
    next_seed = None
    spike_count = 0
    for t, trial in tqdm(
        enumerate(response_array[cell]),
        total=response_array[cell].shape[0],
        desc="Trials",
        leave=False,
    ):
        stimuli, next_seed = load_stimulus_array(
            kind=noise_kind,
            time_bins=time_bins,
            height=height,
            width=width,
            trial_id=t,
            stimuli_path=stimuli_path,
            seed=next_seed,
        )
        for b, bin_count in enumerate(trial):
            if b >= kernel_size and bin_count > 0:
                summed_stimuli += bin_count * np.transpose(
                    stimuli[b - kernel_size : b, ...], (1, 2, 0)
                )
                spike_count += bin_count

    return scale(summed_stimuli / spike_count, (vmin, vmax))


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def get_mis_pos(array):
    """Returns the position of the most important stixel in the array."""
    temporal_variances = np.var(array, axis=0)
    pix_pos = np.unravel_index(
        np.argmax(temporal_variances), (array.shape[1], array.shape[2])
    )
    time_pos = np.argmin(array[:, pix_pos[0], pix_pos[1]])
    return time_pos, pix_pos[0], pix_pos[1]


def show_arrays_as_video(arrays):
    num_cells = len(arrays)
    fig, axes = plt.subplots(
        ncols=(num_cells // 10) + 1,
        nrows=(num_cells // 8) + 1,
        squeeze=True,
        sharex="all",
        sharey="all",
        layout="constrained",
        figsize=(20, 16),
    )

    images = {}  # imshow objects
    #     cs = {}  # circles
    vls = {}  # vertical lines
    insets = {}  # inset_axes
    focus_insets = {}  # insets for zoomed in receptive field

    for a, sta in enumerate(tqdm(arrays)):
        ax = axes.flatten()[a]
        mis_pos = get_mis_pos(sta)
        mis_sta = sta[:, mis_pos[1], mis_pos[2]]

        images[a] = ax.imshow(
            sta[
                0, mis_pos[1] - 30 : mis_pos[1] + 30, mis_pos[2] - 30 : mis_pos[2] + 30
            ],
            cmap="gray",
        )
        #         cs[a] = plt.Rectangle((mis_pos[2] - 30, mis_pos[1] - 30), 60, 60,
        #                               fc=(0.0, 0.0, 0.0, 0.0), ec="tab:orange")
        #         ax.add_patch(cs[a])

        insets[a] = mpl_il.inset_axes(ax, width="25%", height="25%", loc="lower right")

        insets[a].plot(mis_sta, lw=1)
        vls[a] = insets[a].axvline(0, lw=1, ls="--", color="tab:red")
        insets[a].xaxis.tick_top()
        insets[a].set(xticklabels=[], yticklabels=[])

    total_axes = axes.flatten().size
    total_cells = len(arrays)
    diff = total_cells - total_axes

    if diff > 0:
        for i in range(diff):
            ax = axes.flatten()[::-1][i]
            ax.remove()

    plt.close()

    def init():
        for a, array in enumerate(tqdm(arrays, desc="Animating frame 0", leave=False)):
            mis_pos_arr = get_mis_pos(array)
            images[a].set_data(
                array[
                    0,
                    mis_pos_arr[1] - 30 : mis_pos_arr[1] + 30,
                    mis_pos_arr[2] - 30 : mis_pos_arr[2] + 30,
                ]
            )
            vls[a].set_xdata(0)

    def animate(i):
        for a, array in enumerate(
            tqdm(arrays, desc=f"Animating frame {i + 1}", leave=False)
        ):
            mis_pos_arr = get_mis_pos(array)
            images[a].set_data(
                array[
                    i,
                    mis_pos_arr[1] - 30 : mis_pos_arr[1] + 30,
                    mis_pos_arr[2] - 30 : mis_pos_arr[2] + 30,
                ]
            )
            vls[a].set_xdata(i)
        return 0

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=arrays.shape[1], interval=100
    )

    return anim


def find_frametimings_file(exp, fol):
    outpath = None
    for path in os.listdir(fol / "frametimings"):
        if path.split("_")[0] == str(exp):
            outpath = path

    return Path(fol) / "frametimings" / outpath


def find_spiketime_files(exp, fol):
    path_list = []
    for path in os.listdir(fol / "spiketimes"):
        if path.split("_")[0] == str(exp):
            path_list.append(path)

    return [Path(fol) / "spiketimes" / path for path in path_list]


def estimate_filter_kernel(stim, spike_times, spike_bin_ids, kernel_size):
    snippets = np.zeros((spike_times.size, kernel_size))

    for s, spike in enumerate(spike_times):
        spike_bin = spike_bin_ids[s]
        if spike_bin > kernel_size:
            snippets[s] = stim[spike_bin - kernel_size : spike_bin]

    return snippets.mean(axis=0)


def get_avg_spike_counts(bin_spks, conv_bins, conv_bin_ids, kernel_size):
    average_spike_counts = np.zeros((conv_bins.size - 1))

    for i in range(conv_bins.size):
        bin_conv_pos = np.where(conv_bin_ids == i)[0]
        if bin_conv_pos.size > 0:
            bin_conv_pos += kernel_size - 2
            spikes_in_bin = bin_spks[bin_conv_pos]

            try:
                average_spike_counts[i] = np.mean(spikes_in_bin)
            except IndexError:
                continue

    return average_spike_counts


def get_quantile_bins(data, num_bins):
    quantiles = np.linspace(0.0, 1.0, num_bins)
    bins = np.zeros_like(quantiles)
    for q, quant in enumerate(quantiles):
        bins[q] = np.quantile(data.flatten(), quant)

    return bins


def get_linear_bins(data, num_bins):
    d_min, d_max = np.floor(data.min()), np.ceil(data.max())

    return np.linspace(d_min, d_max, num_bins)


def predict_non_linearities(
    stim_file, pulse_file, spiketime_files, kernel_size, convolved_bins_method="quant"
):
    plt.close()

    num_units = len(spiketime_files)
    nrows = num_units // 5 + 1
    ncols = 5

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex="all",
        sharey="all",
        squeeze=True,
        constrained_layout=True,
        num=2,
        figsize=(36, nrows * 5),
    )

    pulse_timings = np.loadtxt(pulse_file)
    stimulus = np.loadtxt(stim_file)

    for sf, spike_file in enumerate(spiketime_files):
        spiketimes = np.loadtxt(spike_file)
        binned_spks, _ = np.histogram(spiketimes, bins=pulse_timings)
        bin_spk_ids = np.digitize(spiketimes, pulse_timings)

        est_filter = estimate_filter_kernel(
            stimulus, spiketimes, bin_spk_ids, kernel_size
        )

        convolved_signal = np.correlate(
            stimulus[: pulse_timings.size], est_filter, mode="valid"
        )

        if convolved_bins_method == "quant":
            convolved_bins = get_quantile_bins(convolved_signal, 20)
        elif convolved_bins_method == "lin":
            convolved_bins = np.linspace(
                convolved_signal.min(), convolved_signal.max(), 101
            )
        else:
            raise ValueError(
                "Invalid input for convolved bins method! Valid inputs: lin, quant."
            )

        convolved_bin_ids = np.digitize(convolved_signal, convolved_bins)

        average_spike_counts = get_avg_spike_counts(
            binned_spks, convolved_bins, convolved_bin_ids, kernel_size
        )

        xs = (convolved_bins[1:] + convolved_bins[:-1]) / 2

        try:
            curve_params, _ = curve_fit(
                lambda x, a, b: a * np.exp(b * x), xs, average_spike_counts
            )
        except RuntimeError:
            curve_params = 0.0, 0.0
        fitted_curve = curve_params[0] * np.exp(curve_params[1] * xs)

        spike_counts_df = pd.DataFrame(
            {
                "generator signal values": xs,
                "average spike count / bin": average_spike_counts,
                "fitted exponential curve": fitted_curve,
            }
        )
        sns.lineplot(
            x="generator signal values",
            y="average spike count / bin",
            data=spike_counts_df,
            ax=ax.flatten()[sf],
            marker="o",
            label="est. non-lin.",
        )
        sns.lineplot(
            x="generator signal values",
            y="fitted exponential curve",
            data=spike_counts_df,
            ax=ax.flatten()[sf],
            ls="--",
            color="grey",
            label=f"exp. fit: {curve_params[0]:.2f}, {curve_params[1]:.2f}",
        )

        ax.flatten()[sf].set_ylim(
            [-0.01 * average_spike_counts.max(), 1.05 * average_spike_counts.max()]
        )
        ax.flatten()[sf].set_xlim(
            [convolved_signal.min() - 0.5, convolved_signal.max() + 0.5]
        )
        ax.flatten()[sf].set_title(f"{spike_file.name.split('.')[0]}")
        ax.flatten()[sf].legend(loc="upper left")

        inset_ax = mpl_il.inset_axes(
            ax.flatten()[sf], width="25%", height="25%", loc="lower right"
        )
        inset_ax.plot(est_filter, color="tab:orange")
        inset_ax.xaxis.tick_top()
        inset_ax.set(xticklabels=[], yticklabels=[])

    fig.suptitle(
        f"Estimating non-linearities for each cell in {pulse_file.name.split('.')[0]}\n(corresp. filter kernels shown "
        f"in inset)",
        size=24,
    )

    return fig


def convolve_stimulus_with_kernels(
    stimulus, spat_kern, temp_kern, total_trials, sta_x, sta_y, sta_window
):
    convolved_response = np.zeros(
        (total_trials, stimulus.shape[1] - temp_kern.size + 1)
    )
    for tr, trial in tqdm(
        enumerate(stimulus[:total_trials]), total=total_trials, desc="Trials"
    ):
        trial = trial[
            :,
            sta_y - sta_window : sta_y + sta_window,
            sta_x - sta_window : sta_x + sta_window,
        ]
        spat_conv = (
            spat_kern * trial.reshape((trial.shape[0], trial.shape[1] * trial.shape[2]))
        ).mean(axis=-1)
        convolved_response[tr] = np.convolve(spat_conv, temp_kern, mode="valid")
    return convolved_response


def convolve_stimulus_with_kernels_for_sc(
    stimulus, spat_kern, temp_kern, total_trials, sta_x, sta_y, sta_window
):
    convolved_response_i_mean = np.zeros(
        (total_trials, stimulus.shape[1] - temp_kern.size + 1)
    )
    convolved_response_lsc = np.zeros(
        (total_trials, stimulus.shape[1] - temp_kern.size + 1)
    )
    for tr, trial in tqdm(
        enumerate(stimulus[:total_trials]), total=total_trials, desc="Trials"
    ):
        trial = trial[
            :,
            sta_y - sta_window : sta_y + sta_window,
            sta_x - sta_window : sta_x + sta_window,
        ]
        spat_conv = (
            spat_kern * trial.reshape((trial.shape[0], trial.shape[1] * trial.shape[2]))
        ).sum(axis=-1)
        loc_spat_con = np.sqrt(
            (
                (
                    spat_kern
                    * trial.reshape((trial.shape[0], trial.shape[1] * trial.shape[2]))
                    - spat_conv
                )
                ** 2
            ).sum(axis=-1)
            / (spat_kern.size - 1)
        )
        convolved_response_i_mean[tr] = np.convolve(spat_conv, temp_kern, mode="valid")
        convolved_response_lsc[tr] = np.convolve(loc_spat_con, temp_kern, mode="valid")
    return convolved_response_i_mean, convolved_response_lsc


def get_predicted_response(
    convolved_signal, conv_sig_values, avg_resp_counts, method="average", fit_func=None
):
    if not conv_sig_values.size == avg_resp_counts.size:
        raise AttributeError(
            "Convolved signal values and response counts should be of the same length"
        )
    dig_conv_sig = np.digitize(convolved_signal, conv_sig_values)
    if method == "greater":
        return np.array([avg_resp_counts[i] for i in dig_conv_sig])
    elif method == "lesser":
        return np.array([avg_resp_counts[i - 1] for i in dig_conv_sig])
    elif method == "average":
        return np.array(
            [(avg_resp_counts[i] + avg_resp_counts[i - 1]) / 2 for i in dig_conv_sig]
        )
    elif method == "fit":
        if fit_func is not None:
            params, _ = curve_fit(fit_func, conv_sig_values, avg_resp_counts)
            return fit_func(convolved_signal, *params)
        else:
            raise ValueError('fit_func must be provided for the "fit" method!')
    else:
        raise ValueError(
            "Invalid method requested! Choose between greater, lesser and average."
        )


def dimos_model(x, b, m, r, g):
    first = (m - b) / (1 + np.exp(-r * (x - g)))
    return b + first


def get_spat_temp_kern(sta, mis_pos, sta_window):
    t_, y_, x_ = mis_pos
    mis_val = sta[mis_pos]

    # np.sign corrects for a negative or positive MIS
    spat_kern = (
        np.sign(mis_val)
        * sta[
            t_, y_ - sta_window : y_ + sta_window, x_ - sta_window : x_ + sta_window
        ].flatten()
    )
    spat_kern /= np.linalg.norm(spat_kern)

    temp_kern = sta[:, mis_pos[1], mis_pos[2]].flatten()[::-1]
    temp_kern /= np.linalg.norm(temp_kern)

    return spat_kern, temp_kern


def gauss_2d(x, a, b, mu, sigma):
    exp = -(x - mu).T @ np.linalg.inv(sigma) @ (x - mu) / 2
    out = a * 1 / (2 * np.pi * np.sqrt(np.abs(sigma))) * np.exp(exp) + b

    return out
