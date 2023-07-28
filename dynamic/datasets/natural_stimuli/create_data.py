import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib.request
from matplotlib import animation
import seaborn as sns

SEED = 2022
rng = np.random.default_rng(SEED)

REFERENCE_IMAGE_SIZE = [1532, 1024]
SINGLE_IMAGE_SIZE = [1000, 800]
SINGLE_IMAGE_CROP_SIZE = [960, 800]
STIMULATOR_SCREEN_SIZE = [800, 600]

TOTAL_IMAGES = 14544
MOVIE_START_IMAGE_IDX = 215
MOVIE_END_IMAGE_IDX = 17528

# movie begins from frame 215
# movie ends at frame 14112
# credits begin at frame 14136
# credits end at frame 16990
# post-credits scene begins at 17029
# post-credits scene ends at 17528
#    -->  num of available frames 16 776, when split in half - 33 550

# We need 515 100 frames for two hours of recordings. 100x60x85 (min x sec x framerate) = 510 000 for the running
# stimuli and 1x60x85 = 5 100 for the control stimuli which will be repated 20 times this is 66 660 times an image
# for three frames and 78 780 times and image for four frames i.e. total 145 440

frames_required = 515100
running_frames = 510000
frozen_frames = 5100

# therefore, we need approximately 145 440/ 33 550 = 4.3 different coordinate starts
# We take 14 544 images and create fixations for them. Then we use them for the other half and also for the
# 4 other different shifts

CONTROL_START = 10088
CONTROL_END = 11522

# The test set will contain 1434 images
num_of_test_images = CONTROL_END - CONTROL_START

THREE_FRAMES_IMAGES = 6666  # 0.458333 probability
FOUR_FRAMED_IMAGES = 7878

PIXELS_FROM_EDGE = 200
PIXELS_PER_DEGREE = 37.5
MICROMETERS_PER_DEGREE = 100
# MICROMETERS_PER_PIXEL_PROJECTOR = 7.5  # for most setups
MICROMETERS_PER_PIXEL_PROJECTOR = 6.0  # for CMOS
# MICROMETERS_PER_PIXEL_PROJECTOR = 2.5  # for Saruman

REFRESH_RATE = 85  # Hz, for marmoset
TOTAL_DURATION = 3600 * REFRESH_RATE  # 1 hour, in frames
CHUNK_DURATION = 10 * REFRESH_RATE  # 10 seconds, in frames

NUMBER_OF_FIXATIONS = 61510  # number of fixations
SCALE_FIXATIONS = 200  # scale parameter for exponential distribution (milliseconds)
ALPHA_FIXATIONS = 100  # refractory period in milliseconds
JITTER_MEAN = 0.0
JITTER_STD = 2.0

SCALE_SACCADES = 5  # scale for the exponential distribution for amplitude (in degrees)
DURATION_VALUES_SACCADES = (2, 3, 4)  # duration values to sample from (in frames)
DURATION_PROBABILITIES_SACCADES = (
    0.35,
    0.4,
    0.25,
)  # relative probabilities of the different duration
# values (hand-wavy; should add up to 1.0)

# Paths
FINAL_TRACES_OUTPUT_FOLDER = pathlib.Path(
    "/mnt/storage/data/movies/natural_stimuli/stimulus_folder/"
)
LOCAL_IMAGES_FOLDER = pathlib.Path("/mnt/storage/data/movies/tears_of_steel/pngs/")

REMOTE_IMAGES_FOLDER = "https://media.xiph.org/tearsofsteel/tearsofsteel-1080-png/"

ANIMATION_SAVE_PATH = pathlib.Path(
    "/mnt/storage/data/movies/natural_stimuli/generated_fixations_test.mp4"
)
SOURCE_FIXATIONS_FILE_PATH = pathlib.Path(
    "/home/shashwat/work/marmoset_experiments/marmoset_movie_347im_85Hz/fixations.txt"
)

RAW_IMAGES_SAVE_FOLDER = pathlib.Path(
    "/mnt/storage/data/movies/natural_stimuli/stimulus_folder/images_trial/"
)

SANITY_PLOTS_FOLDER = pathlib.Path("/home/shashwat/work/movie_stimulus_marmoset/images")

PNG_FILENAME_STUB = "graded_edit_final_"
OUTPUT_FIXATIONS_FILENAME_STUB = "fixations_trial_"


def generate_fixation_durations(
    size=NUMBER_OF_FIXATIONS, scale=SCALE_FIXATIONS, alpha=ALPHA_FIXATIONS
):
    fixations_sample = rng.exponential(
        scale=scale, size=size
    )  # draw random sample from exp. dist.
    fixations_sample = fixations_sample[
        fixations_sample >= alpha
    ]  # enforce refractory period requirement
    return fixations_sample


def generate_saccades(
    size=NUMBER_OF_FIXATIONS,
    scale=SCALE_SACCADES,
    duration_values=DURATION_VALUES_SACCADES,
    duration_probabilities=DURATION_PROBABILITIES_SACCADES,
):
    saccade_amplitudes = rng.exponential(scale=scale, size=size)
    saccade_durations = rng.choice(
        duration_values, size=size, replace=True, p=duration_probabilities
    )
    saccade_directions = rng.choice(np.arange(360), size=size, replace=True)
    return saccade_amplitudes, saccade_durations, saccade_directions


def generate_single_fixation(scale=SCALE_FIXATIONS, alpha=ALPHA_FIXATIONS):
    fixation_sample = rng.exponential(scale=scale)  # draw random sample from exp. dist.
    while fixation_sample < alpha:  # enforce refractory period requirement
        fixation_sample = rng.exponential(scale=scale)
    return fixation_sample


def generate_single_saccade(
    scale=SCALE_SACCADES,
    duration_values=DURATION_VALUES_SACCADES,
    duration_probabilities=DURATION_PROBABILITIES_SACCADES,
):
    saccade_amplitude = rng.exponential(scale=scale)
    saccade_duration = rng.choice(
        duration_values, replace=True, p=duration_probabilities
    )
    saccade_direction = rng.choice(np.arange(360), replace=True)
    return saccade_amplitude, saccade_duration, saccade_direction


def generate_saccades_and_fixations():
    x_values, y_values = [], []
    amps, durs, angs = [], [], []
    fix_durs = []

    movement_id = 0
    while len(x_values) < TOTAL_DURATION and len(y_values) < TOTAL_DURATION:
        x_chunk, y_chunk = [0.0], [0.0]

        while len(x_chunk) < CHUNK_DURATION and len(y_chunk) < CHUNK_DURATION:
            fix_dur = generate_single_fixation()
            amp, dur, ang = generate_single_saccade()

            fix_durs.append(fix_dur)
            amps.append(amp)
            durs.append(dur)
            angs.append(ang)

            # add fixed frames after jittering them
            add_frames = int(np.floor(fix_dur * REFRESH_RATE / 1000))
            x_fix_frames = jitter_fixations(np.array([x_chunk[-1]] * add_frames))
            y_fix_frames = jitter_fixations(np.array([y_chunk[-1]] * add_frames))
            x_chunk.extend(x_fix_frames)
            y_chunk.extend(y_fix_frames)

            # add saccadic frames
            ang_rad = ang * (np.pi / 180)  # saccade direction in radians
            x_increment = (
                amp * np.cos(ang_rad) * MICROMETERS_PER_DEGREE
            )  # increment in x direction in micrometers
            mx = x_increment / dur  # x slope (um / frame)
            for i in range(dur):
                x_chunk.append(interp(i, mx, x_chunk[-1]))

            y_increment = (
                amp * np.sin(ang_rad) * MICROMETERS_PER_DEGREE
            )  # increment in y direction in micrometers
            my = y_increment / dur  # y slope (um / frame)}")
            for i in range(dur):
                y_chunk.append(interp(i, my, y_chunk[-1]))

            movement_id += 1

        if values_make_sense(x_chunk) and values_make_sense(y_chunk):
            x_values.extend(x_chunk)
            y_values.extend(y_chunk)

    x_values = np.array(x_values) / MICROMETERS_PER_PIXEL_PROJECTOR
    y_values = np.array(y_values) / MICROMETERS_PER_PIXEL_PROJECTOR

    return (
        x_values,
        y_values,
        np.array(fix_durs),
        np.array(amps),
        np.array(durs),
        np.array(angs),
    )


def values_make_sense(chunk_list):
    if np.amax(np.abs(chunk_list)) < PIXELS_FROM_EDGE * MICROMETERS_PER_PIXEL_PROJECTOR:
        return True
    else:
        return False


def interp(val, slope, intercept):
    return slope * val + intercept


def jitter_fixations(fixation_array):
    jitters = rng.normal(loc=JITTER_MEAN, scale=JITTER_STD, size=fixation_array.size)
    return fixation_array + jitters


def convert_to_gray_scale(img, b_coef=0.0722, g_coef=0.7152, r_coef=0.2126, plot=False):
    gray_scale_img = (
        b_coef * img[:, :, 0] + g_coef * img[:, :, 1] + r_coef * img[:, :, 2]
    )
    if plot:
        plt.imshow(gray_scale_img, cmap="gray")
        plt.show()
    return gray_scale_img


def crop_to_single_image_size(image):
    x_diff = (
        abs(SINGLE_IMAGE_SIZE[0] - image.shape[0]) // 2
        if SINGLE_IMAGE_SIZE[0] != image.shape[0]
        else None
    )
    y_diff = (
        abs(SINGLE_IMAGE_SIZE[1] - image.shape[1]) // 2
        if SINGLE_IMAGE_SIZE[1] != image.shape[1]
        else None
    )

    if x_diff is not None and y_diff is not None:
        cropped_image = image[x_diff:-x_diff, y_diff:-y_diff]
    elif x_diff is not None:
        cropped_image = image[x_diff:-x_diff]
    elif y_diff is not None:
        cropped_image = image[:, y_diff:-y_diff]
    else:
        cropped_image = image
    return cropped_image


def get_image(img_index):
    local_path = (
        LOCAL_IMAGES_FOLDER / f"{PNG_FILENAME_STUB}{str(img_index).zfill(5)}.png"
    )
    if local_path.exists():
        image = get_local_image(local_path)
    else:
        image = download_image(img_index)

    return image


def download_image(image_index):
    req = urllib.request.urlopen(
        REMOTE_IMAGES_FOLDER + PNG_FILENAME_STUB + f"{str(image_index).zfill(5)}.png"
    )
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv.imdecode(arr, cv.IMREAD_COLOR)
    return image


def get_local_image(image_path):
    image = np.asarray(Image.open(image_path))
    return image


def get_stimuli(img_index):
    img = get_image(img_index)
    img = convert_to_gray_scale(img)
    img = img.T
    img = crop_to_single_image_size(img)
    return img


def save_fixations_to_file(file_dir, img_indices, fixations):
    with open(f"{file_dir}/fixations.txt", "w") as fixation_file:
        for img_index, fixation in zip(img_indices, fixations):
            line = f"{img_index:.02f} {0.000:.2f} {11.8:.1f} {fixation[0]:.0f} {fixation[2]:.0f}\n"
            fixation_file.write(line)
    fixation_file.close()


def center_and_shift_single_trace(trace, center_along=0, shift_by=0.0):
    centering = SINGLE_IMAGE_CROP_SIZE[center_along] // 2 - trace.mean()
    change = centering + shift_by
    trace += change
    return trace


def generate_eye_traces(shifts, sanity_plots=False):
    traces = []
    fixdurs, amps, durs, angs = [], [], [], []
    for n, shift_by in enumerate(shifts):
        x_pixels, y_pixels, fixdur, amp, dur, ang = generate_saccades_and_fixations()
        fixdurs.append(fixdur)
        amps.append(amp)
        durs.append(dur)
        angs.append(ang)

        x_pixels = center_and_shift_single_trace(
            x_pixels, center_along=0, shift_by=shift_by[0]
        )
        y_pixels = center_and_shift_single_trace(
            y_pixels, center_along=1, shift_by=shift_by[1]
        )

        traces.append(np.vstack([np.round(x_pixels), np.round(y_pixels)]).T)

    if sanity_plots:
        plot_trace_statistics(fixdurs, amps, durs, angs)
        plot_traces(traces)
        plot_fixations_on_screen(traces)

    return traces


def plot_trace_statistics(fixation_durations, amplitudes, durations, angles):
    for i, (fixdurs, amps, durs, angs) in enumerate(
        zip(fixation_durations, amplitudes, durations, angles)
    ):
        with sns.plotting_context("talk"), sns.axes_style("white"):
            fig, axes = plt.subplots(ncols=4, figsize=(15, 4), layout="constrained")

            sns.histplot(x=fixdurs, bins=100, ax=axes[0])
            axes[0].set_title("fixation durations")
            axes[0].set_xlabel("milliseconds")
            axes[0].set_ylabel("")
            axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            sns.histplot(x=amps, bins=100, ax=axes[1])
            axes[1].set_title("saccade amplitudes")
            axes[1].set_xlabel("degrees")
            axes[1].set_ylabel("")
            axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            sns.histplot(durs, ax=axes[2])
            axes[2].set_title("saccade durations")
            axes[2].set_xlabel("frames")
            axes[2].set_ylabel("")
            axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            sns.histplot(angs, bins=360, ax=axes[3])
            axes[3].set_title("saccade directions")
            axes[3].set_xlabel("degrees")
            axes[3].set_ylabel("")

            plt.savefig(SANITY_PLOTS_FOLDER / f"set_{i}_sanity_plots.png", dpi=300)
            plt.show()


def plot_traces(traces_list):
    for t, traces in enumerate(traces_list):
        with sns.plotting_context("talk"), sns.axes_style("ticks"):
            fig, axes = plt.subplots(
                nrows=2,
                figsize=(15, 6),
                sharex="all",
                sharey="all",
                layout="constrained",
            )
            plot_start = 0
            plot_only = 1000

            # convert x axis to seconds
            x_ax = (
                np.arange(traces[:, 0][plot_start : plot_start + plot_only].size)
                / REFRESH_RATE
            )
            sns.lineplot(
                x=x_ax,
                y=traces[:, 0][plot_start : plot_start + plot_only],
                lw=2,
                ax=axes[0],
            )
            sns.lineplot(
                x=x_ax,
                y=traces[:, 1][plot_start : plot_start + plot_only],
                lw=2,
                ax=axes[1],
            )
            axes[0].set_ylabel("pixels (x axis)")
            axes[1].set_ylabel("pixels (y axis)")
            sns.despine(ax=axes[0])
            sns.despine(ax=axes[1])
            plt.xlabel("seconds")
            plt.suptitle("sample traces")
            plt.savefig(SANITY_PLOTS_FOLDER / f"set_{t}_sample_traces.png", dpi=300)
            plt.show()


def plot_fixations_on_screen(traces_list):
    for t, traces in enumerate(traces_list):
        with sns.plotting_context("talk"), sns.axes_style("ticks"):
            fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
            sns.histplot(
                x=traces[:, 0],
                y=traces[:, 1],
                ax=ax,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Fixations on screen")
            plt.savefig(
                SANITY_PLOTS_FOLDER / f"set_{t}_fixation_distribution.png", dpi=300
            )
            plt.show()


def get_image_indices(total_images=TOTAL_IMAGES, movie_start_id=MOVIE_START_IMAGE_IDX):
    three_repeats = THREE_FRAMES_IMAGES
    four_repeats = FOUR_FRAMED_IMAGES

    three_probability = three_repeats / (three_repeats + four_repeats)
    four_probability = (
        1 - three_probability
    )  # four_repeats / (three_repeats + four_repeats)

    assignments = rng.choice(
        [
            3,
            4,
        ],
        p=[
            three_probability,
            four_probability,
        ],
        size=total_images,
        replace=True,
    )

    image_indices = []

    for img_id, repeats in enumerate(assignments):
        image_indices.extend([img_id + movie_start_id] * repeats)

    return image_indices


def create_movie_animation(image_indices, fixations):
    frames = []
    fig, ax = plt.subplots(layout="tight")
    ax.set_axis_off()
    prev_img_index = None
    image = None
    for i, (fixation, img_index) in enumerate(zip(fixations, image_indices)):
        if img_index != prev_img_index:
            image = get_stimuli(img_index)
            prev_img_index = img_index
        else:
            image = image
        frames.append(
            [
                ax.imshow(image.T, cmap="gray", animated=True),
                ax.scatter([fixation[0]], [fixation[1]], color="lime"),
            ]
        )
    print("creating animation")

    anim = animation.ArtistAnimation(fig, frames, interval=11.8, blit=True)

    print("saving animation")
    anim.save(ANIMATION_SAVE_PATH.as_posix())


def save_traces_to_file(
    traces_list,
    image_indices,
    control_indices,
    url_to_file_index_dict,
    file_dir,
    fix_stub,
):
    central_traces = traces_list[0]

    with open(f"{file_dir}/{fix_stub}seed{SEED}.txt", "w") as fixation_file:
        for index in control_indices:
            fixation_file.write(
                f"{url_to_file_index_dict[image_indices[index]]:.1f} "
                f"{0.0:.1f} "
                f"{11.8:.1f} "
                f"{central_traces[index][0]:.0f} "
                f"{central_traces[index][1]:.0f} {1}\n"
            )

        for t, trace in enumerate(traces_list):
            for j, image_index in enumerate(image_indices):
                if (t == 0) and (j in control_indices):
                    continue
                else:
                    fixation_file.write(
                        f"{url_to_file_index_dict[image_index]:.1f} "
                        f"{0.0:.1f} "
                        f"{11.8:.1f} "
                        f"{trace[j][0]:.0f} "
                        f"{trace[j][1]:.0f} "
                        f"{(t + 1) % 2}\n"
                    )


def save_images(
    url_to_file_indices,
    image_start_index=MOVIE_START_IMAGE_IDX,
    num_of_images=TOTAL_IMAGES,
):
    for i, image_idx in enumerate(
        range(image_start_index, image_start_index + num_of_images)
    ):
        img = get_stimuli(image_idx)
        img.astype(np.uint8).tofile(
            RAW_IMAGES_SAVE_FOLDER
            / f"{str(url_to_file_indices[image_idx]).zfill(5)}_img_{str(image_idx).zfill(5)}.raw"
        )


def get_img_index_to_file_index_translation(
    movie_start_id=MOVIE_START_IMAGE_IDX,
    total_images=TOTAL_IMAGES,
    control_start=CONTROL_START,
    control_end=CONTROL_END,
):
    index = 0
    url_to_file_index_dict = {}
    file_to_url_index_dict = {}

    for url_index in range(control_start, control_end):
        url_to_file_index_dict[url_index] = index
        file_to_url_index_dict[index] = url_index
        index += 1

    for url_index in range(movie_start_id, movie_start_id + total_images + 1):
        if url_index in range(CONTROL_START, CONTROL_END):
            continue
        else:
            url_to_file_index_dict[url_index] = index
            file_to_url_index_dict[index] = url_index
            index += 1

    assert len(url_to_file_index_dict.values()) - 1 == np.max(
        list(url_to_file_index_dict.values())
    )
    return url_to_file_index_dict, file_to_url_index_dict


if __name__ == "__main__":
    url_to_file, file_to_url = get_img_index_to_file_index_translation()

    image_ids = get_image_indices()

    control_ids, num_of_control = [], []
    for x, x_id in enumerate(image_ids):
        if x_id in range(CONTROL_START, CONTROL_END):
            control_ids.append(x)
            num_of_control.append(x_id)

    shift = 50
    shift_list = [
        (0.0, 0.0),
        (shift, -shift),
        (-shift, shift),
        (shift, shift),
        (-shift, -shift),
    ]

    print("Generating traces...")
    all_traces = generate_eye_traces(shift_list, sanity_plots=False)

    print("Generating sample visualization...")
    indices_to_plot = image_ids[control_ids[0] : control_ids[0] + len(num_of_control)]
    fixations_to_plot = all_traces[0][
        control_ids[0] : control_ids[0] + len(num_of_control)
    ]
    create_movie_animation(indices_to_plot, fixations_to_plot)

    save_traces_to_file(
        traces_list=all_traces,
        image_indices=image_ids,
        control_indices=control_ids,
        file_dir=FINAL_TRACES_OUTPUT_FOLDER,
        url_to_file_index_dict=url_to_file,
        fix_stub=OUTPUT_FIXATIONS_FILENAME_STUB,
    )

    save_images(url_to_file_indices=url_to_file)
