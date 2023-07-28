import numpy as np
import os
import pickle
from tqdm import tqdm
from PIL import Image
from matplotlib import animation
import torch
#from dynamic.utils.global_functions import home
import matplotlib.pyplot as plt
import math

# from line_profiler_pycharm import profile


# @profile
def crop_based_on_fixation(
    img, x_center, y_center, img_h, img_w, flip=False, padding=200
):
    if flip:
        img = torch.fliplr(img)
    x_center += padding
    y_center += padding
    img = img[
        x_center - int(img_w / 2) : x_center + int(img_w / 2),
        y_center - int(img_h / 2) : y_center + int(img_h / 2),
    ]
    assert img.shape == (img_w, img_h)

    return img  # , min(left, right, top, bottom)


def read_file(file):
    f = open(file)
    return f


def read_line(line):
    line = line.split(" ")
    img_index = int(line[0].split(".")[0])
    x_center = int(line[3])
    y_center = int(line[4])
    flipped = int(line[-1])
    return img_index, x_center, y_center, flipped


def get_img(img, directory, show=False):
    # file = [file for file in files if file.startswith(f'{str(img_index).zfill(5)}_img')]
    # assert len(file) == 1

    img = np.fromfile(os.path.join(directory, img), dtype=np.uint8)
    img = img.reshape((1000, 800))
    # if show:
    #     plt.imshow(img)
    return img


def read_files():
    test_files = os.listdir(
        "/data/marmoset_data/cnn_tos_movie_seed2022/repeated_stimuli/"
    )
    train_files = os.listdir(
        "/data/marmoset_data/cnn_tos_movie_seed2022/non_repeating_stimuli/"
    )
    return test_files, train_files


def create_dataset(test_files, train_files, fixations, img_h, img_w):
    trial = 0
    test_done = False
    test_array = np.zeros((img_h, img_w, 5067))
    train_array = np.zeros((img_h, img_w, 2837))
    test_index = 0
    train_index = 0
    for line in tqdm(fixations.readlines()):
        img_index, x_center, y_center, flipped = read_line(line)
        # print(f'line {i}', f'img_index: {img_index}, train_index: {train_index}, test_index: {test_index}')

        if img_index == 0:
            print(line, trial)
        if test_index == 5067:
            # if not test_done:
            # np.save(
            #     f'/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/repeating_stimuli/all_images.npy',
            #     test_array)
            # np.save(
            #     f'/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/repeating_stimuli/all_images2.npy',
            #     test_array[2550:])
            test_done = True
            test_index = -1
            train_index = 0
            test_array = None
        if train_index == 2837:
            if trial >= 8:
                print(f"saving for trial {trial}")
                np.save(
                    f"/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/non_repeating_stimuli/trial_{str(trial).zfill(3)}.npy",
                    train_array,
                )
            trial += 1
            train_index = 0

        if test_index != -1:
            img = get_img(
                img_index,
                test_files,
                "/data/marmoset_data/cnn_tos_movie_seed2022/repeated_stimuli/",
            )
            img = crop_based_on_fixation(img, (x_center, y_center), img_h, img_w)
            test_array[:, :, test_index] = img.transpose()
            test_index += 1
        else:
            img = get_img(
                img_index,
                train_files,
                "/data/marmoset_data/cnn_tos_movie_seed2022/non_repeating_stimuli/",
            )
            img = crop_based_on_fixation(img, (x_center, y_center), img_h, img_w)
            train_array[:, :, train_index] = img.transpose()
            train_index += 1

    np.save(
        f"/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/non_repeating_stimuli/trial{str(trial).zfill(3)}.npy",
        train_array,
    )


def visualize_dataset(trial_file, save_file):
    images = np.load(trial_file)
    print(images.shape)
    frames = []
    fig, ax = plt.subplots(layout="tight")
    ax.set_axis_off()
    for img in tqdm(range(images.shape[2])):
        frames.append(
            [ax.imshow(images[:, :, img], cmap="gray", vmin=-1, vmax=1, animated=True)]
        )
    anim = animation.ArtistAnimation(fig, frames, interval=11.8, blit=True)

    print("saving animation")
    anim.save(f"{save_file}.mp4")


def save_subsampled_dataset(
    dataset_path, dataset_name, new_dataset_name, crop_size=200, resize_ratio=1 / 4
):
    image_list = os.listdir(os.path.join(dataset_path, dataset_name))
    for image_name in tqdm(image_list):
        img = np.load(os.path.join(dataset_path, dataset_name, image_name))
        img = img[crop_size:-crop_size, crop_size:-crop_size]
        new_size = int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)
        img = Image.fromarray((img * 255).astype(np.float32))
        img = img.resize(new_size, Image.LANCZOS)
        img = np.asarray(img)
        new_padding = int(crop_size * resize_ratio)
        padded = np.zeros(
            (img.shape[0] + new_padding * 2, img.shape[1] + new_padding * 2)
        )
        padded[new_padding:-new_padding, new_padding:-new_padding] = img
        # plt.imshow(padded, cmap='gray')
        # plt.show()
        np.save(os.path.join(dataset_path, new_dataset_name, image_name), padded)


def change_fixation_file_based_on_resizing_ratio(
    file_dir, fixation_file, new_fixation_file, resize_ratio
):
    with open(os.path.join(file_dir, fixation_file), "r") as old_fixations:
        fixations = old_fixations.readlines()
    with open(os.path.join(file_dir, new_fixation_file), "w") as new_fixations:
        for line in tqdm(fixations):
            img_index, x_center, y_center, flipped = read_line(line)
            new_x_center = int(x_center * resize_ratio)
            new_y_center = int(y_center * resize_ratio)
            new_line = " ".join(
                [
                    str(img_index),
                    "0.0",
                    "11.8",
                    str(new_x_center),
                    str(new_y_center),
                    f"{flipped}\n",
                ]
            )
            new_fixations.write(new_line)
    new_fixations.close()


if __name__ == "__main__":
    # visualize_dataset('/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/repeating_stimuli/all_images.npy',  '/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/test_set_seed2022.mp4')
    # visualize_dataset('/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/non_repeating_stimuli/trial_000/all_images.npy',  '/usr/users/vystrcilova/retinal_circuit_modeling/datasets/visualization_plots/trial.mp4')
    change_fixation_file_based_on_resizing_ratio(
        "/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/fixations",
        "complete_fixations_seed2022.txt",
        "complete_fixations_seed2022_s2.txt",
        resize_ratio=1 / 2,
    )

    # save_subsampled_dataset('/usr/users/vystrcilova/retinal_circuit_modeling/data/marmoset_data/',
    #                         dataset_name='stimuli_padded/',
    #                         new_dataset_name='stimuli_padded_2/', resize_ratio=1/2, crop_size=200)
    #
    exit()
    test_files, train_files = read_files()
    fixations = read_file("/usr/users/vystrcilova/marmoset_data/fixations.txt")
    create_dataset(test_files, train_files, fixations, 600, 800)
