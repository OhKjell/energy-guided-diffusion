import os

import numpy as np


def save_array_per_trial(dir, num_of_imgs=1500, img_shape=(150, 200)):
    array = np.zeros(img_shape + (num_of_imgs,))
    for img in range(num_of_imgs):
        file = np.load(
            os.path.join(dir, f"{str(img).zfill(len(str(num_of_imgs)))}.npy")
        )
        array[:, :, img] = file
    np.save(os.path.join(dir, "all_images"), array)


if __name__ == "__main__":
    basepath = "/usr/users/vystrcilova/retinal_circuit_modeling"
    data_dir = os.path.join(basepath, "data/non_repeating_stimuli/")
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            print(os.path.join(root, dir))
            save_array_per_trial(os.path.join(root, dir))
