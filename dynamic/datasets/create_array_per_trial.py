import numpy as np
import os


def save_array_per_trial(dir, num_of_imgs=1500, img_shape=(150, 200)):
    """
    Transforms data where each image is saved in a separate .npy file into a single .npy file for each trial
    :param dir:
    :param num_of_imgs:
    :param img_shape:
    :return:
    """
    array = np.zeros(img_shape + (num_of_imgs,))
    for img in range(num_of_imgs):
        file = np.load(
            os.path.join(dir, f"{str(img).zfill(len(str(num_of_imgs)))}.npy")
        )
        array[:, :, img] = file
    np.save(os.path.join(dir, "all_images"), array)


if __name__ == "__main__":
    basepath = "/Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling"
    data_dir = os.path.join(basepath, "data/non_repeating_stimuli/")
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            print(os.path.join(root, dir))
            save_array_per_trial(os.path.join(root, dir))
