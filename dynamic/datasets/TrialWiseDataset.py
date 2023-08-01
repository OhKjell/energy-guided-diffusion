from pathlib import Path
import numpy as np
import os, random
from functools import lru_cache
import torch
import pickle
from dynamic.utils.global_functions import dataset_seed, set_random_seed
from collections import namedtuple
default_image_datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])


class TrialWiseDataset:
    def __init__(
        self,
        responses: dict,
        dir,
        *data_keys: list,
        indices: list,
        use_cache: bool = True,
        trial_prefix: str = "trial",
        test: bool = False,
        cache_maxsize: int = 20,
        crop: int or tuple = 20,
        subsample: int = 1,
        movie_like: bool = False,
        num_of_frames: int = None,
        num_of_layers: int = None,
        device: str = "cpu",
        conv3d: bool = False,
        time_chunk_size: int = None,
        single_prediction: bool = False,
        temporal_dilation=1,
        hidden_temporal_dilation=1,
        num_of_hidden_frames: int = None,
        extra_frame=0,
    ):
        """
        Dataset for the following (example) file structure:
         ├── data
         │   ├── non-repeating stimuli [directory with as many files as trials]
                   |     |-- trial_000
                   |     |     |-- all_images.npy
                   |     |-- trial 001
                   |     |     |-- all_images.npy
                   |     ...
                   |     |-- trial 247
                   |     |     |-- all_images.npy
         │   ├── repeating stimuli [directory with 1 file for test]
                         |-- all_images.npy
         │   ├── responses [directory with as many files as retinas]

        :param responses: dictionary containing train set responses under the key 'train_responses'
                          and test responses under the key 'test_responses'
                          :expected train set response shape: cells x num_of_images x num_of_trials
                          expected test set response shape: cells x num_of_images (i.e. test trials averaged before)
        :param dir: path to directory where images are stored.
                    Expected format for image files is: f'{dir}/{trial_prefix}_{int representing trial number}.zfill(3)/all_images.npy'
                    Expected shape of numpy array in all_images.npy is height x width x num_of_images in trial
        :param data_keys: list of keys to be used for the datapoints, expected ['images', 'responses']
        :param indices: Indices of the trials selected for the given dataset
        :param transforms: List of transformations that are supposed to be performed on images
        :param use_cache: Whether to use caching when loading image data
        :param trial_prefix: prefix of trial file, followed by '_{trial number}'
        :param test: Whether the data we are loading is test data
        :param cache_maxsize: Maximum number of trials that can be in the cache at a given point
                              Cache is NOT implemented as LRU. The last cached item is always kicked out first.
                              This is because the trials are always iterated through in the same order.
        :param crop: How much to crop the images - top, bottom, left, right
        :param subsample: Whether/how much images should be sub-sampled (1 equals no subsampling)
        :param movie_like: If true, time dimension is interpreted as channel,
                           data points under 'images' are of shape time x width x height
        :param num_of_frames: Indicates how many frames should be used to make one prediction
        :param num_of_layers: Number of expected convolutional layers,
                              used to calculate the shrink in dimensions or padding
        :param device:
        :param conv3d: If true, datapoints under 'images' are of shape 1 x time x width x height
        :param time_chunk_size: Indicates how many predictions should be made at once by the model.
               The 'images' in datapoints are padded accordingly in the temporal dimension with respect to num_of_frames
               and num_of_layers. Only valid if single_prediction is false.
        :param single_prediction: Whether to allow to return more than one time-point in a single datapoint.
        """

        self.use_cache = use_cache
        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_image_datapoint
        if self.use_cache:
            self.cache_maxsize = cache_maxsize
        if isinstance(crop, int):
            crop = (crop, crop, crop, crop)
        self.crop = crop
        self.extra_frame = extra_frame
        self.num_of_layers = num_of_layers
        self.temporal_dilation = temporal_dilation

        self.num_of_frames = num_of_frames
        if num_of_hidden_frames is None:
            self.num_of_hidden_frames = num_of_frames
        else:
            self.num_of_hidden_frames = num_of_hidden_frames

        if hidden_temporal_dilation is None:
            hidden_temporal_dilation = 1

        if isinstance(hidden_temporal_dilation, str):
            hidden_temporal_dilation = int(hidden_temporal_dilation)

        if isinstance(hidden_temporal_dilation, int):
            hidden_temporal_dilation = (hidden_temporal_dilation,) * (
                self.num_of_layers - 1
            )
        if isinstance(self.num_of_hidden_frames, int):
            self.num_of_hidden_frames = (self.num_of_hidden_frames, ) * (self.num_of_layers-1)


        hidden_reach = sum(
            (f - 1) * d for f, d in zip(self.num_of_hidden_frames, hidden_temporal_dilation)
        )

        self.frame_overhead = (
            self.num_of_frames - 1
        ) * self.temporal_dilation + hidden_reach

        if time_chunk_size is not None:
            self.time_chunk_size = time_chunk_size + self.frame_overhead
        self.conv3d = conv3d
        self.overlapping = single_prediction
        self.subsample = subsample
        self.device = device
        self.trial_prefix = trial_prefix
        self.data_keys = data_keys
        self.basepath = Path(dir).absolute()
        if indices is not None:
            self.train_responses = torch.from_numpy(
                responses["train_responses"]
            ).float()
        if self.overlapping:
            assert (num_of_layers is not None) and (num_of_frames is not None)

        self.test_responses = torch.from_numpy(responses["test_responses"]).float()
        self.indices = indices
        # assert self.test_responses.shape[0] == self.train_responses.shape[0]
        # checking if the number of cells is the same
        self.random_indices = np.random.permutation(indices)
        self.n_neurons = self.train_responses.shape[0]
        self.num_of_trials = self.train_responses.shape[2]
        self.num_of_imgs = int(self.train_responses.shape[1])
        # assert self.train_responses.shape[1] % self.trial_parts == 0

        # print('reshaping with order F')
        # self.train_responses = np.reshape(self.train_responses, (self.train_responses.shape[0], self.num_of_imgs,
        #                                                      self.num_of_trials), order='F')
        # Checks if trials are saved in evenly sized files
        if test:
            self.num_of_imgs = self.test_responses.shape[1]
        self.movie_like = movie_like
        if self.movie_like or self.conv3d:
            assert num_of_frames is not None
        assert self.movie_like != self.conv3d
        # time can be either included as channels - movie_like
        # or as a third dimension - conv3d
        self._test = test
        if self._test:
            if movie_like:
                self._len = self.num_of_imgs - self.num_of_frames
            elif conv3d:
                if self.overlapping:
                    self._len = int(np.floor(self.num_of_imgs - self.frame_overhead))
                else:
                    self._len = (
                        int(
                            np.floor(
                                (self.num_of_imgs - self.frame_overhead)
                                / (self.time_chunk_size - self.frame_overhead)
                            )
                        )
                        - self.extra_frame
                    )
            #                     self._len = 1
            else:
                self._len = self.num_of_imgs

        else:
            if movie_like:
                # self._len = len(self.indices) * self.num_of_imgs - (self.num_of_trials * self.num_of_frames)
                self._len = len(self.indices) * (self.num_of_imgs - self.num_of_frames)
            elif conv3d:
                if self.overlapping:
                    self._len = len(self.indices) * int(
                        np.floor(self.num_of_imgs - self.frame_overhead)
                    )
                else:
                    self._len = (
                        int(
                            len(self.indices)
                            * np.floor(
                                (self.num_of_imgs - self.frame_overhead)
                                / (self.time_chunk_size - self.frame_overhead)
                            )
                        )
                        - self.extra_frame
                    )
            else:
                self._len = len(self.indices) * self.num_of_imgs

        self._cache = {data_key: {} for data_key in data_keys}

    def purge_cache(self):
        self._cache = {data_key: {} for data_key in self.data_keys}

    def transform_image(self, images):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        if len(images.shape) == 3:
            h, w, num_of_imgs = images.shape
            images = images[
                self.crop[0] : h - self.crop[1] : self.subsample,
                self.crop[2] : w - self.crop[3] : self.subsample,
                :,
            ]
            return images

        elif len(images.shape) == 4:
            h, w, num_of_imgs = images.shape[:2]
            images = images[
                self.crop[0][0] : h - self.crop[0][1] : self.subsample,
                self.crop[1][0] : w - self.crop[1][1] : self.subsample,
                :,
            ]
            images = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Image shape has to be three dimensional (time as channels) or four dimensional "
                f"(time, with w x h x c). got image shape {images.shape}"
            )
        return images

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self.movie_like:
            x = self.get_sequence_items(item)
            return x
        elif self.conv3d:
            if self.overlapping:
                x = self.get_overlapping_sequences(item)
            else:
                x = self.get_whole_trials(item)
            return x
        # TODO: I think up from here, it is not used and I am not sure what it is actually doing
        trial_index: int = int(np.floor(item / self.num_of_imgs))
        trial_file_index = self.indices[trial_index]
        img_index = item % self.num_of_imgs

        ret = []
        for data_key in self.data_keys:
            if data_key == "images":
                if not self._test:
                    if self.use_cache and trial_file_index in list(
                        self._cache[data_key].keys()
                    ):
                        value = torch.unsqueeze(
                            self._cache[data_key][trial_file_index][:, :, img_index], 0
                        )
                    else:
                        imgs = self.get_trial_file(trial_file_index, data_key=data_key)
                        value = torch.unsqueeze(imgs[:, :, img_index], 0)

                else:
                    if self.use_cache and len(self._cache[data_key]) > 0:
                        value = torch.unsqueeze(
                            self._cache[data_key][:, self.num_of_frames :, img_index], 0
                        )
                    else:
                        value = self.get_trial_file(trial_file_index, data_key=data_key)
                ret.append(value)

            else:
                if not self._test:
                    value = self.train_responses[:, img_index, trial_file_index]
                else:
                    value = self.test_responses[:, img_index]
                ret.append(value)

        x = self.data_point(*ret)

        return x

    def get_whole_trials(self, item):
        """
        Used when self.conv3d is true and self.single_prediction false. Currently, used for CNNs.
        :param item: index of an item
        :return: A datapoint in the form of {
                  'images': shape - 1 x self.time_chunk_size x width x height,
                  'responses': shape - number of neurons x self.time_chunk_size
                  }
                 number of neurons comes inherently from the data structure, no need to specify here
        """
        trial_index = int(
            np.floor(
                item
                / np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )
        )
        trial_file_index = self.indices[trial_index]
        trial_portion = int(
            item
            % np.floor(
                (self.num_of_imgs - self.frame_overhead)
                / (self.time_chunk_size - self.frame_overhead)
            )
        )
        starting_img_index = int(trial_portion * self.time_chunk_size)
        starting_img_index -= trial_portion * self.frame_overhead
        ending_img_index = int(starting_img_index + self.time_chunk_size)
        ret = []
        # print(f'item: {item}')
        # print(f'trial index: {trial_index}, trial file index: {trial_file_index}, trial portion: {trial_portion}')
        # print(f'starting img: {starting_img_index}, ending img: {ending_img_index}')
        # print(f'starting response: {starting_img_index+self.num_of_layers*(self.num_of_frames-1)}, ending response: {ending_img_index}')
        # print()
        for data_key in self.data_keys:
            if data_key == "images":
                value = self.get_trial_portion(
                    trial_file_index, data_key, starting_img_index, ending_img_index
                )
                value = torch.unsqueeze(value, 0)
                # print(value.shape)
            else:
                if not self._test:
                    value = self.train_responses[
                        :,
                        starting_img_index
                        + self.frame_overhead
                        + self.extra_frame : ending_img_index
                        + self.extra_frame,
                        trial_file_index,
                    ]
                else:
                    ### CAREFUL with this + 1
                    value = self.test_responses[
                        :,
                        starting_img_index
                        + self.frame_overhead
                        + self.extra_frame : ending_img_index
                        + self.extra_frame,
                    ]

            ret.append(value)

        x = self.data_point(*ret)

        return x

    def get_sequence_items(self, item):
        """
        Used when self.movie_like is true. Used when loading into an LN model.
        Output shape is self.num_of_frames x height x width after crop.
        :param item: index of item
        :return:  A datapoint in the form of {
                  'images': shape - self.num_of_frames x width x height,
                  'responses': shape - number of neurons x self.num_of_frames
                  }
                 number of neurons comes inherently from the data structure, no need to specify here
        """
        trial_index = int(np.floor(item / (self.num_of_imgs - self.num_of_frames)))
        trial_file_index = self.indices[trial_index]
        starting_img_index = item % (self.num_of_imgs - self.num_of_frames)
        ending_img_index = starting_img_index + self.num_of_frames

        ret = []
        # print(f'item {item}')
        # print(f'trial_index: {trial_index}, trial_file_index: {trial_file_index}')
        # print(f'starting img: {starting_img_index}, ending img: {ending_img_index}')
        for data_key in self.data_keys:
            if data_key == "images":
                value = self.get_trial_portion(
                    trial_file_index, data_key, starting_img_index, ending_img_index
                )
                ret.append(value)

            else:
                if not self._test:
                    value = self.train_responses[
                        :, ending_img_index - 1, trial_file_index
                    ]
                else:
                    value = self.test_responses[:, ending_img_index - 1]
                ret.append(value)

        x = self.data_point(*ret)

        return x

    def get_overlapping_sequences(self, item):
        """
        Used when both self.conv3d and self.single_prediction are true. Currently not used for any model.
        :param item: index of item
        :return: A datapoint in the form of {
                'images': shape - 1 x self.num_of_layers * (self.num_of_frames - 1) + 1 x width x height
                'responses': shape - number of neurons x self.num_of_layers * (self.num_of_frames - 1) + 1
                }
                number of neurons comes inherently from the data structure, no need to specify here
        """
        trial_index = int(np.floor(item / (self.num_of_imgs - self.frame_overhead)))
        trial_file_index = self.indices[trial_index]
        starting_img_index = int(
            item % int(np.floor(self.num_of_imgs - self.frame_overhead))
        )
        ending_img_index = starting_img_index + self.frame_overhead
        # print(f'item {item}')
        # print(
        #     f'item: {item} out of {self._len} \n trial_index: {trial_index}, trial_file_index: {trial_file_index},\n starting_img_index:{starting_img_index}, end_index:{ending_img_index}, response_index: {starting_img_index + self.num_of_frames-1}')
        ret = []
        for data_key in self.data_keys:
            if data_key == "images":
                value = self.get_trial_portion(
                    trial_file_index, data_key, starting_img_index, ending_img_index
                )
                value = torch.unsqueeze(value, 0)
                ret.append(value)
            else:
                if not self._test:
                    value = self.train_responses[
                        :, ending_img_index - 1, trial_file_index
                    ]
                else:
                    value = self.test_responses[:, ending_img_index - 1]
                ret.append(value)

        x = self.data_point(*ret)

        return x

    def get_trial_file(self, trial_file_index, data_key):
        if not self._test:
            # print('loading new file', f'{self.trial_prefix}_{str(trial_file_index).zfill(3)}/all_images.npy')
            imgs = np.load(
                os.path.join(
                    self.basepath,
                    f"{self.trial_prefix}_{str(trial_file_index).zfill(3)}/all_images.npy",
                )
            )
            imgs = torch.from_numpy(imgs).to(self.device).contiguous()
            imgs = self.transform_image(imgs)
            if self.use_cache:
                if len(list(self._cache[data_key].keys())) >= self.cache_maxsize:
                    last_key = list(self._cache[data_key].keys())[-1]
                    # print(f'deleting {last_key} from cache')
                    del self._cache[data_key][last_key]
                self._cache[data_key][trial_file_index] = imgs
                # print(f'loading {trial_file_index} into cache')
            return imgs
        else:
            imgs = np.load(os.path.join(self.basepath, f"all_images.npy"))
            imgs = torch.from_numpy(imgs).to(self.device).contiguous()
            imgs = self.transform_image(imgs)
            if self.use_cache:
                self._cache[data_key] = imgs
            return imgs

    def get_trial_portion(
        self, trial_file_index, data_key, starting_img_index, ending_img_index
    ):
        if not self._test:
            if self.use_cache and trial_file_index in list(
                self._cache[data_key].keys()
            ):
                value = self._cache[data_key][trial_file_index][
                    :, :, starting_img_index:ending_img_index
                ]
            else:
                imgs = self.get_trial_file(trial_file_index, data_key=data_key)
                # print('')
                value = imgs[:, :, starting_img_index:ending_img_index]
            value = value.permute(2, 0, 1)
            # print(value.shape)
        else:
            if self.use_cache and len(self._cache[data_key]) > 0:
                value = self._cache[data_key][:, :, starting_img_index:ending_img_index]
            else:
                imgs = self.get_trial_file(trial_file_index, data_key=data_key)
                value = imgs[:, :, starting_img_index:ending_img_index]
            value = value.permute(2, 0, 1)
        return value
