import os
from pathlib import Path
import torch
import numpy as np
from datasets.natural_stimuli.create_dataset import get_img, crop_based_on_fixation
from tqdm import tqdm
from collections import namedtuple

# from line_profiler_pycharm import profile
default_image_datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])


class FrameWiseDataset:
    def __init__(
            self,
            responses: dict,
            dir,
            *data_keys: list,
            indices: list,
            frames,
            fixations,
            temporal_dilation=1,
            hidden_temporal_dilation=1,
            test_frames: int = 5100,
            train_frames: int = 25500,
            transforms: list = None,
            img_dir_name="stimuli",
            frame_file: str = "_img_",
            test: bool = False,
            crop: int or tuple = 0,
            subsample: int = 1,
            num_of_frames: int = 15,
            num_of_hidden_frames: int = 15,
            num_of_layers: int = None,
            device: str = "cpu",
            time_chunk_size: int = None,
            full_img_w=1000,
            full_img_h=800,
            img_w=800,
            img_h=600,
            padding=200,
    ):
        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_image_datapoint

        if isinstance(crop, int):
            crop = (crop, crop, crop, crop)
        self.crop = crop
        self.temporal_dilation = temporal_dilation
        self.num_of_layers = num_of_layers

        self.img_h = img_h
        self.img_w = img_w
        self.full_img_h = full_img_h
        self.full_img_w = full_img_w
        self.padding = padding

        self.test_frames = test_frames
        self.train_frames = train_frames

        self.num_of_frames = num_of_frames
        if num_of_hidden_frames is None:
            self.num_of_hidden_frames = self.num_of_frames
        else:
            self.num_of_hidden_frames = num_of_hidden_frames

        if isinstance(hidden_temporal_dilation, str):
            hidden_temporal_dilation = int(hidden_temporal_dilation)

        if isinstance(hidden_temporal_dilation, int):
            hidden_temporal_dilation = (hidden_temporal_dilation,) * (
                    self.num_of_layers - 1
            )
        if isinstance(num_of_hidden_frames, int):
            num_of_hidden_frames = (num_of_hidden_frames,) * (self.num_of_layers - 1)

        if self.num_of_layers > 1 is not None:
            hidden_reach = sum(
            (f - 1) * d for f, d in zip(num_of_hidden_frames, hidden_temporal_dilation)
        )
        else:
            hidden_reach = 0

        self.frame_overhead = (
                                      num_of_frames - 1
                              ) * self.temporal_dilation + hidden_reach

        if time_chunk_size is not None:
            self.time_chunk_size = time_chunk_size + self.frame_overhead

        self.subsample = subsample
        self.device = device
        self.basepath = Path(dir).absolute()
        self.img_dir_name = img_dir_name
        self.frame_file = frame_file
        self.response_dict = responses
        if indices is not None:
            self.train_responses = torch.from_numpy(
                responses["train_responses"]
            ).float()
        self.test_responses = torch.from_numpy(responses["test_responses"]).float()

        self.fixations = fixations
        self.indices = indices
        self.frames = frames

        self.random_indices = np.random.permutation(indices)
        self.n_neurons = self.train_responses.shape[0]
        self.num_of_trials = self.train_responses.shape[2]
        self.num_of_imgs = int(self.train_responses.shape[1])

        if test:
            self.num_of_imgs = self.test_responses.shape[1]

        self.cache = []
        self.last_start_index = -1
        self.last_end_index = -1

        self._test = test
        if self._test:
            self._len = int(
                np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )
        else:
            self._len = int(
                len(self.indices)
                * np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )

    def transform_image(self, images):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        if len(images.shape) == 3:
            h, w, num_of_imgs = images.shape
            images = images[
                     self.crop[0]: h - self.crop[1]: self.subsample,
                     self.crop[2]: w - self.crop[3]: self.subsample,
                     :,
                     ]
            return images

        elif len(images.shape) == 4:
            h, w, num_of_imgs = images.shape[:2]
            images = images[
                     self.crop[0][0]: h - self.crop[0][1]: self.subsample,
                     self.crop[1][0]: w - self.crop[1][1]: self.subsample,
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
        trial_index = int(
            np.floor(
                item
                / np.floor(
                    (self.num_of_imgs - self.frame_overhead)
                    / (self.time_chunk_size - self.frame_overhead)
                )
            )
        )
        actual_trial_index = self.indices[trial_index]
        #         print('actual trial index', actual_trial_index)
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
        # print(f'item: {item}')
        # print(f'trial index: {trial_index}, actual trial index: {actual_trial_index}, trial portion: {trial_portion}')
        # print(f'starting img: {starting_img_index}, ending img: {ending_img_index}')
        # print(f'starting response: {starting_img_index + self.frame_overhead}, ending '
        #       f'response: {ending_img_index}')
        ret = []

        for data_key in self.data_keys:
            if data_key == "images":
                value = self.get_frames(
                    actual_trial_index, data_key, starting_img_index, ending_img_index
                )
                value = torch.unsqueeze(value, 0)
            else:
                if not self._test:
                    value = self.train_responses[
                            :,
                            starting_img_index + self.frame_overhead: ending_img_index,
                            actual_trial_index,
                            ]
                else:
                    value = self.test_responses[
                            :, starting_img_index + self.frame_overhead: ending_img_index
                            ]

            ret.append(value)

        x = self.data_point(*ret)

        return x

    # @profile
    def get_frames(self, trial_index, data_key, starting_img_index, ending_img_index):
        cache = []
        if self._test:
            starting_line = starting_img_index
            ending_line = ending_img_index
        else:
            starting_line = (
                                    starting_img_index + self.test_frames
                            ) + self.num_of_imgs * trial_index
            ending_line = starting_line + self.time_chunk_size
            # print(f'first line: {starting_line} last line: {ending_line}')

        fixations = self.fixations[int(starting_line): int(ending_line)]
        # print(fixations[0], fixations[-1])
        # print()
        frames = torch.zeros((self.time_chunk_size, self.img_h, self.img_w))
        for i, (fixation, index) in enumerate(
                zip(fixations, range(starting_img_index, ending_img_index))
        ):
            if (index >= self.last_start_index) and (index < self.last_end_index):
                img = self.cache[index - self.last_start_index]
            else:
                img = torch.from_numpy(self.frames[fixation["img_index"]])
                # print(fixation["img_index"], fixation["center_x"], fixation["center_y"])
                img = crop_based_on_fixation(
                    img=img,
                    x_center=fixation["center_x"],
                    y_center=fixation["center_y"],
                    flip=fixation["flip"] == 0,
                    img_h=self.img_h,
                    img_w=self.img_w,
                    padding=self.padding,
                )
                img = torch.movedim(img, 0, 1)
            frames[i] = img
            cache.append(img)
        frames = torch.movedim(frames, 0, 2)
        frames = self.transform_image(frames)
        frames = torch.movedim(frames, 2, 0)
        self.last_start_index = starting_img_index
        self.last_end_index = ending_img_index
        self.cache = cache
        return frames
