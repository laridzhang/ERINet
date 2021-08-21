# load data in multithreading

import numpy as np
import os
import cv2
import random
import warnings
# import pandas
# import scipy.io as scio
import torch
# import torch.nn as nn
# import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
import torchvision
import psutil
# import warnings
from os.path import join
import pickle
import string
# import shutil
import datetime

from src.utils import ndarray_to_tensor, print_red, make_path, overlay_image_center, check_image
from src.data_path import DataPath


class PreloadData:
    def __init__(self, icon_path, background_path, is_random_background=True, is_preload=False, is_transfrom=False, is_transform_in_gray=False):
        # image_path: path of all image file
        # density_map_path: path of all density map file
        # roi_path: path of all region of interest file
        self.icon_path = icon_path
        self.background_path = background_path
        self.is_random_background = is_random_background
        self.is_preload = is_preload
        self.is_transform = is_transfrom
        self.is_transform_in_gray = is_transform_in_gray

        self.image2tensor = torchvision.transforms.ToTensor()

        if self.is_transform:
            self.image2pil = torchvision.transforms.ToPILImage()
            self.color_jitter = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
            if self.is_transform_in_gray:
                self.image2gray = torchvision.transforms.Grayscale(num_output_channels=1)
                self.image2grayRGB = torchvision.transforms.Grayscale(num_output_channels=3)

        self.min_available_memory = 4 * 1024 * 1024 * 1024  # GB

        # make path for pickle
        time_now = datetime.datetime.now()
        self.pickle_path = os.path.join(r'./pickle', '%4d%02d%02d%02d%02d%02d%06d_%s' %
                                        (time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second, time_now.microsecond, ''.join(random.sample(string.ascii_letters, 4))))
        make_path(self.pickle_path)

        # get all icon image filename
        self.icon_filename_list = [f for f in os.listdir(self.icon_path) if os.path.isfile(join(self.icon_path, f))]
        self.icon_filename_list.sort()

        self.number_of_samples = len(self.icon_filename_list)

        # get max size of all icon image
        self.max_icon_height = 0
        self.max_icon_width = 0
        for filename in self.icon_filename_list:
            image = cv2.imread(join(self.icon_path, filename), cv2.IMREAD_UNCHANGED)
            height = image.shape[0]
            width = image.shape[1]
            if height > self.max_icon_height:
                self.max_icon_height = height
            if width > self.max_icon_width:
                self.max_icon_width = width

        # preload icon image file
        self.preload_icon_dict = dict()  # store all preload data in this dict
        index = 0
        for filename in self.icon_filename_list:
            if self.is_preload:
                if psutil.virtual_memory().available > self.min_available_memory:
                    index += 1
                    self.preload_icon_dict[filename] = cv2.imread(join(self.icon_path, filename), cv2.IMREAD_UNCHANGED)
                    if index % 100 == 0:
                        print('Loaded %6d of %d files.' % (index, self.number_of_samples))
                else:
                    self.preload_icon_dict[filename] = None
            else:
                self.preload_icon_dict[filename] = None
        print('Completed loading %d files. %d files are preloaded.' % (self.number_of_samples, index))

        if self.is_random_background:
            # get all background image filename
            self.background_filename_list = [f for f in os.listdir(self.background_path) if os.path.isfile(join(self.background_path, f))]
            self.background_filename_list.sort()

            # check if any background filename and icon filename are duplicated. make sure write and read pickles correctly by filename
            for filename in self.background_filename_list:
                if filename in self.icon_filename_list:
                    raise Exception('background image filename %s is duplicated' % filename)

            self.number_of_background = len(self.background_filename_list)

            # preload background image file
            self.preload_background_dict = dict()  # store all preload data in this dict
            index = 0
            for filename in self.background_filename_list:
                if self.is_preload:
                    if psutil.virtual_memory().available > self.min_available_memory:
                        index += 1
                        self.preload_background_dict[filename] = cv2.imread(join(self.background_path, filename), cv2.IMREAD_COLOR)
                        if index % 100 == 0:
                            print('Loaded %6d of %d files.' % (index, self.number_of_background))
                    else:
                        self.preload_background_dict[filename] = None
                else:
                    self.preload_background_dict[filename] = None
            print('Completed loading %d files. %d files are preloaded.' % (self.number_of_background, index))

        return

    def get_number_of_samples(self):
        return self.number_of_samples

    def get_background(self, height, width):
        # return random cropped background image that fits height and width
        if not self.is_random_background:
            raise Exception('background is not available because is_random_background is set to False')

        background_filename = self.background_filename_list[random.randrange(self.number_of_background)]
        background_image = self.preload_background_dict[background_filename]

        if background_image is None:  # no data is preloaded
            pickle_file_path = os.path.join(self.pickle_path, background_filename + '.pickle')
            if os.path.isfile(pickle_file_path):
                with open(pickle_file_path, 'rb') as f:
                    background_image = pickle.load(f)
            else:
                background_image = cv2.imread(join(self.background_path, background_filename), cv2.IMREAD_COLOR)
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(background_image, f)


        background_height = background_image.shape[0]
        background_width = background_image.shape[1]
        y = random.randrange(background_height - height)
        x = random.randrange(background_width - width)
        crop_background_image = background_image.copy()[y:y + height, x:x + width, :]

        return crop_background_image

    def get_icon_by_index(self, index):
        icon_filename = self.icon_filename_list[index]
        icon_name, _ = os.path.splitext(icon_filename)
        icon_image = self.preload_icon_dict[icon_filename]

        if icon_image is None:  # no data is preloaded
            pickle_file_path = os.path.join(self.pickle_path, icon_filename + '.pickle')
            if os.path.isfile(pickle_file_path):
                with open(pickle_file_path, 'rb') as f:
                    icon_image = pickle.load(f)
            else:
                icon_image = cv2.imread(join(self.icon_path, icon_filename), cv2.IMREAD_UNCHANGED)
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(icon_image, f)

        return icon_image, icon_name

    def pad_image(self, image, height, width):
        # pad image to fit expected height and width
        image_height = image.shape[0]
        image_width = image.shape[1]
        if image_height < height:
            image = cv2.copyMakeBorder(image, 0, height - image_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif image_height == height:
            pass
        else:
            raise Exception('image_height %d is larger than expected %d' % (image_height, height))
        if image_width < width:
            image = cv2.copyMakeBorder(image, 0, 0, 0, width - image_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif image_width == width:
            pass
        else:
            raise Exception('image_width %d is larger than expected %d' % (image_width, width))

        return image

    def cv2image2BGR(self, image):
        image_type = check_image(image)
        if image_type == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image_type == 'color':
            pass
        elif image_type == 'transparent':
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise Exception('invalid image_type')

        return image

    def get_blob_by_index(self, index):
        icon_image, icon_name = self.get_icon_by_index(index)

        if self.is_random_background:
            # random choose same or different icon and background
            # 0 or 1. (same) 40% same icon with different background
            # 2. (different) 20% same icon (one is original, another is cover by a small rectangle) with same background
            # 3. (different) 20% different icon with same background
            # 4. (different) 20% different icon with different background
            label = np.zeros(2, dtype=np.int)
            pad_size = 10  # height and width of background is 2*pad_size pixels larger than icon_image
            flag = random.randrange(5)
            # flag = 1
            # warnings.warn('data generating flag is set to %d' % flag)
            if flag == 0 or flag == 1:  # same icon with different background
                label[0] = 1  # same

                another_icon_image, another_icon_name = self.get_icon_by_index(index)  # same icon image

                # get different background images
                icon_height = icon_image.shape[0]
                icon_width = icon_image.shape[1]
                crop_height = icon_height + 2 * pad_size
                crop_width = icon_width + 2 * pad_size
                crop_background_image = self.get_background(crop_height, crop_width)
                another_crop_background_image = self.get_background(crop_height, crop_width)
            elif flag == 2:  # same icon (one is original, another is cover by a small rectangle) with same background
                label[1] = 1  # different

                another_icon_image, another_icon_name = self.get_icon_by_index(index)  # same icon image

                # put a small rectangle with random color on random position of icon image
                another_icon_image_height = another_icon_image.shape[0]
                another_icon_image_width = another_icon_image.shape[1]
                rectangle_height = round(another_icon_image_height / 2)
                rectangle_width = round(another_icon_image_width / 2)
                x = random.randrange(another_icon_image_width - rectangle_width)
                y = random.randrange(another_icon_image_height - rectangle_height)
                another_icon_image_type = check_image(another_icon_image)
                if another_icon_image_type == 'grayscale':
                    # rectangle = np.random.randint(low=0, high=255, size=(rectangle_height, rectangle_width), dtype=np.uint8)
                    # another_icon_image[y: y + rectangle_height, x: x + rectangle_width] = rectangle
                    another_icon_image[y: y + rectangle_height, x: x + rectangle_width] += 128
                elif another_icon_image_type == 'color':
                    # rectangle = np.random.randint(low=0, high=255, size=(rectangle_height, rectangle_width, 3), dtype=np.uint8)
                    # another_icon_image[y: y + rectangle_height, x: x + rectangle_width, :] = rectangle
                    another_icon_image[y: y + rectangle_height, x: x + rectangle_width, :] += 128
                elif another_icon_image_type == 'transparent':
                    another_icon_image[y: y + rectangle_height, x: x + rectangle_width, 3] = 0
                else:
                    raise Exception('invalid another_icon_image_type')

                # get same background images
                icon_height = icon_image.shape[0]
                icon_width = icon_image.shape[1]
                crop_height = icon_height + 2 * pad_size
                crop_width = icon_width + 2 * pad_size
                crop_background_image = self.get_background(crop_height, crop_width)
                another_crop_background_image = crop_background_image.copy()
            elif flag == 3:  # different icon with same background
                label[1] = 1  # different

                # choose another index for another icon image
                another_index = random.randrange(self.number_of_samples)
                while another_index == index:
                    another_index = random.randrange(self.number_of_samples)

                another_icon_image, another_icon_name = self.get_icon_by_index(another_index)  # different icon image

                # get same background images
                icon_height = icon_image.shape[0]
                icon_width = icon_image.shape[1]
                another_icon_height = another_icon_image.shape[0]
                another_icon_width = another_icon_image.shape[1]
                crop_height = max(icon_height, another_icon_height) + 2 * pad_size
                crop_width = max(icon_width, another_icon_width) + 2 * pad_size
                this_background_image = self.get_background(crop_height, crop_width)
                crop_background_image = this_background_image.copy()[0: icon_height + 2 * pad_size, 0: icon_width + 2 * pad_size, :]
                another_crop_background_image = this_background_image.copy()[0: another_icon_height + 2 * pad_size, 0: another_icon_width + 2 * pad_size, :]
            elif flag == 4:  # different icon with different background
                label[1] = 1  # different

                # choose another index for another icon image
                another_index = random.randrange(self.number_of_samples)
                while another_index == index:
                    another_index = random.randrange(self.number_of_samples)

                another_icon_image, another_icon_name = self.get_icon_by_index(another_index)  # different icon image

                # get different background images
                icon_height = icon_image.shape[0]
                icon_width = icon_image.shape[1]
                crop_height = icon_height + 2 * pad_size
                crop_width = icon_width + 2 * pad_size
                crop_background_image = self.get_background(crop_height, crop_width)
                another_icon_height = another_icon_image.shape[0]
                another_icon_width = another_icon_image.shape[1]
                another_crop_height = another_icon_height + 2 * pad_size
                another_crop_width = another_icon_width + 2 * pad_size
                another_crop_background_image = self.get_background(another_crop_height, another_crop_width)
            else:
                raise Exception('invalid flag of same or different')

            # overlay icon image on background image
            icon_image = overlay_image_center(icon_image, crop_background_image)
            another_icon_image = overlay_image_center(another_icon_image, another_crop_background_image)

            height = self.max_icon_height + 2 * pad_size  # expected height of icon_image
            width = self.max_icon_width + 2 * pad_size  # expected width of icon_image
        else:
            # random choose same or different icon
            label = np.zeros(2, dtype=np.int)
            flag = random.randrange(2)
            if flag == 0:  # same icon
                label[0] = 1  # same

                another_icon_image, another_icon_name = self.get_icon_by_index(index)  # same icon image
            elif flag == 1:  # different icon
                label[1] = 1  # different

                # choose another index for another icon image
                another_index = random.randrange(self.number_of_samples)
                while another_index == index:
                    another_index = random.randrange(self.number_of_samples)

                another_icon_image, another_icon_name = self.get_icon_by_index(another_index)  # different icon image
            else:
                raise Exception('invalid flag of same or different')

            icon_image = self.cv2image2BGR(icon_image)
            another_icon_image = self.cv2image2BGR(another_icon_image)

            height = self.max_icon_height  # expected height of icon_image
            width = self.max_icon_width  # expected width of icon_image

        icon_image = self.pad_image(icon_image, height, width)
        icon_image = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
        icon_image = self.image2tensor(icon_image)

        another_icon_image = self.pad_image(another_icon_image, height, width)
        another_icon_image = cv2.cvtColor(another_icon_image, cv2.COLOR_BGR2RGB)
        another_icon_image = self.image2tensor(another_icon_image)

        this_blob = dict()
        this_blob['image'] = icon_image
        this_blob['image_name'] = icon_name
        this_blob['another_image'] = another_icon_image
        this_blob['another_image_name'] = another_icon_name
        this_blob['label'] = ndarray_to_tensor(label, is_cuda=False)

        # transform image
        if self.is_transform:
            image = this_blob['image']
            image = self.image_color_jitter(image)
            this_blob['image'] = image

            image = this_blob['another_image']
            image = self.image_color_jitter(image)
            this_blob['another_image'] = image

        return this_blob

    def image_color_jitter(self, image):
        # input tensor. return tensor

        image = self.image2pil(image)
        # image.show()
        if self.is_transform_in_gray:
            image = self.image2gray(image)
            image = self.color_jitter(image)
            image = self.image2grayRGB(image)
        else:
            image = self.color_jitter(image)
        # image.show()
        image = self.image2tensor(image)

        return image


class Data(Dataset):
    def __init__(self, preload_data):
        # image_path: path of all image file
        # density_map_path: path of all density map file
        # roi_path: path of all region of interest file
        self.preload_data = preload_data

    def __len__(self):
        return self.preload_data.get_number_of_samples()

    def __getitem__(self, index):
        return self.preload_data.get_blob_by_index(index)


def multithread_dataloader(data_config):
    # data_config: dict, a dictionay contains several datasets info,
    #              key is dataset name,
    #              value is a dict which contains is_preload and is_label and is_mask
    data_path = DataPath()

    data_dict = dict()

    for name in data_config:
        this_dataset_flag = data_config[name]
        is_preload = this_dataset_flag['preload']
        if 'shuffle' in this_dataset_flag:
            is_shuffle = this_dataset_flag['shuffle']
        else:
            is_shuffle = False
        if 'seed' in this_dataset_flag:
            random_seed = this_dataset_flag['seed']
        else:
            random_seed = None
        if 'batch_size' in this_dataset_flag:
            batch_size = this_dataset_flag['batch_size']
        else:
            batch_size = 1
        if 'transform' in this_dataset_flag:
            is_transform = this_dataset_flag['transform']
            if 'transform_in_gray' in this_dataset_flag:
                is_transform_in_gray = this_dataset_flag['transform_in_gray']
            else:
                is_transform_in_gray = False
        else:
            is_transform = False
            is_transform_in_gray = False

        if random_seed is not None:
            def worker_init_fn(x):
                seed = random_seed + x
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                return
        else:
            worker_init_fn = None

        path = data_path.get_path(name)
        preload_data = PreloadData(path['icon'], path['background'], is_preload=is_preload, is_transfrom=is_transform, is_transform_in_gray=is_transform_in_gray)
        this_data = Data(preload_data)
        this_dataloader = DataLoader(this_data, batch_size=batch_size, shuffle=is_shuffle, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn)

        this_dataset_dict = dict()
        this_dataset_dict['data'] = this_dataloader

        data_dict[name] = this_dataset_dict

    return data_dict
