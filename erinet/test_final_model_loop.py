import os
import torch
# import torch.nn.functional as functional
import numpy as np
import cv2
import time
import tqdm

from src.utils import log, is_only_one_bool_is_true, gray_to_bgr, make_path, calculate_game
from src.crowd_count import CrowdCount
# from src.data import Data
from src.data_multithread_preload import multithread_dataloader
from src import network

if __name__ == '__main__':
    torch.cuda.set_device(0)

    log_path = "log_test.txt"

    test_flag = dict()
    test_flag['preload'] = False

    test_model_path = r'./final_with_attention.h5'
    original_dataset_name = 'icon'

    test_data_config = dict()
    test_data_config['icon_128_test'] = test_flag.copy()

    # load data
    all_data = multithread_dataloader(test_data_config)

    net = CrowdCount()

    network.load_net(test_model_path, net)

    net.cuda()
    net.eval()

    log_info = list()
    correct_cent_list = list()

    loop_time = 20

    total_forward_time = 0.0
    total_forward_index = 0

    # calculate error on the test dataset
    for data_name in test_data_config:
        data = all_data[data_name]['data']

        for loop in range(loop_time):

            index = 0
            correct_count = 0
            for blob in tqdm.tqdm(data):
                image_data = blob['image']
                another_image_data = blob['another_image']
                # image_name = blob['image_name']
                # another_image_name = blob['another_image_name']
                ground_truth_label = blob['label']

                image = torch.cat((image_data, another_image_data), dim=0)

                start_time = time.perf_counter()

                with torch.no_grad():
                    estimate_label, _, _ = net(image)

                total_forward_time += time.perf_counter() - start_time
                total_forward_index += 1

                ground_truth_flag = torch.argmax(ground_truth_label, dim=1)
                estimate_flag = torch.argmax(estimate_label.cpu(), dim=1)
                correct_flag = (ground_truth_flag == estimate_flag).to(torch.float32)
                correct_count += torch.sum(correct_flag).item()
                index += len(correct_flag)

            this_correct_cent = correct_count / index
            correct_cent_list.append(this_correct_cent)
            log_info.append('loop %4d: %d samples with correct cent %f' % (loop, index, this_correct_cent))
            log(log_path, log_info)

    print('total forward time is %f seconds of %d samples.' % (total_forward_time, total_forward_index))

    average_correct_cent = sum(correct_cent_list) / len(correct_cent_list)
    log_info.append('average correct cent of %d loops is %f' % (len(correct_cent_list), average_correct_cent))
    log(log_path, log_info)
