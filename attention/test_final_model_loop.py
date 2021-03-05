import os
import torch
import torch.nn.functional as functional
import numpy as np
import cv2
import time
import tqdm

from src.utils import log, is_only_one_bool_is_true, gray_to_bgr, make_path, calculate_game, dilate_mask, dilate_mask_tensor
from src.models import ModelWithLoss
# from src.data import Data
from src.data_multithread_preload import multithread_dataloader
from src import network

if __name__ == '__main__':
    torch.cuda.set_device(0)

    log_path = "log_test.txt"

    test_flag = dict()
    test_flag['preload'] = False

    test_model_path = r'./final_unet.h5'
    original_dataset_name = 'icon'

    test_data_config = dict()
    test_data_config['icon_128_test'] = test_flag.copy()

    # load data
    all_data = multithread_dataloader(test_data_config)

    net = ModelWithLoss()

    network.load_net(test_model_path, net)

    net.cuda()
    net.eval()

    log_info = list()

    total_forward_time = 0.0
    total_forward_index = 0

    loop_time = 20

    # calculate error on the test dataset
    for data_name in test_data_config:
        data = all_data[data_name]['data']

        average_iou_list = list()

        for loop in range(loop_time):
            index = 0
            total_iou = 0.0
            for blob in tqdm.tqdm(data):
                image = blob['image']
                ground_truth_mask = blob['mask']
                image_name = blob['image_name']

                start_time = time.perf_counter()

                with torch.no_grad():
                    estimate_mask, _, _ = net(image)

                total_forward_time += time.perf_counter() - start_time
                total_forward_index += 1

                index += 1

                # calculate iou
                sigmoid_estimate_mask = torch.sigmoid(estimate_mask)
                estimate_flag = (sigmoid_estimate_mask > 0.5).to(torch.int).cpu()
                ground_truth_flag = ground_truth_mask.to(torch.int)
                this_iou = 0.0
                this_intersection = torch.sum(estimate_flag * ground_truth_flag)
                this_union = torch.sum(torch.clamp(estimate_flag + ground_truth_flag, 0, 1))
                if this_union > 0:
                    this_iou += this_intersection / this_union
                else:
                    if this_intersection == 0:
                        this_iou += 1.0
                    else:
                        this_iou += 0.0
                total_iou += this_iou

            # print('average iou is %f' % (total_iou / index))
            this_average_iou = total_iou / index
            log_info.append('average iou of loop %d is %f' % (loop, this_average_iou))
            log(log_path, log_info)

            average_iou_list.append(this_average_iou)

        average_iou = sum(average_iou_list) / len(average_iou_list)
        log_info.append('average iou of %d loops is %f' % (average_iou, len(average_iou_list)))
        log(log_path, log_info)

    print('total forward time is %f seconds of %d samples.' % (total_forward_time, total_forward_index))
