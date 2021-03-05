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

    # log_info = []

    save_path = './test_output'
    make_path(save_path)
    make_path(os.path.join(save_path, 'image_mask'))
    make_path(os.path.join(save_path, 'mask'))

    total_forward_time = 0.0
    total_iou = 0.0

    # calculate error on the test dataset
    for data_name in test_data_config:
        data = all_data[data_name]['data']

        index = 0
        for blob in tqdm.tqdm(data):
            image = blob['image']
            ground_truth_mask = blob['mask']
            image_name = blob['image_name']

            start_time = time.perf_counter()

            with torch.no_grad():
                estimate_mask, _, _ = net(image)

            total_forward_time += time.perf_counter() - start_time
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

            sigmoid_estimate_mask = torch.sigmoid(estimate_mask)
            estimate_mask = (sigmoid_estimate_mask > 0.5).to(torch.float).cpu()

            # dilated_estimate_mask = dilate_mask(estimate_mask, 10, is_cuda=False)
            dilated_estimate_mask = dilate_mask_tensor(estimate_mask, 10)

            estimate_mask_image = estimate_mask * image
            ground_truth_mask_image = ground_truth_mask * image
            dilated_estimate_mask_image = dilated_estimate_mask * image

            # from tensor to ndarry
            image = image[0].data.cpu().numpy()
            image = image * 255.0
            image = np.moveaxis(image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            estimate_mask_image = estimate_mask_image[0].data.cpu().numpy()
            estimate_mask_image = estimate_mask_image * 255.0
            estimate_mask_image = np.moveaxis(estimate_mask_image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            estimate_mask_image = cv2.cvtColor(estimate_mask_image, cv2.COLOR_RGB2BGR)

            ground_truth_mask_image = ground_truth_mask_image[0].data.cpu().numpy()
            ground_truth_mask_image = ground_truth_mask_image * 255.0
            ground_truth_mask_image = np.moveaxis(ground_truth_mask_image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            ground_truth_mask_image = cv2.cvtColor(ground_truth_mask_image, cv2.COLOR_RGB2BGR)

            dilated_estimate_mask_image = dilated_estimate_mask_image[0].data.cpu().numpy()
            dilated_estimate_mask_image = dilated_estimate_mask_image * 255.0
            dilated_estimate_mask_image = np.moveaxis(dilated_estimate_mask_image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            dilated_estimate_mask_image = cv2.cvtColor(dilated_estimate_mask_image, cv2.COLOR_RGB2BGR)

            # save image and mask
            cv2.imwrite(os.path.join(save_path, 'image_mask', '%s_image.jpg' % image_name[0]), image)
            cv2.imwrite(os.path.join(save_path, 'image_mask', '%s_estimate_mask_image.jpg' % image_name[0]), estimate_mask_image)
            cv2.imwrite(os.path.join(save_path, 'image_mask', '%s_ground_truth_mask_image.jpg' % image_name[0]), ground_truth_mask_image)
            cv2.imwrite(os.path.join(save_path, 'image_mask', '%s_dilated_estimate_mask_image.jpg' % image_name[0]), dilated_estimate_mask_image)

            # mask
            ground_truth_mask_np = ground_truth_mask[0][0].data.numpy()
            ground_truth_mask_np = ground_truth_mask_np * 255.0
            ground_truth_mask_np = ground_truth_mask_np.astype(np.uint8)
            ground_truth_mask_np = cv2.cvtColor(ground_truth_mask_np, cv2.COLOR_GRAY2BGR)
            sigmoid_estimate_mask = torch.sigmoid(estimate_mask)
            estimate_flag = (sigmoid_estimate_mask > 0.5).to(torch.float)
            estimate_mask_np = estimate_flag[0][0].data.cpu().numpy()
            estimate_mask_np = estimate_mask_np * 255.0
            estimate_mask_np = estimate_mask_np.astype(np.uint8)
            estimate_mask_np = cv2.cvtColor(estimate_mask_np, cv2.COLOR_GRAY2BGR)

            # save mask
            cv2.imwrite(os.path.join(save_path, 'mask', '%s_ground_turth_mask.jpg' % image_name[0]), ground_truth_mask_np)
            cv2.imwrite(os.path.join(save_path, 'mask', '%s_estimate_mask.jpg' % image_name[0]), estimate_mask_np)

        print('total forward time is %f seconds of %d samples.' % (total_forward_time, index))
        print('average iou is %f' % (total_iou / index))
