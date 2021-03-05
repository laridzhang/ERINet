import os
import torch
import torch.nn.functional as functional
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

    # log_info = []

    save_path = './test_output'
    make_path(save_path)
    make_path(os.path.join(save_path, 'image_class_map'))

    total_forward_time = 0.0

    # calculate error on the test dataset
    for data_name in test_data_config:
        data = all_data[data_name]['data']

        index = 0
        for blob in tqdm.tqdm(data):
            image_data = blob['image']
            another_image_data = blob['another_image']
            image_name = blob['image_name']
            another_image_name = blob['another_image_name']
            ground_truth_label = blob['label']

            image = torch.cat((image_data, another_image_data), dim=0)

            start_time = time.perf_counter()

            with torch.no_grad():
                estimate_label, _, visual_dict = net(image)

            total_forward_time += time.perf_counter() - start_time
            index += 1

            pad_size = 2

            # image from tensor to ndarry
            image = image_data[0].data.cpu().numpy()
            image = image * 255.0
            image = np.moveaxis(image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height = image.shape[0]
            width = image.shape[1]

            # another image from tensor to ndarry
            another_image = another_image_data[0].data.cpu().numpy()
            another_image = another_image * 255.0
            another_image = np.moveaxis(another_image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
            another_image = cv2.cvtColor(another_image, cv2.COLOR_RGB2BGR)

            # pad and concatenate image and another image
            pad_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            pad_another_image = cv2.copyMakeBorder(another_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            two_images = np.concatenate((pad_image, pad_another_image), axis=1)

            # pad and concatenate estimated class maps
            estimate_class_map = visual_dict['class_map'].data.cpu().numpy()
            estimate_class_map_list = list()
            for map in estimate_class_map[0]:
                map = map / np.max(estimate_class_map)
                map = gray_to_bgr(map)
                map = cv2.resize(map, (width, height))
                map = cv2.copyMakeBorder(map, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                estimate_class_map_list.append(map)
            two_class_maps = np.concatenate(estimate_class_map_list, axis=1)

            # concatenate images and class maps
            image_class_map = np.concatenate((two_images, two_class_maps), axis=0)

            # put text on image
            label_dict = {0: 'same', 1: 'different'}
            ground_truth_label_text = label_dict[np.argmax(ground_truth_label.data.cpu().numpy(), axis=1)[0]]
            estimate_label_text = label_dict[np.argmax(estimate_label.data.cpu().numpy(), axis=1)[0]]
            cv2.putText(image_class_map, 'GT: %s' % ground_truth_label_text, org=(2, 14), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
            cv2.putText(image_class_map, 'ET: %s' % estimate_label_text, org=(2, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

            # save image
            cv2.imwrite(os.path.join(save_path, 'image_class_map', '%s_%s_estimate_image_class_map.jpg' % (image_name[0], another_image_name[0])), image_class_map)

            # display wrong result
            if np.argmax(ground_truth_label.data.cpu().numpy(), axis=1)[0] != np.argmax(estimate_label.data.cpu().numpy(), axis=1)[0]:
                cv2.imshow('%s_%s' % (image_name[0], another_image_name[0]), image_class_map)
                cv2.waitKey()

    print('total forward time is %f seconds of %d samples.' % (total_forward_time, index))
    # log(log_path, log_info)
