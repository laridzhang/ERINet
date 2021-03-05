# import numpy as np
import torch
# import torch.nn as nn
# import math
import os
import sys

import tqdm

from src.crowd_count import CrowdCount
from src import network
# from src.utils import ndarray_to_tensor


def evaluate_model(model_path, data):
    net = CrowdCount()
    network.load_net(model_path, net)
    net.cuda()
    net.eval()

    correct_count = 0.0
    index = 0

    evaluate_data = data['data']

    for blob in tqdm.tqdm(evaluate_data, desc='evaluate', file=sys.stdout, mininterval=1.0):
        image_data = blob['image']
        another_image_data = blob['another_image']
        ground_truth_label = blob['label']

        image = torch.cat((image_data, another_image_data), dim=0)

        with torch.no_grad():
            estimate_label, _, _ = net(image)

        estimate_label = estimate_label.cpu()

        ground_truth_flag = torch.argmax(ground_truth_label, dim=1)
        estimate_flag = torch.argmax(estimate_label, dim=1)
        correct_flag = (ground_truth_flag == estimate_flag).to(torch.float32)
        correct_count += torch.sum(correct_flag)
        index += len(correct_flag)

    result_dict = dict()
    result_dict['name'] = os.path.basename(model_path)
    result_dict['number'] = int(index)
    result_dict['correct_cent'] = float(correct_count / index)

    return result_dict
