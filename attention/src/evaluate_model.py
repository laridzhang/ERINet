# import numpy as np
import torch
# import torch.nn as nn
# import math
import os
import sys

import tqdm

from src.models import ModelWithLoss
from src import network
# from src.utils import ndarray_to_tensor


def evaluate_model(model_path, data):
    net = ModelWithLoss()
    network.load_net(model_path, net)
    net.cuda()
    net.eval()

    correct_pixel = 0
    total_pixel = 0
    total_iou = 0.0
    index = 0

    evaluate_data = data['data']

    for blob in tqdm.tqdm(evaluate_data, desc='evaluate', file=sys.stdout, mininterval=1.0):
        image = blob['image']
        ground_truth_mask = blob['mask']

        with torch.no_grad():
            estimate_mask, _, _ = net(image)

        estimate_mask = estimate_mask.cpu()

        sigmoid_estimate_mask = torch.sigmoid(estimate_mask)
        estimate_flag = (sigmoid_estimate_mask > 0.5).to(torch.int)
        ground_truth_flag = ground_truth_mask.to(torch.int)

        correct_flag = (estimate_flag == ground_truth_flag).to(torch.int)
        correct_pixel += torch.sum(correct_flag).item()
        total_pixel += ground_truth_flag.numel()

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

        index += 1

    result_dict = dict()
    result_dict['name'] = os.path.basename(model_path)
    result_dict['number'] = int(index)
    result_dict['average_precision'] = float(correct_pixel / total_pixel)
    result_dict['average_iou'] = float(total_iou / index)

    return result_dict
