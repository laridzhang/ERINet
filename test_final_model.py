import torch
import time
import tqdm

from src.utils import log
from src.erinet import ERINet
from src.data_multithread_preload import multithread_dataloader
from src import network

if __name__ == '__main__':
    torch.cuda.set_device(0)

    log_path = "log_test.txt"

    test_flag = dict()
    test_flag['preload'] = False

    test_model_path = r'./final_model.h5'

    test_data_config = dict()
    test_data_config['icon_128_test'] = test_flag.copy()

    # load data
    all_data = multithread_dataloader(test_data_config)

    net = ERINet()

    network.load_net(test_model_path, net)

    net.cuda()
    net.eval()

    log_info = list()

    total_forward_time = 0.0
    total_forward_index = 0

    # calculate error on the test dataset
    for data_name in test_data_config:
        data = all_data[data_name]['data']

        index = 0
        correct_count = 0
        for blob in tqdm.tqdm(data):
            image_data = blob['image']
            another_image_data = blob['another_image']
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
        log_info.append('%d samples with correct cent %f' % (index, this_correct_cent))
        log(log_path, log_info)

    log_info.append('total forward time is %f seconds of %d samples' % (total_forward_time, total_forward_index))
    log(log_path, log_info)
