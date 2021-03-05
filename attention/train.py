import os
import torch
import numpy as np
import sys
# import cv2
import tqdm

from src.utils import log, ExcelLog, compare_result, gray_to_bgr, make_path
from src.models import ModelWithLoss
from src import network
# from src.data import Data
from src.evaluate_model import evaluate_model
from src.data_multithread_preload import multithread_dataloader

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # training configuration
    max_epoch = 1000  # maximum training times
    lr_adjust_epoch = None  # lr * 0.1 every lr_adjust_epoch steps
    lr = 0.00001 * 1.0  # default 0.00001
    # random_seed = 64678
    random_seed = None
    train_batch_size = 8

    is_load_pretrained_model = True  # load parameters of fine-tuned model

    key_error = 'average_iou'

    if is_load_pretrained_model:
        pretrained_model_path = list()
        pretrained_model_path.append('../pretrained_unet_carvana_scale1_epoch5.h5')

    log_path = "log.txt"

    train_flag = dict()
    train_flag['preload'] = False
    train_flag['shuffle'] = True
    train_flag['seed'] = random_seed
    train_flag['batch_size'] = train_batch_size
    train_flag['transform'] = True

    evaluate_flag = dict()
    evaluate_flag['preload'] = False

    # do not preload data when debugging
    if sys.gettrace() is not None:
        train_flag['preload'] = False
        evaluate_flag['preload'] = False

    train_data_config = dict()
    validation_data_config = dict()
    test_data_config = dict()

    original_dataset_name = 'icon'
    output_dir = './saved_models/'

    train_data_config['icon_128_train'] = train_flag.copy()
    # validation_data_config['shtA1_train'] = evaluate_flag.copy()
    test_data_config['icon_128_test'] = evaluate_flag.copy()

    # Check if there are duplicate keys in the config dict
    for key in train_data_config:
        if key in validation_data_config or key in test_data_config:
            raise Exception('duplicate dataset')
    for key in validation_data_config:
        if key in test_data_config:
            raise Exception('duplicate dataset')

    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # load data
    all_data = multithread_dataloader({**train_data_config, **validation_data_config, **test_data_config})

    # get train label weights
    # train_data_name, = train_data_config
    # train_data = all_data[train_data_name]
    # label_weights = train_data['label_weights']

    # initialize net
    net = ModelWithLoss(bce_init_weights=None)
    network.weights_normal_init(net, dev=0.01)  # default dev=0.01
    # network.save_net('model_init.h5', net)
    if is_load_pretrained_model:
        for path in pretrained_model_path:
            network.load_net_safe(path, net)
    # network.load_net(finetune_model, net)
    # network.save_net('model_loaded.h5', net)

    net.cuda()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.Adam([{'params': net.features.transition.parameters()},
    #                               {'params': net.features.map.parameters()}], lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if lr_adjust_epoch is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    standard_error_dict = dict()
    standard_error_dict['name'] = 'none'  # model name
    standard_error_dict['number'] = 0  # number of samples
    standard_error_dict['average_precision'] = 0.0
    standard_error_dict['average_iou'] = 0.0
    if key_error not in standard_error_dict:
        raise Exception('invalid key_error')
    best_result_dict = dict()
    for data_name in {**validation_data_config, **test_data_config}:
        best_result_dict[data_name] = standard_error_dict.copy()

    # display_interval = 1000
    txt_log_info = list()
    excel_log = ExcelLog('log.xlsx')

    log_best_model_history_list = list()  # put best model name in this list after writing to the front of the log file

    for data_name in train_data_config:
        txt_log_info.append('train data: %s' % data_name)
    for data_name in validation_data_config:
        txt_log_info.append('validation data: %s' % data_name)
    for data_name in test_data_config:
        txt_log_info.append('test data: %s' % data_name)

    for epoch in range(max_epoch):
        step = -1
        train_loss = 0.0
        number_of_train_samples = 0  # number of samples which are actually used to train

        for _, param_group in enumerate(optimizer.param_groups):
            txt_log_info.append("learning rate: {:.2e}".format(float(param_group['lr'])))
        log(log_path, txt_log_info)

        if len(train_data_config) > 1:
            raise Exception('more than one train dataset is provided')

        train_data_name, = train_data_config
        train_data = all_data[train_data_name]
        data = train_data['data']

        data_tqdm = tqdm.tqdm(data, desc='', file=sys.stdout, mininterval=1.0)
        for blob in data_tqdm:
            image = blob['image']
            ground_truth_mask = blob['mask']
            # image_name = blob['image_name']

            step += 1
            number_of_train_samples += 1

            estimate_mask, loss_dict, visual_dict = net(image, ground_truth=ground_truth_mask)

            loss = net.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sigmoid_estimate_mask = torch.sigmoid(estimate_mask)
            estimate_flag = (sigmoid_estimate_mask > 0.5).to(torch.int).cpu()
            ground_truth_flag = ground_truth_mask.to(torch.int)
            correct_flag = (estimate_flag == ground_truth_flag).to(torch.float32)
            correct_cent = torch.sum(correct_flag).item() / ground_truth_flag.numel()
            data_tqdm.set_description(desc='epoch: %4d, correct cent: %f' % (epoch, correct_cent), refresh=False)

        # display_interval = np.ceil(number_of_train_samples / 300) * 100
        train_loss = train_loss / number_of_train_samples

        txt_log_info.append('epoch: %4d train loss: %.20f' % (epoch, train_loss))

        model_name = '{}_{}_{}.h5'.format(original_dataset_name, epoch, step)
        save_model_path = os.path.join(output_dir, model_name)
        network.save_net(save_model_path, net)

        # evaluate the model of this epoch
        evaluate_result_dict = dict()
        for data_name in best_result_dict:
            evaluate_data = all_data[data_name]
            result = evaluate_model(save_model_path, evaluate_data)
            evaluate_result_dict[data_name] = result
            txt_log_info.append('evaluate %s on %s: %s: %6.4f' % (result['name'], data_name, key_error, result[key_error]))

        # check if this model is new best model
        best_result_dict = compare_result(evaluate_result_dict, best_result_dict, key_error, reverse=True)
        for data_name in best_result_dict:
            result = best_result_dict[data_name]
            txt_log_info.append('best model on %s is %s with %s of %.4f' % (data_name, result['name'], key_error, result[key_error]))

        log(log_path, txt_log_info)

        excel_log.add_log(evaluate_result_dict)

        if validation_data_config:  # if validation data is not empty
            for validation_data_name in validation_data_config:
                result = best_result_dict[validation_data_name]
                best_model_on_validation_now = result['name']
                if best_model_on_validation_now not in log_best_model_history_list:
                    log_best_model_history_list.append(best_model_on_validation_now)
                    # write test result of this model to the front of the log file
                    for test_data_name in test_data_config:
                        result = evaluate_result_dict[test_data_name]
                        if result['name'] != best_model_on_validation_now:
                            raise Exception('model name on validation (%s) and test (%s) mismatch' % (best_model_on_validation_now, result['name']))
                        txt_log_info.append('best model on validation %s: evaluate %s on %s: %s: %6.4f' % (validation_data_name, result['name'], test_data_name, key_error, result[key_error]))
                        log(log_path, txt_log_info, line=len(log_best_model_history_list) - 1)

        if lr_adjust_epoch is not None:
            scheduler.step()

    log(log_path, txt_log_info)
