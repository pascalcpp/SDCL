from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util import *
from losses import *
from dataloaders import get_ema_model_and_dataloader
import torch.nn.functional as F

"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed_test = 2022
seed_reproducer(seed = seed_test)

data_root, split_name = '../Datasets/pancreas/data', 'pancreas'
result_dir = 'result/pancreas/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
label_percent = 20
self_train_name = 'self_train'





def test_model(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    load_path = Path(result_dir) / self_train_name
    load_net(net1, load_path / 'best_ema_20_self.pth')
    load_net(net2, load_path / 'best_ema_20_self_resnet.pth')

    print('Successful Loaded')
    avg_metric, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
    avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)
    print(avg_metric)
    print(avg_metric2)
    print(avg_metric3)


if __name__ == '__main__':
    try:
        net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        test_model(net1, net2, test_loader)

    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


