import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (ACDCDataSet, BaseDataSets, RandomGenerator, TwoStreamBatchSampler,
                                 ThreeStreamBatchSampler)
from networks.net_factory import BCP_net
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
from utils.LA_utils import to_cuda

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/acdc', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SDCL', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-3, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_logits) ** 2
    return mse_loss


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def mix_mse_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 4)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 4)

    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight


    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss


voxel_kl_loss = nn.KLDivLoss(reduction="none")


def mix_max_kl_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask

    with torch.no_grad():
        s1 = torch.softmax(net3_output, dim=1)
        l1 = torch.argmax(s1, dim=1)
        img_diff_mask = (l1 != img_l)
        patch_diff_mask = (l1 != patch_l)

        uniform_distri = torch.ones(net3_output.shape)
        uniform_distri = uniform_distri.cuda()

    kl_loss = torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri),
                         dim=1) * mask * img_diff_mask * image_weight
    kl_loss += torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri),
                          dim=1) * patch_mask * patch_diff_mask * patch_weight

    sum_diff = torch.sum(mask * img_diff_mask * diff_mask) + torch.sum(patch_mask * patch_diff_mask * diff_mask)
    loss = torch.sum(diff_mask * kl_loss) / (sum_diff + 1e-16)
    return loss


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask
  
import csv

from test_ACDC import TESTACDC

def pre_train(args, snapshot_path):
    num_classes = args.num_classes
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    model = BCP_net(model="UNet", in_chns=1, class_num=num_classes)
    model2 = BCP_net(model="ResUNet", in_chns=1, class_num=num_classes)



    db_val = ACDCDataSet(base_dir=args.root_path, split="val", logging=logging)
    c_batch_size = 12

    trainset_lab_a = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                 logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)


    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)



    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    logging.info("optim.Adam pre_training")

    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainset_lab_a)))

    model.train()
    model2.train()

    iter_num = 0
    best_performance = 0.0
    best_performance2 = 0.0
    max_epoch = 101
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()

            img_mask, loss_mask = generate_mask(img_a)


            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            out_mixl_2 = model2(net_input)

            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss_dice_2, loss_ce_2 = mix_loss(out_mixl_2, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2
            loss_2 = (loss_dice_2 + loss_ce_2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            iter_num += 1

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))


        if epoch >= 50 and epoch % 5 == 0:
            model.eval()
            model2.eval()
            metric_list = 0.0
            metric_list_2 = 0.0
            for _, (img_val, lab_val) in tqdm(enumerate(valloader), ncols=70):
                metric_i = val_2d.test_single_volume(img_val, lab_val, model, classes=num_classes)
                metric_i_2 = val_2d.test_single_volume(img_val, lab_val, model2, classes=num_classes)

                metric_list += np.array(metric_i)
                metric_list_2 += np.array(metric_i_2)

            metric_list = metric_list / len(db_val)
            metric_list_2 = metric_list_2 / len(db_val)

            performance = np.mean(metric_list, axis=0)[0]
            performance2 = np.mean(metric_list_2, axis=0)[0]

            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path,
                                              'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                save_net_opt(model, optimizer, save_mode_path)
                save_net_opt(model, optimizer, save_best_path)

            if performance2 > best_performance2:
                best_performance2 = performance2
                save_mode_path = os.path.join(snapshot_path,
                                              'iter_{}_dice_{}_res.pth'.format(iter_num, round(best_performance2, 4)))
                save_best_path = os.path.join(snapshot_path, 'best_model_res.pth')
                save_net_opt(model2, optimizer2, save_mode_path)
                save_net_opt(model2, optimizer2, save_best_path)

            logging.info('iteration %d : mean_dice : %f, val_maxdice : %f' % (iter_num, performance, best_performance))
            logging.info(
                'resnet iteration %d : mean_dice : %f, val_maxdice : %f' % (iter_num, performance2, best_performance2))
            model.train()
            model2.train()



def self_train(args, pre_snapshot_path, snapshot_path):
    num_classes = args.num_classes
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model = os.path.join(pre_snapshot_path, 'best_model.pth')
    pre_trained_model2 = os.path.join(pre_snapshot_path, 'best_model_res.pth')

    model = BCP_net(model="UNet", in_chns=1, class_num=num_classes)
    model2 = BCP_net(model="ResUNet", in_chns=1, class_num=num_classes)
    ema_model = BCP_net(model="UNet", in_chns=1, class_num=num_classes, ema=True)

    db_val = ACDCDataSet(base_dir=args.root_path, split="val", logging=logging)
    c_batch_size = 12

    trainset_lab_a = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                 logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = ACDCDataSet(base_dir=args.root_path, split="train_unlab",
                                   transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = ACDCDataSet(base_dir=args.root_path, split="train_unlab",
                                   transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                   logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)


    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    load_net_opt(ema_model, optimizer, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)

    load_net_opt(model2, optimizer2, pre_trained_model2)

    logging.info("Loaded from {}".format(pre_trained_model))

    logging.info("Start self_training")


    model.train()
    model2.train()
    ema_model.train()


    iter_num = 0
    best_performance = 0.0
    best_performance2 = 0.0
    best_performance_mean = 0.0
    max_epoch = 401
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch in iterator:
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():
                pre_a = ema_model(unimg_a)
                pre_b = ema_model(unimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)



            net_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net_input_unl = img_a * img_mask + unimg_b * (1 - img_mask)

            out_l = model(net_input_l)
            out_unl = model(net_input_unl)

            out_l_2 = model2(net_input_l)
            out_unl_2 = model2(net_input_unl)

            l_dice, l_ce = mix_loss(out_l, plab_a, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            unl_dice, unl_ce = mix_loss(out_unl, lab_a, plab_b, loss_mask, u_weight=args.u_weight)

            l_dice_2, l_ce_2 = mix_loss(out_l_2, plab_a, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, lab_a, plab_b, loss_mask, u_weight=args.u_weight)

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice

            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            with torch.no_grad():
                diff_mask1 = get_XOR_region(out_l, out_l_2)
                diff_mask2 = get_XOR_region(out_unl, out_unl_2)

            net1_mse_loss_lab = mix_mse_loss(out_l, plab_a.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(out_l, plab_a.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(out_unl, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(out_unl, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(out_l_2, plab_a.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(out_l_2, plab_a.long(), lab_b, loss_mask, unlab=True,
                                               diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(out_unl_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(out_unl_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            loss = (loss_dice + loss_ce) / 2 + 0.5 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.05 * (
                        net1_kl_loss_lab + net1_kl_loss_unlab)

            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + 0.5 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.05 * (
                        net2_kl_loss_lab + net2_kl_loss_unlab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)


            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f '
                         'net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, '
                         'net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f' % (
                         iter_num, loss, loss_dice, loss_ce, net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                         net1_kl_loss_lab.item(), net1_kl_loss_unlab.item()))

            if iter_num % 200 == 0:
                model.eval()
                model2.eval()
                metric_list = 0.0
                metric_list_2 = 0.0
                metric_list_mean = 0.0

                for _, (img_val, lab_val) in tqdm(enumerate(valloader), ncols=70):
                    metric_i = val_2d.test_single_volume(img_val, lab_val, model, classes=num_classes)
                    metric_i_2 = val_2d.test_single_volume(img_val, lab_val, model2, classes=num_classes)
                    metric_i_mean = val_2d.test_single_volume_mean(img_val, lab_val, model, model2, classes=num_classes)

                    metric_list += np.array(metric_i)
                    metric_list_2 += np.array(metric_i_2)
                    metric_list_mean += np.array(metric_i_mean)

                metric_list = metric_list / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)
                metric_list_mean = metric_list_mean / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]
                performance2 = np.mean(metric_list_2, axis=0)[0]
                performance_mean = np.mean(metric_list_mean, axis=0)[0]

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_res.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, 'best_model_res.pth')
                    save_net_opt(model2, optimizer2, save_mode_path)
                    save_net_opt(model2, optimizer2, save_best_path)

                if performance_mean > best_performance_mean:
                    best_performance_mean = performance_mean

                    save_mode_path1 = os.path.join(snapshot_path, 'iter_{}_dice_{}_v.pth'.format(iter_num, round(
                        best_performance_mean, 4)))
                    save_best_path1 = os.path.join(snapshot_path, 'best_model_v.pth')

                    save_mode_path2 = os.path.join(snapshot_path, 'iter_{}_dice_{}_r.pth'.format(iter_num, round(
                        best_performance_mean, 4)))
                    save_best_path2 = os.path.join(snapshot_path, 'best_model_r.pth')

                    save_net_opt(model, optimizer, save_mode_path1)
                    save_net_opt(model, optimizer, save_best_path1)

                    save_net_opt(model2, optimizer2, save_mode_path2)
                    save_net_opt(model2, optimizer2, save_best_path2)
                  
                TESTACDC(iter_num, phase='self_train')
                logging.info('iteration %d : mean_dice : %f, val_maxdice : %f' % (iter_num, performance, best_performance))
                logging.info(
                    'resnet iteration %d : mean_dice : %f, val_maxdice : %f' % (iter_num, performance2, best_performance2))
                logging.info('mean iteration %d : mean_dice : %f, val_maxdice : %f' % (
                    iter_num, performance_mean, best_performance_mean))

                model.train()
                model2.train()



if __name__ == "__main__":
    # if args.deterministic:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/SDCL/ACDC_{}_{}_labeled/pre_train".format(args.exp, 7)
    self_snapshot_path = "./model/SDCL/ACDC_{}_{}_labeled/self_train".format(args.exp, 7)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_train.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
        # cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
        # cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    self_train(args, pre_snapshot_path, self_snapshot_path)




