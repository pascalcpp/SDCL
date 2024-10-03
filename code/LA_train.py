from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
# from dataloaders.dataset import *
from dataloaders.LADataset import LAHeart
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables
from utils.LA_utils import to_cuda
from utils.BCP_utils import *
from pancreas.losses import *

from pancreas.Vnet import VNet
from networks.ResVNet import ResVNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/la/data', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='SDCL', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()


def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_ResVnet(ema=False):
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def get_cut_mask_two(out1, out2, thres=0.5, nms=0):
    probs1 = F.softmax(out1, 1)
    probs2 = F.softmax(out2, 1)
    probs = (probs1 + probs2) / 2

    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def pre_train(args, snapshot_path):
    model = create_Vnet()
    model2 = create_ResVnet()

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    model2.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    max_epoch = 81
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            outputs2, _ = model2(volume_batch)
            loss_ce2 = F.cross_entropy(outputs2, label_batch)
            loss_dice2 = DICE(outputs2, label_batch)
            loss2 = (loss_ce2 + loss_dice2) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

        if epoch_num % 5 == 0:
            model.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path, epoch_num)
                save_net_opt(model, optimizer, save_best_path, epoch_num)
                logging.info("save best model to {}".format(save_mode_path))

            model.train()

            model2.eval()
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_resnet.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(snapshot_path, 'best_model_resnet.pth'.format(args.model))
                save_net_opt(model2, optimizer2, save_mode_path, epoch_num)
                save_net_opt(model2, optimizer2, save_best_path, epoch_num)
                logging.info("save best resnet model to {}".format(save_mode_path))
            model2.train()



def self_train(args, pre_snapshot_path, self_snapshot_path):
    model1 = create_Vnet()
    model2 = create_ResVnet()
    ema_model1 = create_Vnet(ema=True).cuda()



    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)


    pretrained_model = os.path.join(pre_snapshot_path, 'best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_resnet.pth')

    load_net_opt(model1, optimizer, pretrained_model)
    load_net_opt(model2, optimizer2, pretrained_model2)

    load_net_opt(ema_model1, optimizer, pretrained_model)


    model1.train()
    model2.train()
    ema_model1.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    mean_best_dice = 0
    max_epoch = 276
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():

                unoutput_a_1, _ = ema_model1(unimg_a)
                unoutput_b_1, _ = ema_model1(unimg_b)


                plab_a = get_cut_mask(unoutput_a_1, nms=1)
                plab_b = get_cut_mask(unoutput_b_1, nms=1)

                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            mixl_img = unimg_a * img_mask + img_b * (1 - img_mask)
            mixu_img = img_a * img_mask + unimg_b * (1 - img_mask)


            outputs_l, _ = model1(mixl_img)
            outputs_u, _ = model1(mixu_img)
            loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            outputs_l_2, _ = model2(mixl_img)
            outputs_u_2, _ = model2(mixu_img)
            loss_l_2 = mix_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_u_2 = mix_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            with torch.no_grad():
                diff_mask1 = get_XOR_region(outputs_l, outputs_l_2)
                diff_mask2 = get_XOR_region(outputs_u, outputs_u_2)

            net1_mse_loss_lab = mix_mse_loss(outputs_l, plab_a.long(), lab_b, loss_mask, unlab=True,
                                             diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(outputs_l, plab_a.long(), lab_b, loss_mask, unlab=True,
                                               diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(outputs_u, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(outputs_u, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, unlab=True,
                                             diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(outputs_l_2, plab_a.long(), lab_b, loss_mask, unlab=True,
                                               diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(outputs_u_2, lab_a, plab_b.long(), loss_mask, diff_mask=diff_mask2)

            loss = (loss_l + loss_u) + 0.5 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.05 * (
                        net1_kl_loss_lab + net1_kl_loss_unlab)

            loss_2 = (loss_l_2 + loss_u_2) + 0.5 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.05 * (
                        net2_kl_loss_lab + net2_kl_loss_unlab)


            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            logging.info('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f \
               net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f \
               ' % (epoch, iter_num, loss, loss_l, loss_u, net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                    net1_kl_loss_lab.item(), net1_kl_loss_unlab.item()))

            update_ema_variables(model1, ema_model1, 0.99)

        if epoch % 5 == 0:
            model1.eval()
            model2.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            mean_dice_sample = test_3d_patch.var_all_case_LA_mean(model1, model2, num_classes=num_classes,
                                                                  patch_size=patch_size, stride_xy=18, stride_z=4)

            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
                logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))

            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(self_snapshot_path,
                                              'iter_{}_dice_{}_res.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(self_snapshot_path, 'best_model_res.pth')
                torch.save(model2.state_dict(), save_mode_path)
                torch.save(model2.state_dict(), save_best_path)
                logging.info("resnet cur dice %.4f, max dice %.4f" % (dice_sample2, best_dice2))

            if mean_dice_sample > mean_best_dice:
                mean_best_dice = round(mean_dice_sample, 4)
                save_mode_path1 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_v.pth'.format(iter_num, mean_best_dice))
                save_best_path1 = os.path.join(self_snapshot_path, 'best_model_v.pth')

                save_mode_path2 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_r.pth'.format(iter_num, mean_best_dice))
                save_best_path2 = os.path.join(self_snapshot_path, 'best_model_r.pth')

                torch.save(model1.state_dict(), save_mode_path1)
                torch.save(model1.state_dict(), save_best_path1)

                torch.save(model2.state_dict(), save_mode_path2)
                torch.save(model2.state_dict(), save_best_path2)

                logging.info("mean save best model to {}".format(save_mode_path1))
                logging.info("mean cur dice %.4f, max dice %.4f" % (mean_dice_sample, mean_best_dice))

            model1.train()
            model2.train()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/SDCL/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/SDCL/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting SDCL training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/LA_train.py', self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    self_train(args, pre_snapshot_path, self_snapshot_path)

