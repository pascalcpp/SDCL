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
seed_test = 2020
seed_reproducer(seed = seed_test)

data_root, split_name = '../Datasets/pancreas/data', 'pancreas'
result_dir = 'result/pancreas/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 101, 321
pretrain_save_step, st_save_step, pred_step = 10, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

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


def pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice1 = 0
    max_dice2 = 0
    measures = CutPreMeasures(writer, logger)

    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % 5 == 0:
            net1.eval()
            net2.eval()
            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net_opt(net1, optimizer1, save_path / f'best_ema{label_percent}_pre_vnet.pth', epoch)
                max_dice1 = val_dice1

            if val_dice2 > max_dice2:
                save_net_opt(net2, optimizer2, save_path / f'best_ema{label_percent}_pre_resnet.pth', epoch)
                max_dice2 = val_dice2

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))

        """Training"""
        net1.train()
        net2.train()
        logger.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out1 = net1(img)[0]
            ce_loss1 = F.cross_entropy(out1, lab)
            dice_loss1 = DICE(out1, lab)
            loss1 = (ce_loss1 + dice_loss1) / 2

            out2 = net2(img)[0]
            ce_loss2 = F.cross_entropy(out2, lab)
            dice_loss2 = DICE(out2, lab)
            loss2 = (ce_loss2 + dice_loss2) / 2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            logger.info("cur epoch: %d step: %d" % (epoch, step+1))
            logger.info("vnet")
            measures.update(out1, lab, ce_loss1, dice_loss1, loss1)
            logger.info("resnet")
            measures.update(out2, lab, ce_loss2, dice_loss2, loss2)
            measures.log(epoch, epoch * len(lab_loader_a) + step)


    return max_dice1

def ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):

    def get_XOR_region(mixout1, mixout2):
        s1 = torch.softmax(mixout1, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        s2 = torch.softmax(mixout2, dim = 1)
        l2 = torch.argmax(s2, dim = 1)

        diff_mask = (l1 != l2)
        return diff_mask

    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    load_net_opt(net2, optimizer2, pretrained_path / f'best_ema{label_percent}_pre_resnet.pth')
    load_net_opt(ema_net1, optimizer1, pretrained_path / f'best_ema{label_percent}_pre_vnet.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice1 = 0
    max_list1 = None
    max_dice2 = 0
    max_dice3 = 0
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if (epoch % 20 == 0) | ((epoch >= 160) & (epoch % 5 ==0)):

            net1.eval()
            net2.eval()

            avg_metric1, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=16, s_z=4)
            avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=16, s_z=4)

            logger.info('average metric is : {}'.format(avg_metric1))
            logger.info('average metric is : {}'.format(avg_metric2))
            logger.info('mean average metric is : {}'.format(avg_metric3))

            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]
            val_dice3 = avg_metric3[0]

            if val_dice1 > max_dice1:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice1 = val_dice1
                max_list1 = avg_metric1

            if val_dice2 > max_dice2:
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_resnet.pth'))
                max_dice2 = val_dice2


            if val_dice3 > max_dice3:
                save_net(net1, str(save_path / f'best_ema_{label_percent}_self_v.pth'))
                save_net(net2, str(save_path / f'best_ema_{label_percent}_self_r.pth'))

                max_dice3 = val_dice3

            logger.info('\nEvaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice1, max_dice1))
            logger.info('resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice2, max_dice2))
            logger.info('mean Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice3, max_dice3))

        """Training"""
        net1.train()
        net2.train()
        ema_net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out_1 = ema_net1(unimg_a)[0]
                unimg_b_out_1 = ema_net1(unimg_b)[0]

                uimg_a_plab = get_cut_mask(unimg_a_out_1, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out_1, nms=True, connect_mode=connect_mode)


                img_mask, loss_mask = generate_mask(img_a, patch_size)


            """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)

            """BCP"""
            """Supervised Loss"""
            mix_lab_out = net1(net3_input_l)
            mix_output_l = mix_lab_out[0]
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_unlab_out = net1(net3_input_unlab)
            mix_output_2 = mix_unlab_out[0]
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)


            """Supervised Loss"""
            mix_output_l_2 = net2(net3_input_l)[0]
            loss_1_2 = mix_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2_2 = net2(net3_input_unlab)[0]
            loss_2_2 = mix_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask)

            """SDCL"""

            with torch.no_grad():
                diff_mask1 = get_XOR_region(mix_output_l, mix_output_l_2)
                diff_mask2 = get_XOR_region(mix_output_2, mix_output_2_2)

            net1_mse_loss_lab = mix_mse_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net1_kl_loss_lab = mix_max_kl_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net1_kl_loss_unlab = mix_max_kl_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)
            net2_kl_loss_lab = mix_max_kl_loss(mix_output_l_2, uimg_a_plab.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)
            net2_kl_loss_unlab = mix_max_kl_loss(mix_output_2_2, lab_a, uimg_b_plab.long(), loss_mask, diff_mask=diff_mask2)

            loss1 = loss_1 + loss_2 + 0.3 * (net1_mse_loss_lab + net1_mse_loss_unlab) + 0.1 * (net1_kl_loss_lab + net1_kl_loss_unlab)

            loss2 = loss_1_2 + loss_2_2 + 0.3 * (net2_mse_loss_lab + net2_mse_loss_unlab) + 0.1 * (net2_kl_loss_lab + net2_kl_loss_unlab)

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            update_ema_variables(net1, ema_net1, alpha)

            logger.info("loss_1: %.4f, loss_2: %.4f, net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, net1_kl_loss_lab: %.4f, net1_kl_loss_unlab: %.4f," % 
                (loss_1.item(), loss_2.item(), net1_mse_loss_lab.item(), net1_mse_loss_unlab.item(),
                    net1_kl_loss_lab.item(), net1_kl_loss_unlab.item()))

        if epoch == self_training_epochs:
            save_net(net1, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
    return max_dice1, max_list1

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
        pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader)
        seed_reproducer(seed = seed_test)
        ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        test_model(net1, net2, test_loader)

    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


