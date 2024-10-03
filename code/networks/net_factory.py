from networks.unet import UNet, UNet_2d
from networks.ResNet2d import ResUNet_2d
import torch.nn as nn



def BCP_net(model = "UNet", in_chns=1, class_num=2, ema=False):

    if model == "UNet":
        net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
    elif model == "ResUNet":
        net = ResUNet_2d(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
    return net

