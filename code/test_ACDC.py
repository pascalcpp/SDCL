import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm   

from networks.net_factory import BCP_net

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/acdc', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SDCL', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric
def test_single_volume_average(case, net1, net2, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out_main1 = net1(input)
            if len(out_main1)>1:
                out_main1=out_main1[0]

            out_main2 = net2(input)
            if len(out_main2)>1:
                out_main2=out_main2[0]

            out = torch.argmax((torch.softmax(out_main1, dim=1) + torch.softmax(out_main2, dim=1)) / 2, dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


import csv
def TESTACDC(iter=-1, phase='pre_train'):
    FLAGS = parser.parse_args()
    FLAGS.stage_name = phase
    with open(FLAGS.root_path + '/data_split/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/SDCL/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/SDCL/ACDC_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)



    net1 = BCP_net(model="UNet", in_chns=1, class_num=4)
    net2 = BCP_net(model="ResUNet", in_chns=1, class_num=4)



    model_path1 = os.path.join(snapshot_path, 'best_model.pth')
    model_path2 = os.path.join(snapshot_path, 'best_model_res.pth')

    net1.load_state_dict(torch.load(str(model_path1))['net'])
    net2.load_state_dict(torch.load(str(model_path2))['net'])

    net1.eval()
    net2.eval()

    first_total1 = 0.0
    second_total1 = 0.0
    third_total1 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net1, test_save_path, FLAGS)
        first_total1 += np.asarray(first_metric)
        second_total1 += np.asarray(second_metric)
        third_total1 += np.asarray(third_metric)
    avg_metric1 = [first_total1 / len(image_list), second_total1 / len(image_list), third_total1 / len(image_list)]

    first_total2 = 0.0
    second_total2 = 0.0
    third_total2 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net2, test_save_path, FLAGS)
        first_total2 += np.asarray(first_metric)
        second_total2 += np.asarray(second_metric)
        third_total2 += np.asarray(third_metric)
    avg_metric2 = [first_total2 / len(image_list), second_total2 / len(image_list), third_total2 / len(image_list)]


    first_total3 = 0.0
    second_total3 = 0.0
    third_total3 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume_average(case, net1, net2, test_save_path, FLAGS)
        first_total3 += np.asarray(first_metric)
        second_total3 += np.asarray(second_metric)
        third_total3 += np.asarray(third_metric)
    avg_metric3 = [first_total3 / len(image_list), second_total3 / len(image_list), third_total3 / len(image_list)]

    with open('./' + '_ssl_test.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iter_num', 'unet_dice', 'resnet_dice', 'average'])
        writer.writerow([iter, (avg_metric1[0]+avg_metric1[1]+avg_metric1[2])/3, (avg_metric2[0]+avg_metric2[1]+avg_metric2[2])/3, (avg_metric3[0]+avg_metric3[1]+avg_metric3[2])/3])


def Inference(FLAGS):
    with open(FLAGS.root_path + '/data_split/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/SDCL/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/SDCL/ACDC_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net1 = BCP_net(model="UNet", in_chns=1, class_num=4)
    net2 = BCP_net(model="ResUNet", in_chns=1, class_num=4)



    model_path1 = os.path.join(snapshot_path, 'best_model.pth')
    model_path2 = os.path.join(snapshot_path, 'best_model_res.pth')

    net1.load_state_dict(torch.load(str(model_path1))['net'])
    net2.load_state_dict(torch.load(str(model_path2))['net'])

    net1.eval()
    net2.eval()

    first_total1 = 0.0
    second_total1 = 0.0
    third_total1 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net1, test_save_path, FLAGS)
        first_total1 += np.asarray(first_metric)
        second_total1 += np.asarray(second_metric)
        third_total1 += np.asarray(third_metric)
    avg_metric1 = [first_total1 / len(image_list), second_total1 / len(image_list), third_total1 / len(image_list)]

    first_total2 = 0.0
    second_total2 = 0.0
    third_total2 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net2, test_save_path, FLAGS)
        first_total2 += np.asarray(first_metric)
        second_total2 += np.asarray(second_metric)
        third_total2 += np.asarray(third_metric)
    avg_metric2 = [first_total2 / len(image_list), second_total2 / len(image_list), third_total2 / len(image_list)]

    first_total3 = 0.0
    second_total3 = 0.0
    third_total3 = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume_average(case, net1, net2, test_save_path, FLAGS)
        first_total3 += np.asarray(first_metric)
        second_total3 += np.asarray(second_metric)
        third_total3 += np.asarray(third_metric)
    avg_metric3 = [first_total3 / len(image_list), second_total3 / len(image_list), third_total3 / len(image_list)]

    print("unet")
    print(avg_metric1)
    print((avg_metric1[0]+avg_metric1[1]+avg_metric1[2])/3)

    print("resunet")
    print(avg_metric2)
    print((avg_metric2[0]+avg_metric2[1]+avg_metric2[2])/3)

    print("average")
    print(avg_metric3)
    print((avg_metric3[0]+avg_metric3[1]+avg_metric3[2])/3)
    # return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
