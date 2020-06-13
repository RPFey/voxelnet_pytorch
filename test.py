import torch.nn as nn
import torch
from torch.autograd import Variable
from config import config as cfg
from data.kitti import KittiDataset
import torch.utils.data as data
import time
from loss import VoxelLoss
from voxelnet import VoxelNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
# from nms.pth_nms import pth_nms
import numpy as np
import torch.backends.cudnn
from test_utils import draw_boxes
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from utils import plot_grad, print_prob
import cv2

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()

def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    gt_box3d_corner = []
    gt_box3d = []

    images = []
    calibs = []
    ids = []
    for i, sample in enumerate(batch):
        voxel_features.append(sample[0])

        voxel_coords.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))

        gt_box3d_corner.append(sample[2])
        gt_box3d.append(sample[3])

        images.append(sample[4])
        calibs.append(sample[5])
        ids.append(sample[6])
    return np.concatenate(voxel_features), \
           np.concatenate(voxel_coords), \
           gt_box3d_corner,\
           gt_box3d,\
           images,\
           calibs, ids

torch.backends.cudnn.enabled=True

import argparse
parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--ckpt', type=str, default=None, help='pre_load_ckpt')
args = parser.parse_args()

def train(net, model_name, hyper, cfg, writer):

    dataset=KittiDataset(cfg=cfg,root='/data/cxg1/VoxelNet_pro/Data',set='val')
    data_loader = data.DataLoader(dataset, batch_size=cfg.N, num_workers=4, collate_fn=detection_collate, shuffle=True, \
                              pin_memory=False)
    
    # define loss function
    criterion = VoxelLoss(alpha=hyper['alpha'], beta=hyper['beta'], gamma=hyper['gamma'])

    running_loss = 0.0
    running_reg_loss = 0.0
    running_conf_loss = 0.0

    # training process
    # batch_iterator = None
    epoch_size = len(dataset) // cfg.N
    print('Epoch size', epoch_size)
    iteration = 0
    for voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, images, calibs, ids in data_loader:

        # wrapper to variable
        voxel_features = torch.tensor(voxel_features).to(cfg.device)

        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(gt_box3d)):
            pos_equal_one_, neg_equal_one_, targets_ = dataset.cal_target(gt_box3d_corner[i], gt_box3d[i], cfg)
            pos_equal_one.append(pos_equal_one_)
            neg_equal_one.append(neg_equal_one_)
            targets.append(targets_)
        
        pos_equal_one = torch.stack(pos_equal_one, dim=0)
        neg_equal_one = torch.stack(neg_equal_one, dim=0)
        targets = torch.stack(targets, dim=0)

        # zero the parameter gradients
        # forward
        score, reg = net(voxel_features, voxel_coords)

        if iteration == 0 : # visualize the first image
            print_prob(score, "pred.png")
            print_prob(pos_equal_one, "gt.png")

        # calculate loss
        conf_loss, reg_loss, xyz_loss, whl_loss, r_loss = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
        loss = hyper['lambda'] * conf_loss + reg_loss

        running_conf_loss += conf_loss.item()
        running_reg_loss += reg_loss.item()
        running_loss += (reg_loss.item() + conf_loss.item())

        pre_image = draw_boxes(reg, score, images, calibs, ids, 'pred')
        gt_image = draw_boxes(targets.float(), pos_equal_one.float(), images, calibs, ids, 'true')
        try :
            writer.add_image("gt_image_box {}".format(iteration), gt_image, global_step=iteration, dataformats='NHWC')
            writer.add_image("predict_image_box {}".format(iteration), pre_image, global_step=iteration, dataformats='NHWC')
        except :
            pass
        iteration += 1

hyper = {'alpha': 1.0,
          'beta': 10.0,
          'pos': 0.6,
          'neg': 0.4,
          'lr':0.01,
          'momentum': 0.9,
          'lambda': 2.0,
          'gamma':2,
          'weight_decay':0.00002}

if __name__ == '__main__':
    cfg.pos_threshold = hyper['pos']
    cfg.neg_threshold = hyper['neg']
    model_name = args.ckpt

    writer = SummaryWriter('runs/test_run')

    net = VoxelNet()
    net.to(cfg.device)
    net.load_state_dict(torch.load(os.path.join('./model',model_name), map_location=cfg.device)['model_state_dict'])
    with torch.no_grad():
        try:
            train(net, model_name, hyper, cfg, writer)
        except KeyboardInterrupt:
            pass
    writer.close()
