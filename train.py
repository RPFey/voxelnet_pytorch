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
from utils import plot_grad
import cv2


import argparse

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--ckpt', type=str, default=None, help='pre_load_ckpt')
parser.add_argument('--index', type=int, default=None, help='hyper_tag')
parser.add_argument('--epoch', type=int , default=160, help="training epoch")
args = parser.parse_args()

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

def train(net, model_name, hyper, cfg, writer, optimizer):

    dataset=KittiDataset(cfg=cfg,root='/data/cxg1/VoxelNet_pro/Data',set='train')
    data_loader = data.DataLoader(dataset, batch_size=cfg.N, num_workers=4, collate_fn=detection_collate, shuffle=True, \
                              pin_memory=False)

    net.train()

    # define optimizer
    
    # define loss function
    criterion = VoxelLoss(alpha=hyper['alpha'], beta=hyper['beta'], gamma=hyper['gamma'])

    running_loss = 0.0
    running_reg_loss = 0.0
    running_conf_loss = 0.0

    # training process
    # batch_iterator = None
    epoch_size = len(dataset) // cfg.N
    print('Epoch size', epoch_size)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epoch*x) for x in (0.7, 0.9)], gamma=0.1)
    scheduler.last_epoch = cfg.last_epoch + 1
    optimizer.zero_grad()

    epoch = cfg.last_epoch
    while epoch < args.epoch :
        iteration = 0
        for voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, images, calibs, ids in data_loader:

            # wrapper to variable
            voxel_features = torch.tensor(voxel_features).to(cfg.device)

            pos_equal_one = []
            neg_equal_one = []
            targets = []

            with torch.no_grad():
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

            # calculate loss
            conf_loss, reg_loss, _, _, _ = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
            loss = hyper['lambda'] * conf_loss + reg_loss

            running_conf_loss += conf_loss.item()
            running_reg_loss += reg_loss.item()
            running_loss += (reg_loss.item() + conf_loss.item())

            # backward
            loss.backward()

            # visualize gradient
            if iteration == 0 and epoch % 30 == 0:
                plot_grad(net.svfe.vfe_1.fcn.linear.weight.grad.view(-1), epoch,  "vfe_1_grad_%d"%(epoch))
                plot_grad(net.svfe.vfe_2.fcn.linear.weight.grad.view(-1), epoch,"vfe_2_grad_%d"%(epoch))
                plot_grad(net.cml.conv3d_1.conv.weight.grad.view(-1), epoch,"conv3d_1_grad_%d"%(epoch))
                plot_grad(net.rpn.reg_head.conv.weight.grad.view(-1), epoch,"reghead_grad_%d"%(epoch))
                plot_grad(net.rpn.score_head.conv.weight.grad.view(-1), epoch,"scorehead_grad_%d"%(epoch))

            # update
            if iteration%10 == 9:
                for param in net.parameters():
                    param.grad /= 10
                optimizer.step()
                optimizer.zero_grad()

            if iteration % 50 == 49:
                writer.add_scalar('total_loss', running_loss/50.0, epoch * epoch_size + iteration)
                writer.add_scalar('reg_loss', running_reg_loss/50.0, epoch * epoch_size + iteration)
                writer.add_scalar('conf_loss',running_conf_loss/50.0, epoch * epoch_size + iteration)

                print("epoch : " + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss: %.4f || Loc Loss: %.4f || Conf Loss: %.4f' % \
                ( running_loss/50.0, running_reg_loss/50.0, running_conf_loss/50.0))

                running_conf_loss = 0.0
                running_loss = 0.0
                running_reg_loss = 0.0

            # visualization
            if iteration == 2000:
                reg_de = reg.detach()
                score_de = score.detach()
                with torch.no_grad():
                    pre_image = draw_boxes(reg_de, score_de, images, calibs, ids, 'pred')
                    gt_image = draw_boxes(targets.float(), pos_equal_one.float(), images, calibs, ids, 'true')
                    try :
                        writer.add_image("gt_image_box {}".format(epoch), gt_image, global_step=epoch * epoch_size + iteration, dataformats='NHWC')
                        writer.add_image("predict_image_box {}".format(epoch), pre_image, global_step=epoch * epoch_size + iteration, dataformats='NHWC')
                    except :
                        pass
            iteration += 1
        scheduler.step()
        epoch += 1
        if epoch % 30 == 0:
            torch.save({
                "epoch": epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('./model', model_name+str(epoch)+'.pt'))

hyper = {'alpha': 1.0,
          'beta': 10.0,
          'pos': 0.75,
          'neg': 0.5,
          'lr':0.005,
          'momentum': 0.9,
          'lambda': 2.0,
          'gamma':2,
          'weight_decay':0.00001}

if __name__ == '__main__':
    pre_model = args.ckpt

    cfg.pos_threshold = hyper['pos']
    cfg.neg_threshold = hyper['neg']
    model_name = "model_%d"%(args.index+1)

    writer = SummaryWriter('runs/%s'%(model_name[:-4]))

    net = VoxelNet()
    net.to(cfg.device)
    optimizer = optim.SGD(net.parameters(), lr=hyper['lr'], momentum = hyper['momentum'], weight_decay=hyper['weight_decay'])

    if pre_model is not None and os.path.exists(os.path.join('./model',pre_model)) :
        ckpt = torch.load(os.path.join('./model',pre_model), map_location=cfg.device)
        net.load_state_dict(ckpt['model_state_dict'])
        cfg.last_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else :
        net.apply(weights_init)     
    train(net, model_name, hyper, cfg, writer, optimizer)
    writer.close()
