from __future__ import division
import os
import os.path
import torch.utils.data as data
import utils
from utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from data_aug import aug_data
from box_overlaps import bbox_overlaps
import numpy as np
import cv2
import torch
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

class KittiDataset(data.Dataset):
    def __init__(self, cfg, root='./KITTI',set='train',type='velodyne_train'):
        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "crop")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib")
        self.label_path = os.path.join(self.data_path, "label_2")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

        self.T = cfg.T
        self.vd = cfg.vd
        self.vh = cfg.vh
        self.vw = cfg.vw
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        self.anchors = torch.tensor(cfg.anchors.reshape(-1,7)).float().to(cfg.device)
        self.anchors_xylwr = self.anchors[..., [0, 1, 5, 4, 6]].contiguous()
        self.feature_map_shape = (int(cfg.H / 2), int(cfg.W / 2))
        self.anchors_per_position = cfg.anchors_per_position
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold

    def cal_target(self, gt_box3d, gt_xyzhwlr, cfg):
        # Input:
        #   labels: (N,)
        #   feature_map_shape: (w, l)
        #   anchors: (w, l, 2, 7)
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview

        anchors_d = torch.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2).to(cfg.device)
        # denote whether the anchor box is pos or neg
        pos_equal_one = torch.zeros((*self.feature_map_shape, 2)).to(cfg.device)
        neg_equal_one = torch.zeros((*self.feature_map_shape, 2)).to(cfg.device)

        targets = torch.zeros((*self.feature_map_shape, 14)).to(cfg.device)

        gt_xyzhwlr = torch.tensor(gt_xyzhwlr, requires_grad=False).float().to(cfg.device)
        gt_xylwr = gt_xyzhwlr[..., [0, 1, 5, 4, 6]]
        
        # BOTTLENECK

        iou = pairwise_iou_rotated(
            self.anchors_xylwr,
            gt_xylwr.contiguous()
        ).cpu().numpy() # (gt - anchor)

        id_highest = np.argmax(iou.T, axis=1)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0 # make sure all the iou is positive
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold,
                                 axis=1) == iou.shape[1])[0] # anchor doesn't match ant ground truth

        for gt in range(iou.shape[1]):
            if gt not in id_pos_gt and iou[id_highest[gt], gt] > self.neg_threshold:
                id_pos = np.append(id_pos, id_highest[gt])
                id_pos_gt = np.append(id_pos_gt, gt)

        # sample the negative points to keep ratio as 1:10 with minimum 500
        num_neg = 10 * id_pos.shape[0]
        if num_neg < 500:
            num_neg = 500
        if id_neg.shape[0] > num_neg:
            np.random.shuffle(id_neg)
            id_neg = id_neg[:num_neg]
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.feature_map_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # ATTENTION: index_z should be np.array

        # parameterize the ground truth box relative to anchor boxs
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = torch.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = torch.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = torch.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1

        return pos_equal_one, neg_equal_one, targets

    def preprocess(self, lidar):
        # This func cluster the points in the same voxel.
        # shuffling the points
        np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)

        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords

    def __getitem__(self, i):

        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'

        calib = utils.load_kitti_calib(calib_file)
        Tr = calib['Tr_velo2cam']
        gt_box3d_corner, gt_box3d = utils.load_kitti_label(label_file, Tr)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


        if self.type == 'velodyne_train':
            image = cv2.imread(image_file)

            # data augmentation
            # lidar, gt_box3d = aug_data(lidar, gt_box3d)

            # specify a range
            lidar, gt_box3d_corner, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d_corner, gt_box3d)

            # voxelize
            voxel_features, voxel_coords = self.preprocess(lidar)

            # bounding-box encoding
            # pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d_corner, gt_box3d)

            return voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, image, calib, self.file_list[i] # pos_equal_one, neg_equal_one, targets, image, calib, self.file_list[i]

        elif self.type == 'velodyne_test':
            image = cv2.imread(image_file)

            lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)

            voxel_features, voxel_coords = self.preprocess(lidar)

            return voxel_features, voxel_coords, image, calib, self.file_list[i]

        else:
            raise ValueError('the type invalid')


    def __len__(self):
        return len(self.file_list)




