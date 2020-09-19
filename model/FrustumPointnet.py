import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.modelbase import ModelBase, init_weight
import torch.nn.functional as F
from model.pointnet2_msg_seg import PointNet2SemSegMSG, CenterRegressionNet, BoxEstimationNet
from utils.loss_util import huber_loss
from utils.utils import get_box3d_corners_helper, class2angle
import numpy as np
from detectron2.layers.rotated_boxes import pairwise_iou_rotated as rotated_iou
from configs.config import type2class, type_mean_size

class FrustumPointnet(nn.Module):
    def __init__(self, config):
        super(FrustumPointnet, self).__init__()
        self.config = config
        self.Pointnet2_seg = PointNet2SemSegMSG(self.config)
        self.CenterRegressionNet = CenterRegressionNet(self.config)
        self.CenterRegressionNet.build_netork()
        self.BoxEstimation = BoxEstimationNet(self.config)
        self.BoxEstimation.build_network()

        self.type2class = type2class
        self.class2type = {type2class[i] : i for i in self.type2class}
        self.type_mean_size = type_mean_size
        self.mean_size_arr = np.zeros((self.config['num_size_cluster'], 3)) # (NUM_SIZE_CLUSTER, 3)
        for i in range(self.config['num_size_cluster']):
            self.mean_size_arr[i, :] = np.array(self.type_mean_size[self.class2type[i]])

    def forward(self, pointcloud, onehot):
        """
        Input:
            pointcloud (B, N, 3 + feature_channel)
            oone_hot: shape (B, num_cls)
        Return:
            endpoint (dict)
                'prob': prob (B, 2, N)

                'center': center # B * 3
                'stage1_center': stage1_center # B * 3
                ‘heading_score': heading_score # B x NUM_HEADING_BIN
                'heading_residuals_normalized': heading_residuals_normalized # BxNUM_HEADING_BIN
                'heading_residuals': heading_residuals # BxNUM_HEADING_BIN
                'size_scores': size_scores # BxNUM_SIZE_CLUSTER
                'size_residuals_normalized': size_residuals_normalized  B , NUM_SIZE_CLUSTER , 3
                'size_residuals': size_residuals  # B, NUM_SIZE_CLUSTER, 3
        """
        if type(self.mean_size_arr) != torch.Tensor:
            self.mean_size_arr = torch.tensor(self.mean_size_arr, dtype=torch.float32, device=pointcloud.device)

        feats, prob = self.Pointnet2_seg(pointcloud, onehot) # feats : (B, 128, N) prob : (B, 2, N)
        object_point_cloud, mask_xyz_mean = point_cloud_masking(pointcloud, prob, self.config['object_points']) # (B, N, 3), (B, 3)
        center_delta = self.CenterRegressionNet(object_point_cloud, onehot) # (B, 3)
        stage1_center = mask_xyz_mean + center_delta
        object_coor_new = object_point_cloud - stage1_center.unsqueeze(1) # (B, N, 3)
        output = self.BoxEstimation(object_coor_new, onehot)
        pred_dict, boxnet_center = self.parse_output_to_tensors(output)

        center = boxnet_center + stage1_center
        endpoint = {
            'prob': prob,
            'center': center,
            'stage1_center': stage1_center
        }

        endpoint.update(pred_dict)
        return endpoint

    def parse_output_to_tensors(self, output:torch.Tensor):
        """
        Input: Parse batch output to separate tensors
            output: tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        Output:
            end_points: dict
        """
        batch = output.shape[0]
        center = output[:, :3] # B * 3
        heading_score = output[:, 3: 3 + self.config['num_heading_bin']] # BxNUM_HEADING_BIN
        heading_residuals_normalized = output[:, 3 + self.config['num_heading_bin']: 3 + 2*self.config['num_heading_bin']] # BxNUM_HEADING_BIN
        heading_residuals = heading_residuals_normalized * ( np.pi / self.config['num_heading_bin'] )

        size_scores = output[:, 3 + 2*self.config['num_heading_bin']: 3 + 2*self.config['num_heading_bin'] + self.config['num_size_cluster']] # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = output[:, 3 + 2*self.config['num_heading_bin'] + self.config['num_size_cluster']: 3 + 2*self.config['num_heading_bin'] + 4*self.config['num_size_cluster']] # Bx (3*NUM_SIZE_CLUSTER)
        size_residuals_normalized = size_residuals_normalized.reshape((batch, self.config['num_size_cluster'], 3)) # B , NUM_SIZE_CLUSTER , 3
        size_residuals = size_residuals_normalized * self.mean_size_arr.unsqueeze(0)

        dict = {'heading_score': heading_score, 'heading_residuals_normalized': heading_residuals_normalized, 'heading_residuals': heading_residuals,
                'size_scores': size_scores, 'size_residuals_normalized': size_residuals_normalized, 'size_residuals': size_residuals}

        return dict, center

    def get_box3d_corners(self, center: torch.Tensor, heading_residual: torch.Tensor, size_residual: torch.Tensor):
        """
        Inputs:
            center: (bs,3)
            heading_residual: (bs, num_heading_bin)
            size_residual: (bs, num_size_cluster, 3)
        Outputs:
            box3d_corners: tensor (bs, NH, NS, 8, 3) 
        """
        bs = center.shape[0]
        heading_bin_centers = torch.arange(0, 2 * np.pi, 2*np.pi / self.config['num_heading_bin'], device=center.device)
        headings = heading_residual + heading_bin_centers.view(1, -1) # (bs,12)

        mean_sizes = self.mean_size_arr.unsqueeze(0) + size_residual  # (1,8,3)+(bs,8,3) = (bs,8,3)
        sizes = mean_sizes + size_residual  # (bs,8,3)
        sizes = sizes.view(bs, 1, self.config['num_size_cluster'], 3).repeat(1, self.config['num_heading_bin'], 1, 1).float()  # (B,12,8,3)
        headings = headings.view(bs, self.config['num_heading_bin'], 1).repeat(1, 1, self.config['num_size_cluster'])  # (bs,12,8)
        centers = center.view(bs, 1, 1, 3).repeat(1, self.config['num_heading_bin'], self.config['num_size_cluster'], 1)  # (bs,12,8,3)
        N = bs * self.config['num_heading_bin'] * self.config['num_size_cluster']
        corners_3d = get_box3d_corners_helper(centers.view(N, 3), headings.view(N), sizes.view(N, 3))
        return corners_3d.view(bs, self.config['num_heading_bin'], self.config['num_size_cluster'], 8, 3)  # [32, 12, 8, 8, 3]

    def class2size(self, pred_cls, residual):
        ''' 
        Inverse function to size2class.
        pred_cls: tensor B, classes of each size
        residual: tensor B, 3 
        return:
            size B, 3
        '''
        mean_size = self.mean_size_arr[pred_cls]
        return mean_size + residual


# Lightening version of Frustum network
class FrustumLight(ModelBase):
    def __init__(self, config, args):
        super(FrustumLight, self).__init__(config)
        self.net = FrustumPointnet(config)
        self.total_epoch = int(args.max_epochs)

    def forward(self, pointcloud, onehot):
        return self.net(pointcloud, onehot)

    def loss(self, mask_label, center_label, heading_class_label, heading_residual_label,
             size_class_label, size_residual_label, endpoint):
        """ Loss functions for 3D object detection.
        Input:
            mask_label: int32 tensor in shape (B, N)
            center_label: tensor in shape (B,3)
            heading_class_label: int32 tensor in shape (B,)
            heading_residual_label: tensor in shape (B,)
            size_class_label: tensor int32 in shape (B,)
            size_residual_label: tensor tensor in shape (B,)
            endpoint: dict, outputs from our model
        Output:
            loss_dict: dict of each loss, the names can be found in the yaml file
            total_loss: scalar tensor
                the total_loss is also added to the losses collection
        """
        batch = mask_label.shape[0]
        mask_loss = F.cross_entropy(endpoint['prob'], mask_label)

        center_dist = torch.norm(center_label - endpoint['center'], dim=1, p=2) #(N, )
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center_label - endpoint['stage1_center'], dim=1, p=2)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        heading_class_softmax = F.softmax(endpoint['heading_score'], dim=1)
        heading_class_loss = F.cross_entropy(heading_class_softmax, heading_class_label)

        hcls_onehot = torch.eye(self.config['num_heading_bin'], device=mask_label.device)[heading_class_label.long()] # (B, NUM_HEADING_BIN)
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / self.config['num_heading_bin'])
        heading_residual_normalized_loss = huber_loss(
            torch.sum(endpoint['heading_residuals_normalized'] * hcls_onehot.float(), dim=1) - heading_residual_normalized_label,
            delta=1.0
        )

        size_softmax = F.softmax(endpoint['size_scores'], dim=1)
        size_class_loss = F.cross_entropy(size_softmax, size_class_label)

        scls_onehot = torch.eye(self.config['num_size_cluster'], device=mask_label.device)[size_class_label.long()] # BxNUM_SIZE_CLUSTER
        scls_onehot_tiled = scls_onehot.unsqueeze(-1).repeat(1, 1, 3) # B, NUM_SIZE_CLUSTER, 3
        predicted_size_residual_normalized = torch.sum(
            endpoint['size_residuals_normalized'] * scls_onehot_tiled, dim=1
        ) # B, 3

        mean_size_arr = self.net.mean_size_arr.unsqueeze(0) # 1, NUM_SIZE_CLUSTER, 3
        mean_size_label = torch.sum(
            scls_onehot_tiled * mean_size_arr, dim=1
        ) # Bx3
        size_residual_label_normalized = size_residual_label / mean_size_label # Bx3
        size_normalized_dist = torch.norm(
            size_residual_label_normalized - predicted_size_residual_normalized, dim=1, p=2
        )
        size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

        corners_3d = self.net.get_box3d_corners(endpoint['center'], endpoint['heading_residuals'], endpoint['size_residuals'])  # (bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        gt_mask = hcls_onehot.unsqueeze(2).repeat(1, 1, self.config['num_size_cluster'])*\
                  scls_onehot.unsqueeze(1).repeat(1, self.config['num_heading_bin'], 1)  # Bxnum_heading_binxNUM_SIZE_CLUSTER
        corners_3d_pred = torch.sum(
            gt_mask.view(batch, self.config['num_heading_bin'], self.config['num_size_cluster'], 1, 1).float() * corners_3d,
            dim=[1, 2]
        )  # (bs,8,3)
        heading_bin_centers = torch.arange(
            0, 2 * np.pi, 2 * np.pi / self.config['num_heading_bin'], device=heading_class_label.device
        ).float()
        heading_label = heading_residual_label.view(batch, 1) + \
                        heading_bin_centers.view(1, self.config['num_heading_bin'])
        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        size_label = mean_size_arr + size_residual_label.unsqueeze(1)
        size_label = torch.sum(
            scls_onehot.view(batch, self.config['num_size_cluster'], 1).float() * size_label, dim=1
        )

        corners_3d_gt = get_box3d_corners_helper(
            center_label, heading_label, size_label
        )

        corners_3d_gt_flip = get_box3d_corners_helper(
            center_label, heading_label + np.pi, size_label
        )

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        loss_dict = {
            'mask_loss': mask_loss,
            'center_loss': center_loss,
            'stage1_center_loss': stage1_center_loss,
            'heading_class_loss': heading_class_loss,
            'heading_residual_normalized_loss': heading_residual_normalized_loss,
            'size_residual_normalized_loss': size_residual_normalized_loss,
            'size_class_loss': size_class_loss,
            'corners_loss': corners_loss
        }

        total_loss = 0
        for item in loss_dict.keys():
            total_loss += self.config['LOSS'][item] * loss_dict[item]

        return loss_dict, total_loss

    def compute_box3d_iou(self, center_pred,
                        heading_logits, heading_residuals,
                        size_logits, size_residuals,
                        center_label,
                        heading_class_label, heading_residual_label,
                        size_class_label, size_residual_label):
        ''' Compute 3D bounding box IoU from network output and labels.
        All inputs are pytorch tensors.

        Inputs:
            center_pred: (B,3)
            heading_logits: (B,NUM_HEADING_BIN)
            heading_residuals: (B,NUM_HEADING_BIN)
            size_logits: (B,NUM_SIZE_CLUSTER)
            size_residuals: (B,NUM_SIZE_CLUSTER,3)
            center_label: (B,3)
            heading_class_label: (B,)
            heading_residual_label: (B,)
            size_class_label: (B,)
            size_residual_label: (B,3)
        Output:
            iou2ds: (B,) birdeye view oriented 2d box ious
            iou3ds: (B,) 3d box ious
        '''
        batch_size = heading_logits.shape[0]
        heading_class = torch.argmax(heading_logits, 1) # B, 
        heading_residual = heading_residuals[list(range(batch_size)),heading_class[list(range(batch_size))]] # B,
        size_class = torch.argmax(size_logits, 1) # B, 
        size_residual = size_residuals[list(range(batch_size)),size_class[list(range(batch_size))],:] #B, 3

        heading_angle = class2angle(heading_class, heading_residual, self.config['num_heading_bin']) # B, 
        box_size = self.net.class2size(size_class, size_residual) # B, 3
        corners_3d = get_box3d_corners_helper(center_pred, heading_angle, box_size).squeeze() #(B, 8, 3) 
        bird_view_box1 = torch.cat([center_pred[:, [0, 2]], box_size[:, [0, 1]], heading_angle.unsqueeze(1)], dim=1) # B, 5
        size_box1 = box_size[:, 0] * box_size[:, 1] # B,
        vol_box1 =  box_size[:, 0] * box_size[:, 1] * box_size[:, 2]

        heading_angle_label = class2angle(heading_class_label, heading_residual_label, self.config['num_heading_bin'])
        box_size_label = self.net.class2size(size_class_label, size_residual_label)
        corners_3d_label = get_box3d_corners_helper(center_label, heading_angle_label, box_size_label).squeeze() #(B, 8, 3)
        bird_view_box2 = torch.cat([center_label[:, [0, 2]], box_size_label[:, [0, 1]], heading_angle_label.unsqueeze(1)], dim=1) # B, 5
        size_box2 = box_size_label[:, 0] * box_size_label[:, 1] # B, 
        vol_box2 =  box_size_label[:, 0] * box_size_label[:, 1] * box_size_label[:, 2]

        iou2d_pair = rotated_iou(bird_view_box1, bird_view_box2)
        iou2d = iou2d_pair[list(range(batch_size)), list(range(batch_size))] # B, 
        intersection = ( size_box1 + size_box2 ) * iou2d

        box1_height_up = corners_3d[:, [0], [1]].squeeze() # (B,)
        box1_height_down = corners_3d[:, [4], [1]].squeeze() # (B, )
        box2_height_up = corners_3d_label[:, [0], [1]].squeeze() # (B, )
        box2_height_down = corners_3d_label[:, [4], [1]].squeeze() # (B, )

        up = torch.where(box1_height_up < box1_height_down, box1_height_up, box2_height_up)
        down = torch.where(box1_height_down > box2_height_down, box1_height_down, box2_height_down)

        intersect = up - down
        intersect.clamp_(min=0.0)

        iou3d = torch.mul(intersect, intersection) / (vol_box1 + vol_box2)
        return iou2d.mean(), iou3d.mean()

    def training_step(self, batch, batch_idx):
        point_set, seg, box3d_center, angle_class, angle_residual, \
        size_class, size_residual, rot_angle, one_hot_vec = batch

        box3d_center = box3d_center.float()
        angle_residual = angle_residual.float()
        size_residual = size_residual.float()

        endpoint = self.net(point_set.contiguous(), one_hot_vec.contiguous())
        loss_dict, loss = self.loss(seg, box3d_center, angle_class, angle_residual,
                                    size_class, size_residual, endpoint)
        lr_rate = self.lr_scheduler.optimizer.param_groups[0]['lr']
        loss_dict['lr'] = lr_rate
        return {'loss': loss, 'log': loss_dict}
    
    def validation_step(self, batch, batchidx):
        point_set, seg, box3d_center, angle_class, angle_residual, \
        size_class, size_residual, rot_angle, one_hot_vec = batch

        box3d_center = box3d_center.float()
        angle_residual = angle_residual.float()
        size_residual = size_residual.float()

        endpoint = self.net(point_set.contiguous(), one_hot_vec.contiguous())
        iou2d, iou3d = self.compute_box3d_iou(endpoint['center'], endpoint['heading_score'],
                                              endpoint['heading_residuals'], endpoint['size_scores'],
                                              endpoint['size_residuals'], box3d_center, 
                                              angle_class, angle_residual, size_class, size_residual)
        return {'iou3d': iou3d, 'iou2d': iou2d}

    def validation_epoch_end(self, outputs):
        avg_iou3d = torch.stack([x['iou3d'] for x in outputs]).mean()
        avg_iou2d = torch.stack([x['iou2d'] for x in outputs]).mean()
        tensorboard_logs = {'iou3d': avg_iou3d, 'iou2d': avg_iou2d}
        return {'val_loss': avg_iou3d, 'log': tensorboard_logs}


def point_cloud_masking(point_cloud, logits, max_points, xyz_only=True):
    """
    Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Input:
        point_cloud: shape (B, C, N)
        logits: shape (B, 2, N)
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: shape (B,3)
    """
    batch, num_point = point_cloud.shape[:2]
    mask = (logits[:, 0, :] < logits[:, 1, :]).unsqueeze(-1).float() # (B, N, 1)
    mask_count = torch.sum(mask, dim=1, keepdim=True) # (B, 1, 1)

    point_cloud_xyz = point_cloud[..., [0, 1, 2]] # (B, N, 3)
    mask_xyz_mean = torch.mul(mask.repeat((1,1,3)), point_cloud_xyz)
    mask_xyz_mean = torch.sum(mask_xyz_mean, dim=1, keepdim=True) # (B, 1, 3)
    mask_count_tile = mask_count.repeat(1,1,3)
    mask_xyz_mean = mask_xyz_mean / torch.max(mask_count_tile, torch.ones_like(mask_count_tile)) # (B, 1, 3)

    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean # (B, N, 3)
    object_point = []
    for i in range(batch):
        mask_id = mask[i, ...].reshape(-1) # (N, )
        loc = torch.where(mask_id>0)
        pre_select = point_cloud_xyz_stage1[i, ...][loc]  # (mask_num, 3)
        mask_num = pre_select.shape[0]
        if mask_num > 0:
            if mask_num > max_points:
                select_id = torch.randint(high=mask_num, size=(max_points, ))
                object_point_cloud = pre_select[select_id]
            elif mask_num < max_points:
                select_id = torch.randint(high=mask_num, size=(max_points - mask_num, ))
                object_point_cloud = torch.cat([pre_select, pre_select[select_id]], dim=0)
            else:
                object_point_cloud = pre_select
        else:
            object_point_cloud = torch.ones((max_points, 3), device=pre_select.device) * i
        object_point.append(object_point_cloud)
    return torch.stack(object_point, dim=0), mask_xyz_mean.squeeze(dim=1)


if __name__=='__main__':
    import yaml
    with open('../configs/frustum.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    Pointnet2 = FrustumPointnet(config['MODEL']).cuda()
    one_hot = torch.randn(2, 4).cuda()
    points = torch.randn((2, 200, 4)).cuda()
    out = Pointnet2(points, one_hot)
    pass




