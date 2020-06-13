from utils import get_filtered_lidar, project_velo2rgb, draw_rgb_projections
from config import config as cfg
from data.kitti import KittiDataset
import torch.utils.data as data
# from nms.pth_nms import pth_nms
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn
import cv2
import matplotlib.pyplot as plt
from detectron2.layers import nms_rotated
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True

def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    N = deltas.shape[0]
    deltas = deltas.view(N, -1, 7)
    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.to(cfg.device)
        boxes3d = boxes3d.to(cfg.device)

    anchors_reshaped = anchors.view(-1, 7)

    anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1,2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[...,[3]]) + anchors_reshaped[..., [2]]

    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d

def detection_collate(batch):
    lidars = []
    images = []
    calibs = []

    targets = []
    pos_equal_ones=[]
    ids = []
    for i, sample in enumerate(batch):
        lidars.append(sample[0])
        images.append(sample[1])
        calibs.append(sample[2])
        targets.append(sample[3])
        pos_equal_ones.append(sample[4])
        ids.append(sample[5])
    return lidars,images,calibs,\
           torch.cuda.FloatTensor(np.array(targets)), \
           torch.cuda.FloatTensor(np.array(pos_equal_ones)),\
           ids


def box3d_center_to_corner_batch(boxes_center):
    # (N, 7) -> (N, 8)
    N = boxes_center.shape[0]

    x, y, z, h, w, l,theta = boxes_center.chunk(7, dim=1)
    sizes = (w * l).squeeze()
    zero_pad = torch.zeros((boxes_center.shape[0], 1)).to(cfg.device)
    one_pad = torch.ones((boxes_center.shape[0], 1)).to(cfg.device)
    box_shape = torch.cat([
        -0.5*l , -0.5*l,  0.5*l, 0.5*l, -0.5*l, -0.5*l,  0.5*l, 0.5*l,\
        0.5*w  , -0.5*w, -0.5*w, 0.5*w,  0.5*w, -0.5*w, -0.5*w, 0.5*w,\
        zero_pad, zero_pad, zero_pad, zero_pad, h, h, h, h \
    ], dim = 1).unsqueeze(2).reshape((-1, 3, 8))
    rotMat = torch.cat([
        torch.cos(theta), -torch.sin(theta), zero_pad,\
        torch.sin(theta), torch.cos(theta), zero_pad, \
        zero_pad, zero_pad, one_pad \
    ], dim=1).unsqueeze(2).reshape((-1, 3, 3))
    trans = torch.cat((x,y,z), dim=1).unsqueeze(2) # N * 3 * 1
    corner = torch.bmm(rotMat, box_shape) + trans

    return corner, sizes

def box3d_corner_to_top_batch(boxes3d):
    # [N,8,3] -> [N,4,2] -> [N,8]
    box3d_top=boxes3d[:, :2, :4]
    return box3d_top.reshape((-1, 8))

def nms(boxes_bottom, nms_threshold):
    # boxes_bottom (N, 9)
    x = torch.linspace(0,1,cfg.num_dim)
    y = torch.linspace(0,1,cfg.num_dim)
    X, Y = torch.meshgrid(x,y)
    coords = torch.cat([X.reshape(1, -1), Y.reshape(1, -1)], dim=0).to(cfg.device)
    query = []

    # filter nonsigular
    x1 = boxes_bottom[:, 0] - boxes_bottom[:, 1]
    y1 = boxes_bottom[:, 4] - boxes_bottom[:, 5]
    x2 = boxes_bottom[:, 2] - boxes_bottom[:, 1]
    y2 = boxes_bottom[:, 6] - boxes_bottom[:, 5]

    sizes = torch.sqrt(x1**2 + y1**2) * torch.sqrt(x2**2 + y2**2) # sizes of rectangle

    deter = torch.abs(x1*y2 - x2*y1)
    non_singular = torch.nonzero(deter > 1e-1)
    boxes_bottom = boxes_bottom[non_singular].reshape(-1, 10)
    sizes = sizes[non_singular].reshape(-1)
    if boxes_bottom.shape[0] == 0:
        return None

    alpha_1 = torch.stack([boxes_bottom[:, 0] - boxes_bottom[:, 1], boxes_bottom[:, 4] - boxes_bottom[:, 5]], dim = 1) # (x1, y1)
    alpha_2 = torch.stack([boxes_bottom[:, 2] - boxes_bottom[:, 1], boxes_bottom[:, 6] - boxes_bottom[:, 5]], dim = 1) # (x2, y2)
    trans = torch.stack([boxes_bottom[:, 1], boxes_bottom[:, 5]], dim = 1)

    while trans.shape[0] > 1:
        Rot1 = torch.stack([alpha_1[0, :].t(), alpha_2[0, :].t()], dim=1)
        train_trans = torch.mm(Rot1, coords).unsqueeze(0) + (trans[0, :] - trans[1:, :]).unsqueeze(2).repeat(1, 1, cfg.num_dim**2) # N * 2 * 2601
        Rot2 = torch.cat([alpha_1[1:, :].unsqueeze(2), alpha_2[1:, :].unsqueeze(2)], dim=2)
        translated = torch.bmm(torch.inverse(Rot2), train_trans) # N * 2 * 2601

        x_fit = (0<=translated[:, 0, :])*(translated[:, 0, :]<=1)
        y_fit = (0<=translated[:, 1, :])*(translated[:, 1, :]<=1)
        Intersection = (x_fit * y_fit).float()

        Intersection = torch.sum(Intersection, dim = 1) / cfg.num_dim**2
        IoU = sizes[0] * Intersection / (sizes[0] + sizes[1:] - sizes[0]*Intersection)
        index = torch.nonzero(IoU < nms_threshold) + 1

        trans = trans[index].reshape(-1, 2)
        alpha_1 = alpha_1[index].reshape(-1, 2)
        alpha_2 = alpha_2[index].reshape(-1, 2)
        query.append(boxes_bottom[0])
        boxes_bottom = boxes_bottom[index].reshape(-1, 10)
        sizes = sizes[index].reshape(-1)

    if boxes_bottom.shape[0] == 1:
        query.append(boxes_bottom[0])
    return torch.stack(query, dim=0)

def draw_boxes(reg, prob, images, calibs, ids, tag):
    prob = prob.reshape(cfg.N, -1)
    batch_boxes3d = delta_to_boxes3d(reg, cfg.anchors)
    mask = torch.gt(prob, cfg.score_threshold)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    out_images = []
    for batch_id in range(cfg.N):
        boxes3d = torch.masked_select(batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
        scores = torch.masked_select(prob[batch_id], mask[batch_id])

        image = images[batch_id]
        calib = calibs[batch_id]
        id = ids[batch_id]

        if len(boxes3d) != 0:
            # boxes3d_corner, sizes = box3d_center_to_corner_batch(boxes3d)
            # boxes2d_bottom = box3d_corner_to_top_batch(boxes3d_corner)
            # boxes2d_score = torch.cat((boxes2d_bottom, scores.unsqueeze(1), torch.arange(0, len(boxes2d_bottom)).float().unsqueeze(1).to(cfg.device)), dim=1)

            # args = torch.argsort(boxes2d_score[:, 8], descending=True)
            # boxes2d_score = boxes2d_score[args]

            # vac  = torch.nonzero(sizes > 1e-2)
            # boxes2d_score = boxes2d_score[vac].squeeze()
            # if boxes2d_score.shape[0] == 0:
            #     out_images.append(image)
            #     continue

            # NMS
            # boxes2d_score = nms(boxes2d_score, cfg.nms_threshold)

            index = nms_rotated(boxes3d[..., [0, 1, 5, 4, 6]], scores, 0.01)
            if len(index) is None:
                out_images.append(image)
                continue
            # boxes3d_corner_keep = boxes3d_corner[boxes2d_score[:, 9].long()]
            boxes3d = boxes3d[index]
            print("No. %d objects detected" % len(boxes3d))
            boxes3d_corner_keep, _ = box3d_center_to_corner_batch(boxes3d)
            boxes3d_corner_keep = boxes3d_corner_keep.cpu().numpy()
            boxes3d_corner_keep = np.transpose(boxes3d_corner_keep, (0, 2, 1))
            rgb_2D = project_velo2rgb(boxes3d_corner_keep, calib)
            img_with_box = draw_rgb_projections(image, rgb_2D, color=(0, 0, 255), thickness=1)
            out_images.append(img_with_box)
    return np.array(out_images)

if __name__=='__main__':
    center = torch.tensor([[1, 1, 1, 2, 3, 4, 0.3]]).to(cfg.device)
    box3d_center_to_corner_batch(center)













