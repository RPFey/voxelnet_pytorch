import numpy as np
import torch
import yaml
from easydict import EasyDict
from pathlib import Path

def get_box3d_corners_helper(centers, headings, sizes):
    """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = centers.shape[0]
    l = sizes[:,0].view(N,1)
    w = sizes[:,1].view(N,1)
    h = sizes[:,2].view(N,1)
    #print l,w,h
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
    corners = torch.cat([x_corners.view(N,1,8), y_corners.view(N,1,8),
                            z_corners.view(N,1,8)], dim=1) # (N,3,8)

    ###ipdb.set_trace()
    #print x_corners, y_corners, z_corners
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = torch.ones([N], dtype=torch.float32, device=headings.device)
    zeros = torch.zeros([N], dtype=torch.float32, device=headings.device)
    row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.view(N,1,3), row2.view(N,1,3),row3.view(N,1,3)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = torch.bmm(R, corners) # (N,3,8)
    corners_3d += centers.view(N,3,1).repeat(1,1,8) # (N,3,8)
    corners_3d = torch.transpose(corners_3d,1,2) # (N,8,3)
    return corners_3d

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    '''
    Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
        pred_cls : tensor B, 
        residuals : tensor B,
    return 
        angle B,
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_per_class = torch.ones((pred_cls.shape[0], ), device=pred_cls.device) * angle_per_class
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format :
        loc = torch.where(angle > np.pi)
        angle[loc] -= 2 * np.pi
    return angle

# class PointcloudScale(object):
#     def __init__(self, lo=0.8, hi=1.25):
#         self.lo, self.hi = lo, hi

#     def __call__(self, points):
#         scaler = np.random.uniform(self.lo, self.hi)
#         points[:, 0:3] *= scaler
#         return points


# class PointcloudRotate(object):
#     def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
#         self.axis = axis

#     def __call__(self, points):
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         rotation_matrix = angle_axis(rotation_angle, self.axis)

#         normals = points.size(1) > 3
#         if not normals:
#             return torch.matmul(points, rotation_matrix.t())
#         else:
#             pc_xyz = points[:, 0:3]
#             pc_normals = points[:, 3:]
#             points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
#             points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

#             return points


# class PointcloudRotatePerturbation(object):
#     def __init__(self, angle_sigma=0.06, angle_clip=0.18):
#         self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

#     def _get_angles(self):
#         angles = np.clip(
#             self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
#         )

#         return angles

#     def __call__(self, points):
#         angles = self._get_angles()
#         Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
#         Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
#         Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

#         rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

#         normals = points.size(1) > 3
#         if not normals:
#             return torch.matmul(points, rotation_matrix.t())
#         else:
#             pc_xyz = points[:, 0:3]
#             pc_normals = points[:, 3:]
#             points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
#             points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

#             return points


# class PointcloudJitter(object):
#     def __init__(self, std=0.01, clip=0.05):
#         self.std, self.clip = std, clip

#     def __call__(self, points):
#         jittered_data = (
#             points.new(points.size(0), 3)
#             .normal_(mean=0.0, std=self.std)
#             .clamp_(-self.clip, self.clip)
#         )
#         points[:, 0:3] += jittered_data
#         return points


# class PointcloudToTensor(object):
#     def __call__(self, points):
#         return torch.from_numpy(points).float()


# class PointcloudTranslate(object):
#     def __init__(self, translate_range=0.1):
#         self.translate_range = translate_range

#     def __call__(self, points):
#         translation = np.random.uniform(-self.translate_range, self.translate_range)
#         points[:, 0:3] += translation
#         return points


# class PointcloudRandomInputDropout(object):
#     def __init__(self, max_dropout_ratio=0.875):
#         assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
#         self.max_dropout_ratio = max_dropout_ratio

#     def __call__(self, points):
#         pc = points.numpy()

#         dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
#         drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
#         if len(drop_idx) > 0:
#             pc[drop_idx] = pc[0]  # set to the first point

#         return torch.from_numpy(pc).float()
