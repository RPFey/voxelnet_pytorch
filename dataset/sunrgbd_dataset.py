''' Provider class for RoI binary segmentation task '''
import pickle
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from dataset.utils import roty, load_zipped_pickle
from configs.sunrgbd import type2class, class2type, type2onehotclass, type_mean_size, NUM_HEADING_BIN, NUM_CLASS


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
       
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual

class ROISegBoxDataset(object):
    def __init__(self, npoints, split, random_flip=False, random_shift=False, rotate_to_center=False, overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(BASE_DIR, '%s_1002.zip.pickle'%(split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:
            self.id_list, self.box2d_list, self.input_list, self.type_list, self.frustum_angle_list, self.prob_list = load_zipped_pickle(overwritten_data_path)
        else:
            self.id_list,self.box2d_list,self.box3d_list,self.input_list,self.label_list,self.type_list,self.heading_list,self.size_list,self.frustum_angle_list=load_zipped_pickle(overwritten_data_path)

    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        try:
            # ------------------------------ INPUTS ----------------------------
            rot_angle = self.get_center_view_rot_angle(index)

            # compute one hot vector
            if self.one_hot:
                cls_type = self.type_list[index]
                assert(cls_type in ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'])
                one_hot_vec = np.zeros((NUM_CLASS))
                one_hot_vec[type2onehotclass[cls_type]] = 1

            # Get point cloud
            if self.rotate_to_center:
                point_set = self.get_center_view_point_set(index)
            else:
                point_set = self.input_list[index]
            # Resample
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            point_set = point_set[choice, :]

            if self.from_rgb_detection:
                if self.one_hot:
                    return point_set, rot_angle, self.prob_list[index], one_hot_vec
                else:
                    return point_set, rot_angle, self.prob_list[index]
            
            # ------------------------------ LABELS ----------------------------
            seg = self.label_list[index] 
            seg = seg[choice]

            # Get center point of 3D box
            if self.rotate_to_center:
                box3d_center = self.get_center_view_box3d_center(index)
            else:
                box3d_center = self.get_box3d_center(index)

            # Heading
            if self.rotate_to_center:
                heading_angle = self.heading_list[index] - rot_angle
            else:
                heading_angle = self.heading_list[index]

            # Size
            size_class, size_residual = size2class(self.size_list[index], self.type_list[index])

            # Data Augmentation
            if self.random_flip:
                if np.random.random()>0.5:
                    point_set[:,0] *= -1
                    box3d_center[0] *= -1
                    heading_angle = np.pi - heading_angle
                    # NOTE: rot_angle won't be correct if we have random_flip...
            if self.random_shift:
                dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
                shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
                point_set[:,2] += shift
                box3d_center[2] += shift
                height_shift = np.random.random()*0.4-0.2 # randomly shift +-0.2 meters
                point_set[:,1] += height_shift
                box3d_center[1] += height_shift

            angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

            if self.one_hot:
                return point_set.astype(np.float32), seg.astype(np.int64), box3d_center.astype(np.float32), angle_class, angle_residual, size_class, size_residual, rot_angle, one_hot_vec.astype(np.float32)
            else:
                return point_set.astype(np.float32), seg.astype(np.int64), box3d_center.astype(np.float32), angle_class, angle_residual, size_class, size_residual, rot_angle
        except:
            print(" index : %d wrong"%(index))

    def get_center_view_rot_angle(self, index):
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        box3d_center = (self.box3d_list[index][0,:] + self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        box3d_center = (self.box3d_list[index][0,:] + self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Input ps is NxC points with first 3 channels as XYZ
            z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))