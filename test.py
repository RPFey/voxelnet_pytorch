from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from utils.utils import cfg_from_yaml_file
from configs.config import cfg
from dataset.FrustumData import FrustumDataset
from torch.utils.data import DataLoader
from model.FrustumPointnet import FrustumLight
from pytorch_lightning.callbacks import ModelCheckpoint
import os

torch.backends.cudnn.benchmark = True

def write_detection_results()

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--cfg_file', default=None, type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    val_dataset = FrustumDataset(npoints=512, split='val', root_dir='./',
                                NUM_HEADING_BIN=cfg['MODEL']['num_heading_bin'],
                                rotate_to_center=True, one_hot=True, from_rgb_detection=True)
    val_dataloader = DataLoader(val_dataset, num_workers = args.workers, batch_size=args.batch_size)

    model = FrustumLight(cfg['MODEL'], args)
    model.load_from_checkpoint(args.ckpt)
    model.cuda()
    model.eval()

    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []

    for batch in val_dataloader:
        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec, batch_id = batch

        batch_data = torch.from_numpy(batch_data).cuda()
        batch_one_hot_vec = torch.from_numpy(batch_one_hot_vec).cuda()

        with torch.no_grad():
            centers, heading_cls, heading_res, size_cls, size_res = model.predict(batch_data, batch_one_hot_vec)

        center_list.append(centers.cpu().numpy())
        heading_cls_list.append(heading_cls.cpu().numpy())
        heading_res_list.append(heading_res.cpu().numpy())
        size_cls_list.append(size_cls.cpu().numpy())
        size_res_list.append(size_res.cpu().numpy())
        rot_angle_list.append(batch_rot_angle)
    
    write_detection_results(result_dir, val_dataset.id_list,
        val_dataset.type_list, val_dataset.box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)




