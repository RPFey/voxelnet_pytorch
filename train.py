from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from utils.utils import cfg_from_yaml_file
from dataset.FrustumData import FrustumDataset
from dataset.sunrgbd_dataset import ROISegBoxDataset
from torch.utils.data import DataLoader
from model.FrustumPointnet import FrustumLight
from pytorch_lightning.callbacks import ModelCheckpoint
import os

torch.backends.cudnn.benchmark = True

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--dataset_name', default='kitti', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--cfg_file', default=None, type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    import importlib
    config = importlib.import_module('configs.'+args.dataset_name)
    cfg = config.cfg

    cfg_from_yaml_file(args.cfg_file, cfg)

    train_dataset = ROISegBoxDataset(npoints=cfg['MODEL']['object_points'], split='train', 
                             overwritten_data_path='./frustum_data/train_1002_aug5x.zip.pickle', 
                             rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
    train_dataloader = DataLoader(train_dataset, num_workers = args.workers, batch_size=args.batch_size)

    val_dataset = ROISegBoxDataset(npoints=cfg['MODEL']['object_points'], split='val',
                                 overwritten_data_path='./frustum_data/val_1002.zip.pickle',
                                rotate_to_center=True, one_hot=True)
    val_dataloader = DataLoader(val_dataset, num_workers = args.workers, batch_size=args.batch_size)

    checkpoint_callback = ModelCheckpoint(
        filepath='./sunrgbd',
        save_top_k=4,
        verbose=True,
        monitor='iou3d',
        mode='max',
        save_weights_only=False,
        prefix='',
    )

    model = FrustumLight(cfg['MODEL'], args, config.type2class ,config.type_mean_size)
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback = checkpoint_callback)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)