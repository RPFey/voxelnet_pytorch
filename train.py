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

torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--cfg_file', default=None, type=str)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    train_dataset = FrustumDataset(npoints=512, split='train', root_dir='./',
                            NUM_HEADING_BIN=cfg['MODEL']['num_heading_bin'], 
                             rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
    train_dataloader = DataLoader(train_dataset, num_workers = args.workers, batch_size=args.batch_size, shuffle=True)

    val_dataset = FrustumDataset(npoints=512, split='val', root_dir='./',
                                NUM_HEADING_BIN=cfg['MODEL']['num_heading_bin'],
                                rotate_to_center=True, one_hot=True)
    val_dataloader = DataLoader(val_dataset, num_workers = args.workers, batch_size=args.batch_size, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        filepath='./output',
        save_top_k=4,
        verbose=True,
        monitor='val_loss',
        mode='max',
        save_weights_only=True,
        prefix='',
    )

    model = FrustumLight(cfg['MODEL'], args)
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback = checkpoint_callback)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)