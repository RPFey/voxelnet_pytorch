import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.optim.lr_scheduler as lr_sched
from abc import ABCMeta

def init_weight(m):
    pass

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


class PointNet2Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build_network()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features


class ModelBase(pl.LightningModule):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        self.config = config

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        decay_epochs = [ round(x * self.total_epoch) for x in self.config['OPTIMIZATION']['decay_epoch_list']]

        def lr_lbmd(cur_epoch):
            decay_rate = 1
            for decay_epoch in decay_epochs:
                if cur_epoch > decay_epoch:
                    decay_rate *= self.config['OPTIMIZATION']['lr_decay']
            return max(decay_rate, self.config['OPTIMIZATION']['lr_clip'] / self.config['OPTIMIZATION']['lr'])

        def bn_lbmd(cur_epoch):
            decay_rate = self.config['OPTIMIZATION']['bn_momentum']
            for decay_epoch in decay_epochs:
                if cur_epoch > decay_epoch:
                    decay_rate *= self.config['OPTIMIZATION']['bnm_decay']
            return max(decay_rate, self.config['OPTIMIZATION']['bnm_clip'])

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['OPTIMIZATION']["lr"],
            weight_decay=self.config['OPTIMIZATION']["weight_decay"],
        )
        self.lr_scheduler = lr_sched.LambdaLR(self.optimizer, lr_lbmd)
        self.bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return self.optimizer
