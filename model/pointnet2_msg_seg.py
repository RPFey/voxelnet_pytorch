import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from model.modelbase import PointNet2Base
import torch.nn.functional as F


class PointNet2SemSegMSG(PointNet2Base):

    def build_network(self):
        self.SA_modules = nn.ModuleList()
        c_in = self.config['features']
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[[c_in, 32, 32, 64],
                      [c_in, 64, 64, 128],
                      [c_in, 64, 96, 128]],
                use_xyz=self.config["use_xyz"],
            )
        )

        input_channels_1 = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.4, 0.8, 1.6],
                nsamples=[64, 64, 128],
                mlps=[
                    [input_channels_1, 64, 64, 128],
                    [input_channels_1, 128, 128, 256],
                    [input_channels_1, 128, 128, 256],
                ],
                use_xyz=self.config["use_xyz"],
            )
        )

        input_channels_2 = 128 + 256 + 256
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[input_channels_2, 128, 256, 1024],
                use_xyz=self.config["use_xyz"],
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[c_in + 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[input_channels_1 + 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[input_channels_2 + 1024 + self.config['num_cls'], 128, 128]))

        self.features = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.fc_lyaer = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )

    def forward(self, pointcloud : torch.Tensor, one_hot : torch.Tensor):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            one_hot: shape (B, num_cls)

            return
            ---------
            feats : (B, 128, N)
            prob : (B, 2, N) between (0, 1)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        one_hot = one_hot.unsqueeze(2)

        l3_points = torch.cat([l_features[3], one_hot], dim=1)
        l2_points = self.FP_modules[2](l_xyz[2], l_xyz[3], l_features[2], l3_points)
        l1_points = self.FP_modules[1](l_xyz[1], l_xyz[2], l_features[1], l2_points)
        l0_points = self.FP_modules[0](l_xyz[0], l_xyz[1], l_features[0], l1_points)

        feats = self.features(l0_points)
        prob = self.fc_lyaer(feats)

        return feats, F.softmax(prob, dim=1)


class CenterRegressionNet(nn.Module):
    def __init__(self, config):
        super(CenterRegressionNet, self).__init__()
        self.config = config

    def build_netork(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + self.config['num_cls'], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        )


    def forward(self, obj_point_cloud: torch.Tensor, one_hot: torch.Tensor):
        """
        Parameter:
            obj_point_cloud: (B, N, 3) point cloud of objects
            one_: (B, num_cls) one_hot vector fclass
        """
        num_point = obj_point_cloud.shape[1]
        net = obj_point_cloud.permute(0, 2, 1).unsqueeze(3) # (B, 3, N, 1)
        net = self.cnn(net) # (B, 256, N, 1)
        net = torch.nn.functional.max_pool2d(net, kernel_size=(num_point, 1), stride=(1, 1))
        net = net.squeeze()
        net = torch.cat([net, one_hot], dim=1)
        center = self.fc(net)
        return center


class BoxEstimationNet(nn.Module):
    def __init__(self, config):
        super(BoxEstimationNet, self).__init__()
        self.config = config

    def build_network(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.2,
                nsample=64,
                mlp=[c_in, 64, 64, 128],
                use_xyz=self.config["use_xyz"],
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                mlp=[128, 128, 128, 256],
                radius=0.4,
                nsample=64,
                use_xyz=self.config["use_xyz"],
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 512],
                use_xyz=self.config["use_xyz"],
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + self.config['num_cls'], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 3 + 2*self.config['num_heading_bin'] + 4*self.config['num_size_cluster'])
        )

    def forward(self, obj_point_cloud: torch.Tensor, one_hot: torch.Tensor):
        """
        Parameter:
            obj_point_cloud: (B, N, 3) point cloud of objects
            one_: (B, num_cls) one_hot vector fclass
        """
        batch, num_point = obj_point_cloud.shape[:2]
        l_xyz, l_features = obj_point_cloud[..., [0, 1, 2]], None
        for i in range(len(self.SA_modules)):
            l_xyz, l_features = self.SA_modules[i](l_xyz, l_features)

        net = l_features.reshape(batch, -1)
        net = torch.cat([net, one_hot], dim=1)
        net = self.fc(net)

        return net


