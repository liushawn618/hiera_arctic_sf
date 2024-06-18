from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

"""
Source: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
"""


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, input_dim, shallow_dim, mid_dim, out_dim, global_feat=False):
        super(PointNetfeat, self).__init__()
        #pointnet.shallow 这个浅层用于初步提取点云数据的特征。
        self.shallow_layer = nn.Sequential(
            nn.Conv1d(input_dim, shallow_dim, 1), nn.BatchNorm1d(shallow_dim)
        )
        #pointnet.deep 用于进一步处理浅层特征，以生成更复杂的特征表示。
        self.base_layer = nn.Sequential(
            nn.Conv1d(shallow_dim, mid_dim, 1),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Conv1d(mid_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
        )

        self.global_feat = global_feat
        self.out_dim = out_dim
     
     #输入张量x，通常是一个形状为(batch_size, input_dim, n_pts)的张量，其中batch_size是批次大小，input_dim是输入数据的维度，n_pts是每个样本中的点的数量。
    def forward(self, x):
        n_pts = x.size()[2]
        x = self.shallow_layer(x)
        pointfeat = x

        x = self.base_layer(x)
        #使用最大池化操作，沿第三维（点的数量）取最大值，并保持维度不变。
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)

        trans_feat = None
        trans = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
            #使得每个点都有全局特征和局部特征
            return torch.cat([x, pointfeat], 1), trans, trans_feat
