import torch.nn as nn
import torch
from torch.autograd import Variable
from config import config as cfg
from data.kitti import KittiDataset
import torch.utils.data as data
import time
from loss import VoxelLoss
from voxelnet import VoxelNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
# from nms.pth_nms import pth_nms
import numpy as np
import torch.backends.cudnn
from test_utils import draw_boxes
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from utils import plot_grad
import cv2

import argparse

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--ckpt', type=str, default=None, help='pre_load_ckpt')
parser.add_argument('--index', type=int, default=0, help='hyper_tag')
parser.add_argument('--epoch', type=int , default=160, help="training epoch")
args = parser.parse_args()

def weights_init(m):
    '''
        权重初始化
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()

def detection_collate(batch):
    '''
        批次加载的数据整理（自定义整理函数，而非使用默认的整理函数）
    '''
    voxel_features = [] # 体素特征
    voxel_coords = [] # 体素坐标
    gt_box3d_corner = [] # 地面真值3D框的8个角度坐标
    gt_box3d = [] # 地面真值3D框
    images = [] 
    calibs = []
    ids = []
    for i, sample in enumerate(batch):
        '''
            batch为批次
            i是该批次中的样本序号
            sample是该批次的第i个样本
        '''
        voxel_features.append(sample[0])

        voxel_coords.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))

        gt_box3d_corner.append(sample[2])
        gt_box3d.append(sample[3])

        images.append(sample[4])
        calibs.append(sample[5])
        ids.append(sample[6])
        print('数据ID：',sample[6])
        a=np.concatenate(voxel_features)
        b=np.concatenate(voxel_coords)
        c=gt_box3d_corner
        d=gt_box3d
        e=images
        f=calibs
        g=ids
    return np.concatenate(voxel_features), \
           np.concatenate(voxel_coords), \
           gt_box3d_corner,\
           gt_box3d,\
           images,\
           calibs, ids

hyper = {'alpha': 1.0, # 正样本损失项常数
          'beta': 10.0, # 负样本损失项常数
          'pos': 0.75, # 正样本阈值
          'neg': 0.5,  # 负样本阈值
          'lr':0.005,  # 学习率
          'momentum': 0.9, # 动量（Momentum）是一种优化算法，用于加速神经网络的训练
          'lambda': 2.0,
          'gamma':2,
          'weight_decay':0.00001 # 权值衰减率
          }

torch.backends.cudnn.enabled=True

def train(net, model_name, hyper, cfg, writer, optimizer,train_set='train',train_type='velodyne_train'):
    '''
        net 网络模型
        model_name 训练后保存的模型名称
        hyper 辅助参数设置
        cfg 训练配置文件
        writer 文件输出器
        optimizer 优化函数
        train_set 训练设置，可选只为train、val和test
        train_type 训练类型：训练或者验证集选择velodyne_train，测试集选择velodyne_test
    '''
    dataset = KittiDataset(cfg=cfg,root = r'E:\zqw\PaperCode\data\ObjectDetection\kitti_original',set=train_set,type = train_type)
    voxel_features = dataset[3169][0]
    a=np.concatenate(voxel_features)
    print(a)
    # len_= len(dataset)
    # print('总长度：',len_)
    # for i in range(len_):
    #     print('外围第',i,'个')
    #     id_ = dataset[i][6]
    #     if id_ == '006310':
    #         print('内部第',i,'个')
    #         print(id_)
    #         voxel_features = dataset[i][0]
    #         a=np.concatenate(voxel_features)
    #         print(a)
    #         break
  
    


if __name__ == '__main__':
    pre_model = args.ckpt # 预训练参数
    cfg.pos_threshold = hyper['pos']
    cfg.neg_threshold = hyper['neg']
    model_name = "model_%d"%(args.index+1)

    writer = SummaryWriter('runs/%s'%(model_name[:-4]))

    net = VoxelNet()
    net.to(cfg.device)
    optimizer = optim.SGD(net.parameters(), lr=hyper['lr'], momentum = hyper['momentum'], weight_decay=hyper['weight_decay'])

    if pre_model is not None and os.path.exists(os.path.join('./model',pre_model)) :
        ckpt = torch.load(os.path.join('./model',pre_model), map_location=cfg.device)
        net.load_state_dict(ckpt['model_state_dict'])
        cfg.last_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else :
        net.apply(weights_init)     
    train(net, model_name, hyper, cfg, writer, optimizer)
    writer.close()
