'''
    模型训练与验证
'''
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

# 使用非确定性算法 在非确定性算法的情况下，如果每次 iteration的数据输入大小变化不大的情况下，
# 会自动寻找当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.enabled=True

'''
    相关参数
    ckpt：预训练模型
    epoch：训练轮次

'''
parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--data_root', type=str, default=r'C:\zqw\PaperCode\data\ObjectDetection\kitti_original', help='数据目录')
parser.add_argument('--ckpt', type=str, default=None, help='pre_load_ckpt')
parser.add_argument('--epoch', type=int , default=400, help="training epoch")
parser.add_argument('--lr', type=float , default=0.005, help="Learning rate，学习率，在优化器中使用")
parser.add_argument('--momentum', type=float , default=0.9, help="动量（Momentum）是一种优化算法，用于加速神经网络的训练，在优化器中使用")
parser.add_argument('--weight_decay', type=float , default=0.00001, help="权值衰减率，在优化器中使用")

args = parser.parse_args()
def weights_init(m):
    '''
        权重初始化 #调用torch.nn.init里面的初始化方法对权重和偏置进行初始化
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
            #这里的batch是自定义dataset里面__getitem__方法的返回值，batch里面有batch——size个样本
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
    return np.concatenate(voxel_features), \
           np.concatenate(voxel_coords), \
           gt_box3d_corner,\
           gt_box3d,\
           images,\
           calibs, ids

hyper = {'alpha': 1.5, # 正样本损失项常数（分类损失1）
          'beta': 1.0, # 负样本损失项常数（分类损失2）
          'lambda': 2.0, # 分类损失项常数（总分类损失）
          'gamma':3,   # 回归损失常数项
          }
# 训练数据集
training_dataset = KittiDataset(cfg=cfg,root = args.data_root,set='train',type = 'velodyne_train')
training_data_loader = data.DataLoader(training_dataset, batch_size=cfg.N, num_workers=4, collate_fn = detection_collate, shuffle=True, \
                              pin_memory=False)
# 验证数据集
val_dataset=KittiDataset(cfg=cfg,root= args.data_root,set='val',type = 'velodyne_train')
val_data_loader = data.DataLoader(val_dataset, batch_size=cfg.N, num_workers=4, collate_fn=detection_collate, shuffle=True, \
                              pin_memory=False)

def train(dataset,data_loader,net, hyper, cfg, optimizer,criterion, epoch):
    '''
        training_data_loader：训练数据集
        net 网络模型
        model_name 训练后保存的模型名称前缀
        hyper 辅助参数设置
        cfg 训练配置文件
        optimizer 优化函数
        criterion 损失函数
    '''
    net.train() # 网络设为train模式，可以不断更新权重
   
    epoch_size = len(dataset) // cfg.N
    print('Epoch size(train)', epoch_size)
    iteration = 0 # 每个epoch内训练的迭代次数累计
    running_loss = 0.0 # 每个epoch内所有批次的总损失平均值
    running_reg_loss = 0.0 # 每个epoch内所有批次的预测损失平均值
    running_conf_loss = 0.0 # 每个epoch内所有批次的分类损失平均值
    for voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, images, calibs, ids in data_loader:
        # voxel_features[B,N1,35,7],非空体素每个有35个点，其余voxel_coords[B,N1,3]，每个非空体素的xyz位置用于还原回来用
        # wrapper to variable
        
        voxel_features = torch.tensor(voxel_features).to(cfg.device)
        pos_equal_one = [] # 正样本框
        neg_equal_one = [] # 负样本框
        targets = []  # 正样本框相对于gt的偏移量
        # pos_equal_one、neg_equal_one和targets 需要在无梯度的情况下计算
        with torch.no_grad():
            for i in range(len(gt_box3d)):
                pos_equal_one_, neg_equal_one_, targets_ = dataset.cal_target(gt_box3d_corner[i], gt_box3d[i], cfg)
                pos_equal_one.append(pos_equal_one_)
                neg_equal_one.append(neg_equal_one_)
                targets.append(targets_) 
        #对张量进行扩维拼接(B,H,W,2)         
        pos_equal_one = torch.stack(pos_equal_one, dim=0) 
        #(B,H,W,2)
        neg_equal_one = torch.stack(neg_equal_one, dim=0)
        #(B,H,W,14)
        targets = torch.stack(targets, dim=0) 
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        t0 = time.time()
        # 体素特征和体素的对应网格坐标一起送入网络
        score, reg = net(voxel_features, voxel_coords)
        # calculate loss
        conf_loss, reg_loss, _, _, _ = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
        #loss = hyper['lambda'] * conf_loss + reg_loss
        loss = conf_loss + reg_loss
        running_loss += (reg_loss.item() + conf_loss.item())
        running_conf_loss += conf_loss.item()
        running_reg_loss += reg_loss.item()
        # backward
        loss.backward()
        optimizer.step()
        # 该迭代batch_size结束
        t1 = time.time()
        iteration += 1
        if iteration % 5 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print("epoch(train) : " + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss: %.6f || Loc Loss: %.6f || Conf Loss: %.6f' % \
                ( running_loss/iteration, running_reg_loss/iteration, running_conf_loss/iteration))
        
        
    
    # 该epoch的所有批次数据训练完成
    scheduler.step() # 自动调整学习率（指定的epoch）
    # 输出该epoch的平均总体损失、平均回归损失和平均分类损失
    print('epoch(training) : ' + repr(epoch) + ' ||avg total_loss: %.6f || avg Loc Loss: %.6f || avg Conf Loss: %.6f' % \
     ( running_loss / epoch_size, running_reg_loss/epoch_size, running_conf_loss/epoch_size))
def test(dataset,data_loader,net, model_name, hyper, cfg,criterion, epoch):
    net.eval() # 模型开启评估模式
    epoch_size = len(dataset) // cfg.N
    print('Epoch size(validate)', epoch_size)
    iteration = 0 # 每个epoch内训练的迭代次数累计
    running_loss = 0.0 # 每个epoch内所有批次的总损失平均值
    running_reg_loss = 0.0 # 每个epoch内所有批次的预测损失平均值
    running_conf_loss = 0.0 # 每个epoch内所有批次的分类损失平均值
    with torch.no_grad():
        for voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, images, calibs, ids in data_loader:
            # wrapper to variable
            voxel_features = torch.tensor(voxel_features).to(cfg.device)
            pos_equal_one = []
            neg_equal_one = []
            targets = []
            for i in range(len(gt_box3d)):
                pos_equal_one_, neg_equal_one_, targets_ = dataset.cal_target(gt_box3d_corner[i], gt_box3d[i], cfg)
                pos_equal_one.append(pos_equal_one_)
                neg_equal_one.append(neg_equal_one_)
                targets.append(targets_)
            pos_equal_one = torch.stack(pos_equal_one, dim=0)
            neg_equal_one = torch.stack(neg_equal_one, dim=0)
            targets = torch.stack(targets, dim=0)
            # forward
            score, reg = net(voxel_features, voxel_coords)
            # calculate loss
            conf_loss, reg_loss, xyz_loss, whl_loss, r_loss = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
            loss = conf_loss + reg_loss
            running_conf_loss += conf_loss.item()
            running_reg_loss += reg_loss.item()
            running_loss += (reg_loss.item() + conf_loss.item())
            iteration += 1
    # 如果比前一次训练的好，则保存该模型
    if epoch % 5 == 0:
        torch.save({
            "epoch": epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(r'E:\zqw\PaperCode\OtherClassicalAlgorithm\voxelnet_pytorch_RPFey\TrainModel', model_name+str(epoch)+'.pt'))
    # 输出该epoch的平均总体损失、平均回归损失和平均分类损失
    print('epoch(validate) : ' + repr(epoch) + ' ||avg total_loss: %.4f || avg Loc Loss: %.4f || avg Conf Loss: %.4f' % \
     ( running_loss / epoch_size, running_reg_loss/epoch_size, running_conf_loss/epoch_size))

if __name__ == '__main__':
    pre_model = args.ckpt # 预训练参数
    model_name = "model_t&v_"

    net = VoxelNet().to(cfg.device) # 模型初始化
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay)
    # define loss function
    criterion = VoxelLoss(alpha=hyper['alpha'], beta=hyper['beta'], gamma=hyper['gamma'])
    #动态调整学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epoch*x) for x in (0.5,0.7, 0.9)], gamma=0.1)
    scheduler.last_epoch = cfg.last_epoch + 1
    # 判断是否有预训练模型
    pre_model_root = r'E:\zqw\PaperCode\OtherClassicalAlgorithm\voxelnet_pytorch_RPFey\TrainModel'
    if pre_model is not None and os.path.exists(os.path.join(pre_model_root,pre_model)) :
        '''
            加载预训练模型
        '''
        ckpt = torch.load(os.path.join(pre_model_root,pre_model), map_location=cfg.device)
        net.load_state_dict(ckpt['model_state_dict'])
        cfg.last_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else :
        # 如果没有预训练权重，直接初始化，运用apply()调用weight_init函数.
        net.apply(weights_init)   

    scheduler.last_epoch = cfg.last_epoch + 1
    epoch = cfg.last_epoch
    time_start = time.time()  # 记录开始时间
    #  执行的程序
    while epoch < args.epoch :
        # 训练函数
        train(training_dataset,training_data_loader,net, hyper, cfg, optimizer,criterion, epoch)
        # 验证函数
        test(val_dataset,val_data_loader,net, model_name, hyper, cfg,criterion, epoch)
        epoch += 1
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # 输出训练时间
    print("网络训练与验证时间：", time_sum, "秒")
