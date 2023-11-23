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


import sys
import time


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--ckpt', type=str, default=None, help='pre_load_ckpt')
parser.add_argument('--index', type=int, default=0, help='hyper_tag')
parser.add_argument('--epoch', type=int , default=211, help="training epoch")
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

hyper = {'alpha': 1.0, # 正样本损失项常数
          'beta': 10.0, # 负样本损失项常数
          'pos': 0.75, # 正样本阈值
          'neg': 0.5,  # 负样本阈值
          'lr':0.005,  # 学习率
          'momentum': 0.9, # 动量（Momentum）是一种优化算法，用于加速神经网络的训练
          'lambda': 2.0,
          'gamma':3, # 回归损失函数SmoothL1Loss 在L1~L2之间变化的阈值,默认为1.0
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
    dataset = KittiDataset(cfg=cfg,root = r'C:\zqw\PaperCode\data\ObjectDetection\kitti_original',set=train_set,type = train_type)
    data_loader = data.DataLoader(dataset, batch_size=cfg.N, num_workers=4, collate_fn = detection_collate, shuffle=True, \
                              pin_memory=False)
    net.train() # 网络设为train模式，可以不断更新权重
    # define optimizer
    
    # define loss function
    criterion = VoxelLoss(alpha=hyper['alpha'], beta=hyper['beta'], gamma=hyper['gamma'])
    running_loss = 0.0
    running_reg_loss = 0.0
    running_conf_loss = 0.0
    # training process
    # batch_iterator = None
    epoch_size = len(dataset) // cfg.N
    print('Epoch size', epoch_size)
    #动态调整学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epoch*x) for x in (0.7, 0.9)], gamma=0.1)
    scheduler.last_epoch = cfg.last_epoch + 1
    optimizer.zero_grad()
    epoch = cfg.last_epoch
    while epoch < args.epoch :
        iteration = 0
        for voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, images, calibs, ids in data_loader:
            # voxel_features[B,N1,35,7],非空体素每个有35个点，其余voxel_coords[B,N1,3]，每个非空体素的xyz位置用于还原回来用
            # wrapper to variable
            voxel_features = torch.tensor(voxel_features).to(cfg.device)
            pos_equal_one = [] # 正样本框
            neg_equal_one = [] # 负样本框
            targets = []  # 正样本框相对于gt的偏移量
            with torch.no_grad():
                # Calculate ground-truth
                pos_equal_one, neg_equal_one, targets =  dataset.cal_target2(gt_box3d_corner,gt_box3d,cfg)
            #     for i in range(len(gt_box3d)):
            #         pos_equal_one_, neg_equal_one_, targets_ = dataset.cal_target(gt_box3d_corner[i], gt_box3d[i], cfg)
            #         pos_equal_one.append(pos_equal_one_)
            #         neg_equal_one.append(neg_equal_one_)
            #         targets.append(targets_)  
            # #对张量进行扩维拼接(B,H,W,2)         
            # pos_equal_one = torch.stack(pos_equal_one, dim=0) 
            # #(B,H,W,2)
            # neg_equal_one = torch.stack(neg_equal_one, dim=0)
            # #(B,H,W,14)
            # targets = torch.stack(targets, dim=0)    
            # zero the parameter gradients
            # forward
            # 体素特征和体素的对应网格坐标一起送入网络
            score, reg = net(voxel_features, voxel_coords)
            # calculate loss
            loss, conf_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
            # loss = hyper['lambda'] * conf_loss + reg_loss
            # conf_loss, reg_loss, _, _, _ = criterion(reg, score, pos_equal_one, neg_equal_one, targets)
            #loss = hyper['lambda'] * conf_loss + reg_loss
            running_conf_loss += conf_loss.item()
            running_reg_loss += reg_loss.item()
            running_loss += (reg_loss.item() + conf_loss.item())
            # backward
            loss.backward()
            # visualize gradient
            if iteration == 0 and epoch % 30 == 0:
                plot_grad(net.svfe.vfe_1.fcn.linear.weight.grad.view(-1), epoch,  "vfe_1_grad_%d"%(epoch))
                plot_grad(net.svfe.vfe_2.fcn.linear.weight.grad.view(-1), epoch,"vfe_2_grad_%d"%(epoch))
                plot_grad(net.cml.conv3d_1.conv.weight.grad.view(-1), epoch,"conv3d_1_grad_%d"%(epoch))
                plot_grad(net.rpn.reg_head.conv.weight.grad.view(-1), epoch,"reghead_grad_%d"%(epoch))
                plot_grad(net.rpn.score_head.conv.weight.grad.view(-1), epoch,"scorehead_grad_%d"%(epoch))

            # update 没有每次batch_size进行更新优化参数，相当于提高了batch_size
            if iteration%10 == 9:
                for param in net.parameters():
                    param.grad /= 10
                optimizer.step()
                optimizer.zero_grad()

            if iteration % 20 == 0:
                writer.add_scalar('total_loss', running_loss/20, epoch * epoch_size + iteration)
                writer.add_scalar('reg_loss', running_reg_loss/20, epoch * epoch_size + iteration)
                writer.add_scalar('conf_loss',running_conf_loss/20, epoch * epoch_size + iteration)

                print("epoch : " + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss: %.6f || Loc Loss: %.6f || Conf Loss: %.6f' % \
                ( running_loss/20, running_reg_loss/20, running_conf_loss/3.0))
               
                running_conf_loss = 0.0
                running_loss = 0.0
                running_reg_loss = 0.0

            # visualization
            if iteration == 2000:
                reg_de = reg.detach()
                score_de = score.detach()
                with torch.no_grad():
                    pre_image = draw_boxes(reg_de, score_de, images, calibs, ids, 'pred')
                    gt_image = draw_boxes(targets.float(), pos_equal_one.float(), images, calibs, ids, 'true')
                    try :
                        writer.add_image("gt_image_box {}".format(epoch), gt_image, global_step=epoch * epoch_size + iteration, dataformats='NHWC')
                        writer.add_image("predict_image_box {}".format(epoch), pre_image, global_step=epoch * epoch_size + iteration, dataformats='NHWC')
                    except :
                        pass
            iteration += 1
        scheduler.step()
        epoch += 1
        if epoch % 30 == 0:
            torch.save({
                "epoch": epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(r'E:\zqw\PaperCode\OtherClassicalAlgorithm\voxelnet_pytorch_RPFey\TrainModel', model_name+str(epoch)+'.pt'))



if __name__ == '__main__':
    pre_model = args.ckpt # 预训练参数
    cfg.pos_threshold = hyper['pos']
    cfg.neg_threshold = hyper['neg']
    model_name = "model_%d"%(args.index+1)
    # 构建TensorBoard，方便模型训练过程的可视化
    writer = SummaryWriter('runs/%s'%(model_name[:-4]))

    net = VoxelNet()
    net.to(cfg.device)
    optimizer = optim.SGD(net.parameters(), lr=hyper['lr'], momentum = hyper['momentum'], weight_decay=hyper['weight_decay'])

    pre_model_root = r'E:\zqw\PaperCode\OtherClassicalAlgorithm\voxelnet_pytorch_RPFey\TrainModel'
    if pre_model is not None and os.path.exists(os.path.join(pre_model_root,pre_model)) :
        ckpt = torch.load(os.path.join(pre_model_root,pre_model), map_location=cfg.device)
        net.load_state_dict(ckpt['model_state_dict'])
        cfg.last_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else :
        # 如果没有预训练权重，直接初始化，运用apply()调用weight_init函数.
        net.apply(weights_init)   

    time_start = time.time()  # 记录开始时间
    # function()   执行的程序
    train(net, model_name, hyper, cfg, writer, optimizer)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # 输出训练时间
    print("网络训练时间：", time_sum, "秒")
    writer.close()
