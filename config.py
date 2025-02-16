'''
    在该项目中，裁剪的点云数据用于训练和验证。图像坐标之外的点云将被移除。
'''

import math
import numpy as np

class config:

    # classes
    class_list = ['Car', 'Van'] # 也可以是Pedestrian and Cyclist

    # batch size
    N=1

    # maxiumum number of points per voxel
    T=35

    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2
    
    # points cloud range
    if 'Car' or 'Van' in class_list:
        xrange = (0.0, 70.4)
        yrange = (-40, 40)
        zrange = (-3, 1)
    elif 'Pedestrian' or 'Cyclist' in class_list:
        xrange = (0.0, 48)
        yrange = (-20, 20)
        zrange = (-3, 1)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # iou threshold
    ''' 
        对于车辆检测：
        如果锚点与地面实况的交集（IoU）最高，或者与地面实况的 IoU 高于 0.6（鸟瞰图），则该锚点被认为是正的。 
        如果锚点与所有地面实况框之间的 IoU 小于 0.45，则该锚点被视为负锚点。 
        将锚点视为不关心它们是否具有 0.45 ≤ IoU ≤ 0.6 与任何基本事实。
        设置 α =1.5 和 β =1
        对于行人和骑行者的检测：
        如果一个anchor与groundtruth具有最高的IoU，或者它与groundtruth的IoU高于0.5，将其指定为积极的。 
        如果锚点与每个真实值的IoU小于0.35，则该锚点被视为负锚点。
        对于任何地面实况具有0.35 ≤ IoU ≤ 0.5的锚点，将它们视为无关紧要。


    '''
    if 'Car' or 'Van' in class_list:
        pos_threshold = 0.6  # 正样本阈值
        neg_threshold = 0.45 # 负样本阈值
        ANCHOR_L = 3.9
        ANCHOR_W = 1.6
        ANCHOR_H = 1.56
        ANCHOR_Z = -1.0 - ANCHOR_H/2
    elif 'Pedestrian' or 'Cyclist' in class_list:
        pos_threshold = 0.5  # 正样本阈值
        neg_threshold = 0.35 # 负样本阈值
        ANCHOR_L = 1.76
        ANCHOR_W = 0.6
        ANCHOR_H = 1.73
        ANCHOR_Z = -0.6 - ANCHOR_H/2
    elif class_list == ['Pedestrian']:
        ANCHOR_L = 0.8
    # α,β 是计算损失函数中，平衡正反样本相对重要性的正常数。
    alpha = 1.5 # 正样本损失项常数（分类损失1）
    beta = 1 # 负样本损失项常数（分类损失2）
    #   anchors: (w, l, 2, 7) car(200, 176, 2, 7) x y z h w l r
    x = np.linspace(xrange[0]+vw, xrange[1]-vw, W//2)
    y = np.linspace(yrange[0]+vh, yrange[1]-vh, H//2)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2) 
    cx = np.tile(cx[..., np.newaxis], 2) 
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * ANCHOR_Z # 原本是乘以-1.0
    w = np.ones_like(cx) * ANCHOR_W
    l = np.ones_like(cx) * ANCHOR_L
    h = np.ones_like(cx) * ANCHOR_H
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = 90 / 180 * np.pi # 原本为np.pi/2
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    anchors_per_position = 2

    # non-maximum suppression
    # nms_threshold = 1e-3
    # score_threshold = 0.9
    nms_threshold = 0.1 #skyhehe123中的设置
    score_threshold = 0.96 #skyhehe123中的设置

    device = "cuda:0"
    num_dim = 51

    last_epoch=0
