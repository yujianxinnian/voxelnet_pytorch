from __future__ import division
import os
import os.path
import torch.utils.data as data
import utils
#from utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from data_aug import aug_data
#from box_overlaps import bbox_overlaps
from myutil.helloworld import *
from myutil.box_overlaps import *
import numpy as np
import cv2
import torch
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

class KittiDataset(data.Dataset):
    def __init__(self, cfg, root='./kitti',set='train',type='velodyne_train'):
        '''
            set 可选值为train（训练）、val（验证）和test（测试）
            type 训练类型：训练或者验证集选择velodyne_train，测试集选择velodyne_test
        '''
        self.set = set
        self.type = type
        self.root = root
        if set=='test':
            self.data_path = os.path.join(root, 'testing')
        else:
            self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velo_crop")
        self.image_path = os.path.join(self.data_path, "image_2")
        self.calib_path = os.path.join(self.data_path, "calib")
        self.label_path = os.path.join(self.data_path, "label_2")
        # 数据集划分，训练、验证和测试
        with open(os.path.join(root, 'split','%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

        self.T = cfg.T #每个体素内的最大点云数量
        # 体素大小
        self.vd = cfg.vd #深度
        self.vh = cfg.vh # 高度
        self.vw = cfg.vw # 宽度
        # 点云范围
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        # 设置anchors
        self.anchors = torch.tensor(cfg.anchors.reshape(-1,7)).float().to(cfg.device)
        self.anchors2 = cfg.anchors.reshape(-1,7)
        self.anchors_xylwr = self.anchors[..., [0, 1, 5, 4, 6]].contiguous()
        self.feature_map_shape = (int(cfg.H / 2), int(cfg.W / 2))
        self.anchors_per_position = cfg.anchors_per_position
        # 非极大值抑制
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold

    def cal_target2(self, batch_gt_box3d_corner, batch_gt_box3d, cfg,coordinate = 'lidar'):
        # Input:
        #   batch_gt_box3d: (N, N') # 地面真值 依次为：xyzhwlr
        #   feature_map_shape: (w, l) self.feature_map_shape
        #   anchors: (w, l, 2, 7) self.anchors
        # Output:
        #   pos_equal_one (N, w, l, 2)
        #   neg_equal_one (N, w, l, 2)
        #   targets (N, w, l, 14)
        # Attention: cal IoU on birdview
        batch_size = len(batch_gt_box3d)
        # Defined in eq(1) in 2.2
        anchors_reshaped = self.anchors2
        anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        pos_equal_one = np.zeros((batch_size, *self.feature_map_shape, 2))
        neg_equal_one = np.zeros((batch_size, *self.feature_map_shape, 2))
        targets = np.zeros((batch_size, *self.feature_map_shape, 14))
        for batch_id in range(batch_size):
            gt_box3d = batch_gt_box3d[batch_id]
            gt_box3d_corner = batch_gt_box3d_corner[batch_id]
            # BOTTLENECK; from (x,y,w,l) to (x1,y1,x2,y2)
            anchors_standup_2d = utils.anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
             # BOTTLENECK
            gt_standup_2d = utils.corner_to_standup_box2d(gt_box3d_corner[:, 0:4, 0:2])
            # BOTTLENECK
            #gt_xyzhwlr = torch.tensor(gt_box3d, requires_grad=False).float().to(cfg.device)
            # gt_xylwr = gt_xyzhwlr[..., [0, 1, 5, 4, 6]]
            # iou1 = pairwise_iou_rotated(
            #     torch.from_numpy(anchors_standup_2d).float(),
            #     torch.from_numpy(gt_standup_2d).float()
            # ).cpu().numpy() # (gt - anchor)
            iou = bbox_overlaps(
                np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
                np.ascontiguousarray(gt_standup_2d).astype(np.float32),
            )
            # Find anchor with highest iou (iou should also > 0)
            id_highest = np.argmax(iou.T, axis = 1)
            id_highest_gt = np.arange(iou.T.shape[0])
            mask = iou.T[id_highest_gt, id_highest] > 0
            id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

            # Find anchor iou > cfg.XXX_POS_IOU
            id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
            # Find anchor iou < cfg.XXX_NEG_IOU
            id_neg = np.where(np.sum(iou < self.neg_threshold, axis = 1) == iou.shape[1])[0]
            id_pos = np.concatenate([id_pos, id_highest])
            id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
            # TODO: uniquify the array in a more scientific way
            id_pos, index = np.unique(id_pos, return_index = True)
            id_pos_gt = id_pos_gt[index]
            id_neg.sort()
            # Cal the target and set the equal one
            index_x, index_y, index_z = np.unravel_index(id_pos, (*self.feature_map_shape, self.anchors_per_position))
            pos_equal_one[batch_id, index_x, index_y, index_z] = 1
            # ATTENTION: index_z should be np.array
            targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
                gt_box3d[id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
                gt_box3d[id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
                gt_box3d[id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / cfg.ANCHOR_H
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
                gt_box3d[id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
                gt_box3d[id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
                gt_box3d[id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
            targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box3d[id_pos_gt, 6] - anchors_reshaped[id_pos, 6])
            index_x, index_y, index_z = np.unravel_index(id_neg, (*self.feature_map_shape, 2))
            neg_equal_one[batch_id, index_x, index_y, index_z] = 1
            # To avoid a box be pos/neg in the same time
            index_x, index_y, index_z = np.unravel_index(id_highest, (*self.feature_map_shape, 2))
            neg_equal_one[batch_id, index_x, index_y, index_z] = 0
        return pos_equal_one, neg_equal_one, targets
           

    def cal_target(self, gt_box3d_corner, gt_xyzhwlr, cfg):
        '''
            真实框和锚框送入dataset.cal_target计算真实偏移。然后求这两者的损失
        '''
        # Input:
        #   gt_box3d_corner：(8, 3)地面真值八个角度
        #   gt_xyzhwlr即是gt_box3d: (N, ) # 地面真值 依次为：xyzhwlr
        #   feature_map_shape: (w, l) self.feature_map_shape
        #   anchors: (w, l, 2, 7) self.anchors
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview

        anchors_d = torch.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2).to(cfg.device)
        # denote whether the anchor box is pos or neg
        pos_equal_one = torch.zeros((*self.feature_map_shape, 2)).to(cfg.device)
        neg_equal_one = torch.zeros((*self.feature_map_shape, 2)).to(cfg.device)
        targets = torch.zeros((*self.feature_map_shape, 14)).to(cfg.device)

        gt_xyzhwlr = torch.tensor(gt_xyzhwlr, requires_grad=False).float().to(cfg.device)
        
        # BOTTLENECK
        # gt_xylwr = gt_xyzhwlr[..., [0, 1, 5, 4, 6]]
        # iou = pairwise_iou_rotated(
        #     self.anchors_xylwr,
        #     gt_xylwr.contiguous()
        # ).cpu().numpy() # (gt - anchor)

        '''
            以下是参考Hqss中utils.cal_rpn_target代码计算iou
        '''
        # BOTTLENECK; from (x,y,w,l) to (x1,y1,x2,y2)
        # Defined in eq(1) in 2.2
        anchors_reshaped = self.anchors2
        anchors_standup_2d = utils.anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
            # BOTTLENECK
        gt_standup_2d = utils.corner_to_standup_box2d(gt_box3d_corner[:, 0:4, 0:2])
        # BOTTLENECK
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        '''
            以上是参考Hqss中utils.cal_rpn_target代码计算iou
        '''
        id_highest = np.argmax(iou.T, axis=1)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0 # make sure all the iou is positive
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold,
                                 axis=1) == iou.shape[1])[0] # anchor doesn't match ant ground truth
        '''
            以下skyhehe123代码
        '''
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
        '''
            以上skyhehe123代码，替换掉下面注释掉的原RPFey的代码
        '''
        '''
            以下与skyhehe123不同处
        '''
        # print(iou.shape[1])
        # for gt in range(iou.shape[1]):
        #     print(id_highest[gt])
        #     print(iou[id_highest[gt], gt])
        #     if gt not in id_pos_gt and iou[id_highest[gt], gt] > self.neg_threshold:
        #         id_pos = np.append(id_pos, id_highest[gt])
        #         id_pos_gt = np.append(id_pos_gt, gt)
      
        # # sample the negative points to keep ratio as 1:10 with minimum 500
        # num_neg = 10 * id_pos.shape[0]
        # if num_neg < 500:
        #     num_neg = 500
        # if id_neg.shape[0] > num_neg:
        #     np.random.shuffle(id_neg)
        #     id_neg = id_neg[:num_neg]
        '''
            以上与skyhehe123不同处
        '''
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.feature_map_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # ATTENTION: index_z should be np.array

        # parameterize the ground truth box relative to anchor boxs
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = torch.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = torch.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = torch.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time  （skyhehe123中的代码）
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        return pos_equal_one, neg_equal_one, targets

    def preprocess(self, lidar):
        # This func cluster the points in the same voxel.  
        # voxel_coords是通过范围和目标长宽高计算的体素坐标
        # voxel_features用于存储每个体素内的点云及特征，每个体素内点云有35个点，其中每个点的特征包括xyzr及xyz与平均值差值及δx\δy\δz     
        if set=='train':
            # shuffling the points
            np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)

        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]
        # 返回按纵轴去重后的voxel_coords，以及原始元素出现在新元素的索引值inv_ind及重复行出现的次数voxel_counts
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        voxel_features = np.array(voxel_features)
        return voxel_features, voxel_coords

    def __getitem__(self, i):
        '''
            原始数据路劲，包括点云（lidar_file）、图像（image_file）和参数校准（calib_file）文件
        '''
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'       
        image_file = self.image_path + '/' + self.file_list[i] + '.png'
        # 加载参数校准标定文件并读取数据        
        calib = utils.load_kitti_calib(calib_file)
        Tr = calib['Tr_velo2cam'] # velodyne到camera的旋转平移矩阵
        # 加载点云文件并读取数据
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        # 判断是测试还是训练（验证）
        if self.type == 'velodyne_train':
            '''
                训练和验证都走这个流程
            '''
            label_file = self.label_path + '/' + self.file_list[i] + '.txt'
            # 提取标签数据（地面真值8个角的坐标和地面真实3D框xyzhwlr，并将坐标统一到点云坐标系
            gt_box3d_corner, gt_box3d = utils.load_kitti_label(label_file, Tr)   
            gt_box3d_copy = np.copy(gt_box3d)       
            image = cv2.imread(image_file)            
            if self.set=='train':
                # data augmentation
                # lidar, gt_box3d = aug_data(lidar, gt_box3d) #论文原始代码中gt_box3d是指gt_box3d_corner
                lidar, gt_box3d_corner = aug_data(lidar, gt_box3d_corner)
                # 做了数据增强，那么必须通过gt_box3d_corner重新计算gt_box3d
                gt_box3d = utils.box3d_corner_to_center_batch(gt_box3d_corner)
            # specify a range（不同物体的坐标范围不同，需要过滤掉不在范围内的点云）
            lidar, gt_box3d_corner, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d_corner, gt_box3d)

            # 在体素化和体素采样之前，先进行曲率下采样

            # voxelize，体素化，建立体素坐标及对应体素的特征
            voxel_features, voxel_coords = self.preprocess(lidar)

            # bounding-box encoding
            # pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d_corner, gt_box3d)

            return voxel_features, voxel_coords, gt_box3d_corner, gt_box3d, image, calib, self.file_list[i] # pos_equal_one, neg_equal_one, targets, image, calib, self.file_list[i]

        elif self.type == 'velodyne_test':
            '''
                测试评估，不需要处理标签数据，因为没有标签数据
            '''
            image = cv2.imread(image_file)

            # lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)
            lidar = utils.get_filtered_lidar(lidar)

            voxel_features, voxel_coords = self.preprocess(lidar)

            return voxel_features, voxel_coords, image, calib, self.file_list[i]

        else:
            raise ValueError('the type invalid')


    def __len__(self):
        return len(self.file_list)




