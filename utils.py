from __future__ import division
import numpy as np
from config import config as cfg
import math
# import mayavi.mlab as mlab
import cv2
#from box_overlaps import *
from data_aug import aug_data
import matplotlib.pyplot as plt
import matplotlib
import os
import torch

def get_filtered_lidar(lidar, boxes3d=None, gt_box3d=None):
    '''
        选择指定范围内的点云数据，范围参数在config.py中设置
    '''
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= cfg.xrange[0]) & (boxes3d[:, :, 0] < cfg.xrange[1])
        box_y = (boxes3d[:, :, 1] >= cfg.yrange[0]) & (boxes3d[:, :, 1] < cfg.yrange[1])
        box_z = (boxes3d[:, :, 2] >= cfg.zrange[0]) & (boxes3d[:, :, 2] < cfg.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z,axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz>0], gt_box3d[box_xyz>0]

    return lidar[filter_xyz]

def lidar_to_bev(lidar):

    X0, Xn = 0, cfg.W
    Y0, Yn = 0, cfg.H
    Z0, Zn = 0, cfg.D

    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    qxs=((pxs-cfg.xrange[0])/cfg.vw).astype(np.int32)
    qys=((pys-cfg.yrange[0])/cfg.vh).astype(np.int32)
    qzs=((pzs-cfg.zrange[0])/cfg.vd).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-cfg.zrange[0])
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            top[-qxs[i], -qys[i], -2]=prs[i]


    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image = top[:,:,-1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    return top, density_image


def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections

def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=1):

    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            i,j=k,(k+1)%4

            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)

    return img

def _quantize_coords(x, y):
    xx = cfg.H - int((y - cfg.yrange[0]) / cfg.vh)
    yy = cfg.W - int((x - cfg.xrange[0]) / cfg.vw)
    return xx, yy

def  draw_polygons(image, polygons,color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for polygon in polygons:
        tup0, tup1, tup2, tup3 = [_quantize_coords(*tup) for tup in polygon]
        cv2.line(img, tup0, tup1, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup1, tup2, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup2, tup3, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup3, tup0, color, thickness, cv2.LINE_AA)
    return img

def draw_rects(image, rects, color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for rect in rects:
        tup0,tup1 = [_quantize_coords(*tup) for tup in list(zip(rect[0::2], rect[1::2]))]
        cv2.rectangle(img, tup0, tup1, color, thickness, cv2.LINE_AA)
    return img

def load_kitti_calib(calib_file):
    """
        load projection matrix
        P0 – P3（3x4）：就是对应的cam0 ~ cam3这四个相机矫正后的投影矩阵，每个都是3*4的矩阵
        R0_rect（3x3）：矫正后的相机矩阵，注意在使用的时候需要reshape成4x4，具体方法是在R(4, 4)处添1，其余6个位置添0。
        Tr_velo_to_cam（3x4）：velodyne到camera的旋转平移矩阵，此矩阵包含两个模块，左侧3x3的旋转矩阵和右侧13的平移向量，具体使用时也要reshape成44，具体方法是在最后添加一行（0,0,0,1）。
        Tr_imu_to_velo（3x4）：IMU到velodyne的旋转平移矩阵，结构和使用方法跟Tr_velo_to_cam类似。
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

# def angle_in_limit(angle):
#     # 论文原始代码中是ry_to_rz函数处理 To limit the angle in -pi/2 - pi/2
#     limit_degree = 5
#     while angle >= np.pi / 2:
#         angle -= np.pi
#     while angle < -np.pi / 2:
#         angle += np.pi
#     if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
#         angle = np.pi / 2
#     return angle

def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)
    
    def angle_in_limit(angle):
        # 论文原始代码中是ry_to_rz函数处理 To limit the angle in -pi/2 - pi/2
        limit_degree = 5
        while angle >= np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
        if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
            angle = np.pi / 2
        return angle
    
    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = angle_in_limit(-ry-np.pi/2)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32), np.array([t_lidar[0,0], t_lidar[0,1], t_lidar[0,2], h, w, l, rz])

def anchors_center_to_corner(anchors):
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        box2d = cornerPosInVelo.transpose()
        anchor_corner[i] = box2d
    return anchor_corner


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d

def box3d_corner_to_center_batch(box3d_corner):
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))

    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)

def get_anchor3d(anchors):
    num = anchors.shape[0]
    anchors3d = np.zeros((num,8,3))
    anchors3d[:, :4, :2] = anchors
    anchors3d[:, :, 2] = cfg.z_a
    anchors3d[:, 4:, :2] = anchors
    anchors3d[:, 4:, 2] = cfg.z_a + cfg.h_a
    return anchors3d

def load_kitti_label(label_file, Tr):
    '''
        通过标签数据，提取地面真实框(gt_boxes3d)及八个角(gt_boxes3d_corner)的坐标，并转换成点云坐标系统下的坐标
        并且只提取config.py中class_list配置的目标类别对应的地面真实框及八个角度的坐标
    '''

    with open(label_file,'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []
    gt_boxes3d = []

    num_obj = len(lines)

    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        # 遍历标注目标，只提取配置文件中class_list设定的目标物体
        obj_class = obj[0].strip()
        if obj_class not in cfg.class_list:
            continue

        box3d_corner, box3d_vertice = box3d_cam_to_velo(obj[8:], Tr)

        gt_boxes3d.append(box3d_vertice)
        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1,8,3)
    gt_boxes3d = np.array(gt_boxes3d).reshape(-1, 7)

    return gt_boxes3d_corner, gt_boxes3d


def test():
    import os
    import glob
    import matplotlib.pyplot as plt

    lidar_path = os.path.join('./data/KITTI/training', "crop/")
    image_path = os.path.join('./data/KITTI/training', "image_2/")
    calib_path = os.path.join('./data/KITTI/training', "calib/")
    label_path = os.path.join('./data/KITTI/training', "label_2/")


    file=[i.strip().split('/')[-1][:-4] for i in sorted(os.listdir(label_path))]

    i=2600

    lidar_file = lidar_path + '/' + file[i] + '.bin'
    calib_file = calib_path + '/' + file[i] + '.txt'
    label_file = label_path + '/' + file[i] + '.txt'
    image_file = image_path + '/' + file[i] + '.png'

    image = cv2.imread(image_file)
    print("Processing: ", lidar_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))

    calib = load_kitti_calib(calib_file)
    gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])

    # augmentation
    #lidar, gt_box3d = aug_data(lidar, gt_box3d)

    # filtering
    lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)

    # view in point cloud

    # fig = draw_lidar(lidar, is_grid=False, is_top_region=True)
    # draw_gt_boxes3d(gt_boxes3d=gt_box3d, fig=fig)
    # mlab.show()

    # view in image

    # gt_3dTo2D = project_velo2rgb(gt_box3d, calib)
    # img_with_box = draw_rgb_projections(image,gt_3dTo2D, color=(0,0,255),thickness=1)
    # plt.imshow(img_with_box[:,:,[2,1,0]])
    # plt.show()

    # view in bird-eye view

    top_new, density_image=lidar_to_bev(lidar)
    # gt_box3d_top = corner_to_standup_box2d_batch(gt_box3d)
    # density_with_box = draw_rects(density_image,gt_box3d_top)
    density_with_box = draw_polygons(density_image,gt_box3d[:,:4,:2])
    plt.imshow(density_with_box,cmap='gray')
    plt.show()

def plot_grad(grad, epoch, name):
    '''
        梯度可视化
    '''
    rootPath = r'E:\zqw\PaperCode\OtherClassicalAlgorithm\voxelnet_pytorch_RPFey'
    if not os.path.exists('%s\\vis\\%d'%(rootPath,epoch)):
        #os.mkdir('%s\\vis\\%d'%(rootPath,epoch)) #在磁盘中直接创建目录（一级目录）
        os.makedirs('%s\\vis\\%d'%(rootPath,epoch)) # 在磁盘中创建多级目录
    plt.figure()
    grad = grad.detach().cpu().numpy()
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    plt.hist(grad, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("value")
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    mean = np.mean(grad)
    var = np.var(grad)
    plt.title("mean: %.4f,  var: %.4f"%(mean, var))
    plt.savefig('%s/vis/%d/%s.png'%(rootPath,epoch, name))

def print_prob(score, name):
    score = score.permute(0, 3, 1, 2)
    score = score.detach().cpu().numpy()
    score_1 = score[0, ...]
    up = score_1[0, ...]
    down = score_1[1, ...]
    score = np.bitwise_or(up > 0.8 , down > 0.8)

    plt.figure()
    plt.imshow(score.astype(np.uint8), cmap='hot')
    plt.colorbar()
    plt.savefig(name)

# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4); x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup

if __name__ == '__main__':
    test()