import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from config import config as cfg

'''
    以下是堆叠体素特征层
'''
# Fully Connected Network
class FCN(nn.Module):
    '''
        逐点进行全连接提取特征，并未提取周边点的特征
    '''

    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t,-1))
        x = F.relu(self.bn(x))
        return x.view(kk,t,-1)

# Voxel Feature Encoding layer
class VFE(nn.Module):
    '''
        每个点用FCN提取点特征后，再按体素内的T个点提取其体素局部聚合特征laf，然后将laf加入到点的特征
    '''

    def __init__(self,cin,cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        # 由于FCN层不是最终的输出层（输出C1维特征），FCN之后还需要逐元素最大池化（也是C1维特征），
        # 然后再逐点聚合特征，聚合后的特征维度是cout，也即2*C1=cout，所以此处的FCN输出维度应该是cout // 2 
        self.units = cout // 2 
        self.fcn = FCN(cin,self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        #locally aggregated feature 在T（35）个点中求max，再unsqueeze在指定位置增加一个维度，得到voxel局部特征
        laf = torch.max(pwf,1)[0].unsqueeze(1).repeat(1,cfg.T,1)
        # point-wise concat feature 将点特征和voxel局部特征在特征维度进行相加
        pwcf = torch.cat((pwf,laf),dim=2)
        # apply mask 因为有些点是填充零的，需要把这些点去掉
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    '''
        通过多次堆叠VFE特征网络后，再用FCN逐点特征网络，最后再用最大池化提取每个体素最终的局部特征
    '''
    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7,32)
        self.vfe_2 = VFE(32,128)
        self.fcn = FCN(128,128)
    def forward(self, x):
        #torch.ne用来判断是否不相等，不等于0的赋值1，表明这个点存在，等于零的赋值0，表示是填充的点
        mask = torch.ne(torch.max(x,2)[0], 0) # 滤掉为零的点
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        # 在第二个维度35中求max，后面加[0]表示获取最大的数据而不是数据和下标，这里得到voxel全局特征
        x = torch.max(x,1)[0]
        return x
'''
    以上是堆叠体素特征层
'''
'''
    以下是中间卷积层
'''
# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)
# Convolutional Middle Layer
#输入维度(2,128,10,400,352)
#输出维度(2,64,2,400,352)
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x
'''
    ##############以上是堆叠体素特征层################
'''
'''
    ##############以下是RPN层################
'''
# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self,in_channels,out_channels,k,s,p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k,stride=s,padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x
#nn.Modulelist和nn.Sequential里面的module是会自动注册到整个网络上的，同时module的parameters也会自动添加到整个网络中，但如果使用list存储则不会添加，也就无法训练
#而且nn.Modulelist里面可以像数组一样调用，也就是可与不按照顺序运行，而nn.Sequential里面的顺序的固定的
# Region Proposal Network
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        #输入维度(128,400,352)
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        #这里的x维度是(768,200,176)
        x = torch.cat((x_0,x_1,x_2), dim = 1)
        return self.score_head(x), self.reg_head(x)
'''
    ##############以上是RPN层################
'''

'''
    ##############以下是模型集成层################
'''
class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()
        self.rpn = RPN()
    #sparse_features(2,N,128)
    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1] # 特征维数，包括xyzr及δxδyδz
        #(128,2,10,400,352)
        dense_feature = torch.zeros(cfg.N, cfg.D, cfg.H, cfg.W, dim).to(cfg.device)

        dense_feature[coords[:,0], coords[:,1], coords[:,2], coords[:,3], :]= sparse_features
        #transpose维度交换 (2,128,10,400,352)
        return dense_feature.permute(0, 4, 1, 2, 3)

    def forward(self, voxel_features, voxel_coords):

        # feature learning network
        vwfs = self.svfe(voxel_features) #(2,N,35,7)-----(2,N,128)
        vwfs = self.voxel_indexing(vwfs,voxel_coords) #(2,N,128)----(2,128,10,400,352)

        # convolutional middle network
        cml_out = self.cml(vwfs) #(2,128,10,400,352)----(2,64,2,200,176)

        # （region proposal network）

        # merge the depth and feature dim into one, output probability score map and regression map
        score, reg = self.rpn(cml_out.reshape(cfg.N,-1,cfg.H, cfg.W)) #(2,128,200,176)----(2,2,200,176)and(2,14,200,176)
        score = F.sigmoid(score) # 采用新的loss计算后，就不需要.permute(0,2,3,1)
        '''
            以下是Hqss中添加的代码，skyhehe123中将该代码放到了loss.py中
        '''
        # score = torch.sigmoid(score)
        # score = score.permute((0, 2, 3, 1))
        # score = F.sigmoid(score.permute(0,2,3,1)) [2, 200, 176, 2]
        
        return score, reg

