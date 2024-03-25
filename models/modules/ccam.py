from .resnet import *
import os
import sys
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import PIL.Image
from ai import *

def get_cam_train(ori_image, cams, scale, flag=1):
    # label = np.array([1])
    # ori_w, ori_h = ori_image.size
    # # preprocessing
    # image = copy.deepcopy(ori_image) 
    # image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
    # image = image.transpose((2, 0, 1))

    # image = torch.from_numpy(image)
    # flipped_image = image.flip(-1)
    
    # images = torch.stack([image, flipped_image])
    # images = images.cuda()

    # if flag:
    #     cams = 1 - cams

    # # postprocessing
    # cams = F.relu(cams)
    # # cams = torch.sigmoid(features)
    # cams = cams[0] + cams[1].flip(-1) # 为什么要进行翻转？

    return cams

def get_cam_inference(model, ori_image, scale, flag=1):
    ori_w, ori_h = ori_image.size
    # preprocessing
    image = copy.deepcopy(ori_image) 
    image = image.resize((round(ori_w*scale), round(ori_h*scale)), 
                            resample=PIL.Image.CUBIC) # 调整图像大小为原始宽度和高度乘以比例 scale，采用双三次插值方法（resample=PIL.Image.CUBIC）
    
    image = normalize_fn(image) # 归一化
    image = image.transpose((2, 0, 1))

    image = torch.from_numpy(image) # 将 NumPy 数组转换为 PyTorch 张量
    flipped_image = image.flip(-1) # 使用 flip(-1) 方法沿着图像的最后一个维度（通道维度）进行翻转，得到 flipped_image 变量。
    
    images = torch.stack([image, flipped_image])
    images = images.cuda()
    
    # inferenece
    _, _, cams = model(images, inference=True) # cams 是一个 2x1xHxW 的张量

    if flag:
        cams = 1 - cams

    # postprocessing
    cams = F.relu(cams)
    # cams = torch.sigmoid(features)
    cams = cams[0] + cams[1].flip(-1) # 为什么要进行翻转？

    return cams

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetSeries(nn.Module):
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        return torch.cat([x2, x1], dim=1)

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C] 含义是：每个样本的前景特征，matul是点乘
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam


class Network(nn.Module):
    def __init__(self, pretrained='mocov2', cin=None):
        super(Network, self).__init__()

        self.backbone = ResNetSeries(pretrained=pretrained)
        self.ac_head = Disentangler(cin)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x, inference=False):

        feats = self.backbone(x)
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


def get_model(pretrained, cin=2048+1024):
    return Network(pretrained=pretrained, cin=cin)
