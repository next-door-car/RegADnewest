import torch.nn as nn
import math
import torch
from torch import Tensor
from stn import STNModule
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Optional

affine_par = True

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, 
                     kernel_size=1, stride=stride, 
                     bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, 
                     kernel_size=3, stride=stride,
                     padding=1, dilation=1,
                     bias=False)

class BasicBlock_V0(nn.Module):
    expansion = 1

    def __init__(self,  
                 inplanes, planes, 
                 stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_V1(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # 输入 x: torch.Size([32, 64, 56, 56])
        identity = x 

        out = self.conv1(x) # torch.Size([32, 64, 56, 56])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # torch.Size([32, 64, 56, 56])
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 inplanes, planes, 
                 stride=1,  dilation_ = 1, 
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, 
                            stride,dilation_=dilation__, downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x


class ResNet_locate(nn.Module):
    def __init__(self, args, block, layers):
        super(ResNet_locate,self).__init__()
        self.resnet = ResNet(args, block, layers)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

def resnet50_locate(args):
    model = ResNet_locate(args, block=Bottleneck, layers=[3, 4, 6, 3])
    return model

class ResNet18_STN(nn.Module):

    def __init__(self, args, block, layers):
        super(ResNet18_STN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # block = BasicBlock
        # layer = [2,2,2,2]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) # layer[0] = 2
        self.stn1 = STNModule(64, 1, args) # 128

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # layer[1] = 2
        self.stn2 = STNModule(128, 2, args) # 64

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # layer[2] = 2
        self.stn3 = STNModule(256, 4, args) # 32 => 8

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # block.expansion = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # 64 != 64 * 1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, 
                            stride, 
                            downsample)) # 第一个block=BaiscBlock(64, 64, 1, downsample)
        self.inplanes = planes * block.expansion # 64
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _fixstn(self, x, theta):
        # 含义是stn网络=>变换后的图像
        grid = F.affine_grid(theta, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform

    def forward(self, x: Tensor) -> Tensor:
        # 输入 x: torch.Size([32, 3, 224, 224])
        x = self.conv1(x) # torch.Size([32, 64, 112, 112]) 112 = (224-7+2*3)/2+1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([32, 64, 56, 56]) 56 = (112-3)/2+1
        # torch.Size([32, 64, 56, 56])

        x = self.layer1(x) # torch.Size([32, 64, 56, 56]) 56 = (56-3)/1+1   128
        x, theta1 = self.stn1(x) # x是变换后的图像，theta1是变换矩阵
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea1 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn1_output = self._fixstn(x.detach().cpu(), fixthea1) # 可视化的时候用
        # after layer1 shape:  torch.Size([32, 64, 56, 56])

        x = self.layer2(x) # torch.Size([32, 128, 28, 28]) 28 = (56-3)/2+1   64
        x, theta2 = self.stn2(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea2 = torch.from_numpy(np.linalg.inv(np.concatenate((theta2.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn2_output = self._fixstn(self._fixstn(x.detach().cpu(), fixthea2), fixthea1) # 恢复
        # after layer2 shape:  torch.Size([32, 128, 28, 28])

        x = self.layer3(x) # torch.Size([32, 256, 14, 14]) 14 = (28-3)/2+1   32
        out, theta3 = self.stn3(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea3 = torch.from_numpy(np.linalg.inv(np.concatenate((theta3.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn3_output = self._fixstn(self._fixstn(self._fixstn(out.detach().cpu(), fixthea3), fixthea2), fixthea1) # 恢复
        # after layer3 shape:  torch.Size([32, 256, 14, 14])

        return out

def resnet18_stn(args, pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18_STN(args, block=BasicBlock_V1, layers=[2, 2, 2, 2])

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

    
class WideResNet_STN(nn.Module):
    def __init__(self, args, pretrained=True):
        super(WideResNet_STN, self).__init__()
        self._init_features()
        # hook,含义是在模型的某一层的输出上注册一个函数，当模型的输出到达这一层时，这个函数就会被调用
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=pretrained)
    def _init_features(self):
        self.features = []
