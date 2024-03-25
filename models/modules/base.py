import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def has_weights(model):
    '''
    检查模型的权重
    '''
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() > 0:
            print(f"Layer {name} has weights.")
        else:
            print(f"Layer {name} does not have weights.")
            
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # 
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


def Conv3x3BNReLU(in_channels,out_channels,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def ConvBNReLU(in_channels,out_channels,kernel_size,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

def ConvBN(in_channels,out_channels,kernel_size,stride,padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        
class ResizeConv2d(nn.Module):
    '''
    上采样时，比反卷积更好的方式
    https://blog.csdn.net/weixin_42663567/article/details/104261635
    '''
    def __init__(self, input_filters, output_filters, kernel, strides):
        super(ResizeConv2d, self).__init__()
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel = kernel
        self.strides = strides

    def forward(self, x):
        height, width = x.size(2), x.size(3)
        new_height = height * self.strides * 2
        new_width = width * self.strides * 2

        x_resized = F.interpolate(x, size=(new_height, new_width), mode='nearest') # 插值再卷积

        conv_layer = nn.Conv2d(self.input_filters, self.output_filters, self.kernel, self.strides)
        return conv_layer(x_resized)