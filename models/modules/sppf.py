import torch
from torch import nn

class SPPF(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    '''
    作用是为了提取不同尺度的特征，然后将这些特征进行拼接，从而提高模型的感受野。
    '''
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1) # shape 243=(243-1)/1+1
        self.conv2 = nn.Conv2d(hidden_channels * 4, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)  # 使用ReLU激活函数
        self.act2 = nn.ReLU(inplace=True)  # 使用ReLU激活函数
        if isinstance(kernel_sizes, int): # 单个kernel_size
            self.m = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        else:
            self.m = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
            )       

    def forward(self, x):
        # x shape  [Batch, 512，243，243]
        x = self.act1(self.bn1(self.conv1(x))) # x shape  [Batch, 256，243，243]
        if isinstance(self.kernel_sizes, int):
            y1 = self.m(x) # y1 shape  [Batch, 256，243，243]
            y2 = self.m(y1) # y2 shape  [Batch, 256，243，243]
            x = torch.cat([x, y1, y2, self.m(y2)], dim=1) # x shape  [Batch, 1024，243，243]
        else:
            x = torch.cat([x] + [m(x) for m in self.m], dim=1) # x shape  [Batch, 1024，243，243]
        x = self.act2(self.bn2(self.conv2(x))) # x shape  [Batch, 512，243，243]
        return x