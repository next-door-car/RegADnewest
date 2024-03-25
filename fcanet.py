import math
import torch
import torch.nn as nn
 
 
def get_freq_indices(method):
    # 获得分量排名的坐标
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y
 
 
class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
 
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
 
        assert len(mapper_x) == len(mapper_y)
        # assert channel % len(mapper_x) == 0
 
        self.num_freq = len(mapper_x)
 
        # fixed DCT init
        # 对应于公式中的H,W,i,j,C
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
 
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))
 
        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
 
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))
 
        # num_freq, h, w
 
    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        print(x.shape) # [24, 72, 56, 56]
        print(self.weight.shape) # [288, 56, 56]
        x = x * self.weight
 
        result = torch.sum(x, dim=[2, 3])  # 在空间维度上求和
        return result
 
    def build_filter(self, pos, freq, POS):  # 对应i/j, h/w, H/W
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # 基函数公式的一半
        if freq == 0:
            # 对应gap的形式
            return result
        else:
            return result * math.sqrt(2)
 
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)  # 对于每一个BATCH都是相同的
 
        # c_part = channel // len(mapper_x)  # 每一份的通道长度
        c_part = 1
 
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # dct_filter[i: (i + 1), t_x, t_y] = self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
 
        return dct_filter
 
# c2wh为生成的字典，字典名为输入Fca模块的通道数，字典值为通道数对应的输入特征图大小
c2wh = dict([(16,112),(64,112),(72,56),(120,28),(240,28),(200,14),(184,14),(480,14),(672,14),(960,7)])
# input_c为输入通道数
self.att = MultiSpectralAttentionLayer(input_c, 
                                       c2wh[input_c], c2wh[input_c],  
                                       reduction=reduction, 
                                       freq_sel_method = 'top16')

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # dct_h = c2wh[planes], dct_w = c2wh[planes]
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        # mapper_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2]
        # mapper_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2]
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)  # 16
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
 
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)
 
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
 
