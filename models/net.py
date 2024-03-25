from uu import decode
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.stn import STNModule
# einops是提供常用张量操作的Python包
# pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

# https://gitcode.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cvt.py

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class ResizeConv2d(nn.Module):
    '''
    上采样时，比反卷积更好的方式
    https://blog.csdn.net/weixin_42663567/article/details/104261635
    https://blog.csdn.net/g11d111/article/details/101781549
    '''
    def __init__(self, input_filters, output_filters, kernel, strides, padding, ratio=2):
        super(ResizeConv2d, self).__init__()
        self.input_filters = input_filters 
        self.output_filters = output_filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.ratio = ratio

    def forward(self, x):
        # x.shape (b, 64, 28, 28) # 尺度 (56-3+2*1)/2+1=28
        height, width = x.size(2), x.size(3)
        new_height = height * self.strides * self.ratio  # 上采样 112
        new_width = width * self.strides * self.ratio #  上采样

        x_resized = F.interpolate(x, size=(new_height, new_width), mode='nearest') # 插值再卷积
        # (448-7+2*3)/4+1=112  448/4=112+1=113 448/4=112+1=113
        # (896-7+2*3)/4+1 = 895/4=223+1=224 
        # (112-3+2*1)/2 + 1 = 56  
        conv_layer = nn.Conv2d(self.input_filters, self.output_filters, self.kernel, self.strides, self.padding) # (896-7+2*3)/4+1=223  889/4=222+1=223 895/4=223+1=224
        return conv_layer(x_resized)

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps #  表示为了数值稳定性而添加到方差的小常数，默认值为 1e-5。
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # 是 LayerNorm 操作中可学习的缩放参数（scale），其形状为 (1, dim, 1, 1)，表示每个通道都有一个缩放参数。
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) # 是 LayerNorm 操作中可学习的偏置参数（bias），其形状也为 (1, dim, 1, 1)，表示每个通道都有一个偏置参数。

    def forward(self, x):
        # 计算输入张量 x 在通道维度上的方差 var 和均值 mean，使用 PyTorch 中的 torch.var 和 torch.mean 函数进行计算
        # dim=1 表示在通道维度上进行计算，keepdim=True 表示保持维度。
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        # 对输入张量进行 LayerNorm 操作：(x - mean) / (var + self.eps).sqrt()。
        # 这里先将输入张量减去均值，然后除以标准差（方差加上一个小常数 eps 后开方），这样可以将输入张量进行标准化。
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    # 换为了卷积
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x.shape = (b, emb_dim, h, w)
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, 
                      padding = padding, stride = stride, 
                      groups = dim_in,  
                      bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, 
                      kernel_size = 1, 
                      bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, 
                 dim, proj_kernel, kv_proj_stride, 
                 heads = 8, head_dim = 64, 
                 dropout = 0.):
        '''
        dim 输入特征的维度=>输出不变
        inner_dim 计算内部query和key的维度
        proj_kernel 用于计算内部query和key的卷积核大小（输出特征的维度）+padding
        kv_proj_stride 用于计算query和key的卷积步长
        '''
        super().__init__()
        inner_dim = heads * head_dim  # 内部维度 = 头 * 头维度
        padding = proj_kernel // 2 # padding的大小
        self.heads = heads # 多头注意力的头数
        self.scale = head_dim ** -0.5 # 缩放因子

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # 线性层映射改为卷积层
        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape # shape = (b, emb_dim, per_patch_h, per_patch_w)
        # *shape 将元组 shape 中的元素依次赋值给变量 b, n, _, y，然后将对象 self.heads 的值赋值给变量 h。
        b, n, _, y, h = *shape, self.heads # b = batch_size, n = emb_dim, y = per_patch_h = per_patch_w, h = heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1)) # q.shape = k.shape = v.shape = (b, h * dim_head, per_patch_h, per_patch_w)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale # 计算query和key的点积

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v) # 计算加权和
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)  # out.shape = (b, head * dim_head, per_patch_h, per_patch_w)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, head_dim = 64, mlp_mult = 4, dropout = 0.):
        '''
        dim 输入输出特征的维度
        proj_kernel 用于计算query和key的卷积核大小
        kv_proj_stride 用于计算query和key的卷积步长
        depth transformer的深度
        heads 多头注意力的头数
        head_dim 每个头的维度
        mlp_mult MLP的倍数
        dropout dropout的概率
        '''
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 其中一组qkv
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, 
                          heads = heads, head_dim = head_dim, 
                          dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        # x.shape = (b, emb_dim, h, w)
        for attn, ff in self.layers:
            x = attn(x) + x # x.shape = (b, emb_dim, h, w)
            x = ff(x) + x # ff
        return x

class STNCvT(nn.Module):
    def __init__(
        self,
        args,
        num_classes,
        
        # 多尺度特征提取器（第一层）
        # 输出尺度 224-7+2*3 / 4 + 1 = 56
        # 输出尺度 384-7+2*3 / 4 + 1 = 96 
        s1_emb_en_dim = 64,
        s1_emb_de_dim = 3, # rgb通道
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_emb_ratio = 4, # 上下采样的倍数
        # 内部除了头部，映射层卷积不变
        s1_proj_kernel = 3, # 用于计算query和key的卷积核大小
        s1_kv_proj_stride = 2, # 用于计算query和key的卷积步长
        # heads * head_dim = 64 代表每个头的维度
        s1_en_heads = 1, # 多头注意力的头数
        s1_de_heads = 1, # 多头注意力的头数
        # s1_head_dim = 64, # 默认为64
        s1_depth = 1, # transformer的深度
        s1_mlp_mult = 4, # MLP的倍数

        # 多尺度特征提取器（第二层）
        # 输出尺度  (56-3+2*1)/2+1=28
        # 输出尺度 96-3+2*1 / 2 + 1 = 48
        s2_emb_en_dim = 192,
        s2_emb_de_dim = 64,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_emb_ratio = 2, # 上下采样的倍数
        # 内部除了头部，映射层卷积不变
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        # heads * head_dim = 64 代表每个头的维度
        s2_en_heads = 3,
        s2_de_heads = 3,
        # s2_head_dim = 64, # 默认为64
        s2_depth = 2,
        s2_mlp_mult = 4,
        
        # 多尺度特征提取器（第三层）
        # 输出尺度 (28-3+2*1)/2+1=14
        # 输出尺度 48-3+2*1 / 2 + 1 = 24
        s3_emb_en_dim = 384,
        s3_emb_de_dim = 192,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_emb_ratio = 2, # 上下采样的倍数
        # 内部除了头部，映射层卷积不变
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        # heads * head_dim = 64 代表每个头的维度
        s3_en_heads = 6,
        s3_de_heads = 6,
        # s3_head_dim = 64, # 默认为64
        s3_depth = 10,
        s3_mlp_mult = 4,
        
        dropout = 0.,
        channels = 3
    ):
        super().__init__()
    
        # encoder
        kwargs = dict(locals())
        encoder_dim = channels # 输入特征的维度
        encoder_layers = []
        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs) # 目的是将参数分组，分别传入不同的层

            encoder_layers.append(nn.Sequential(
                nn.Conv2d(encoder_dim, config['emb_en_dim'], 
                          kernel_size = config['emb_kernel'], # 7
                          padding = (config['emb_kernel'] // 2),  # (224-7+2*3)/4+1=56
                          stride = config['emb_stride']), # 输出shape = (b, emb_dim, h, w) 相当于patch embedding
                LayerNorm(config['emb_en_dim']),
                Transformer(dim = config['emb_en_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], 
                            depth = config['depth'], 
                            heads = config['en_heads'], head_dim = 64,
                            mlp_mult = config['mlp_mult'],
                            dropout = dropout), # 输出shape = (b, dim=emb_endim, h, w)
            ))
            # 最后一层的输出作为下一层的输入
            encoder_dim = config['emb_en_dim'] # 编码维度

        self.encoder_layers = nn.Sequential(*encoder_layers) # layers 是一个 nn.Sequential 容器，包含了多尺度特征提取器的所有层
        self.encoder_layer1 = nn.Sequential(*encoder_layers[0]) # layers 是一个 nn.Sequential 容器，包含了多尺度特征提取器的所有层
        self.stn1 = STNModule(args, 64,  24)  # 输入 96
        self.encoder_layer2 = nn.Sequential(*encoder_layers[1])
        self.stn2 = STNModule(args, 192, 12)  # 输入 48
        self.encoder_layer3 = nn.Sequential(*encoder_layers[2])
        self.stn3 = STNModule(args, 384, 6)   # 输入 24
        self.last_encoder_norm = LayerNorm(encoder_dim) # 最后一层的 LayerNorm 层
        
        # decoder
        kwargs = dict(locals())
        decoder_dim = encoder_dim # 默认为encoder的维度
        decoder_layers = []
        for prefix in ('s3', 's2', 's1'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs) # 目的是将参数分组，分别传入不同的层

            decoder_layers.append(nn.Sequential(
                Transformer(dim = decoder_dim, proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], 
                            depth = config['depth'], 
                            heads = config['de_heads'], head_dim = 64,
                            mlp_mult = config['mlp_mult'],
                            dropout = dropout), # 输出shape = (b, dim, h, w) 
                LayerNorm(decoder_dim),
                ResizeConv2d(input_filters = decoder_dim, output_filters = config['emb_de_dim'], 
                             kernel = config['emb_kernel'], 
                             padding = (config['emb_kernel'] // 2), # (56
                             strides = config['emb_stride'],
                             ratio=config['emb_ratio']), # 输出shape = (b, de_dim, h, w)
            ))
            # 最后一层的输出作为下一层的输入
            decoder_dim = config['emb_de_dim'] # 解码维度
            
        self.decoder_layers = nn.Sequential(*decoder_layers) # layers 是一个 nn.Sequential 容器，包含了多尺度特征提取器的所有层
        
        # 分类头
        self.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 自适应平均池化层，将输入的任意形状的特征图转换为固定形状的特征图,shape = (b, emb_dim, 1, 1)
            Rearrange('... () () -> ...'), # 将输入的特征图的维度从 (b, emb_dim, 1, 1) 转换为 (b, emb_dim)
            nn.Linear(encoder_dim, num_classes) # 线性层，将输入的特征图转换为输出的类别数
        )

    def _fixstn(self, x, theta):
        # 含义是stn网络=>变换后的图像
        grid = F.affine_grid(theta, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform
    
    def forward_stn(self, x):
        # x.shape = (b, c, h, w)
        x = self.encoder_layer1(x)
        x, theta1 = self.stn1(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea1 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn1_output = self._fixstn(x.detach().cpu(), fixthea1) # 可视化的时候用
        
        x = self.encoder_layer2(x)
        x, theta2 = self.stn2(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea2 = torch.from_numpy(np.linalg.inv(np.concatenate((theta2.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn2_output = self._fixstn(self._fixstn(x.detach().cpu(), fixthea2), fixthea1) # 恢复
        
        x = self.encoder_layer3(x)
        out, theta3 = self.stn3(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea3 = torch.from_numpy(np.linalg.inv(np.concatenate((theta3.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cpu() # cuda(1)
        self.stn3_output = self._fixstn(self._fixstn(self._fixstn(out.detach().cpu(), fixthea3), fixthea2), fixthea1) # 恢复
        
        # 通过self.last_norm对最后一层进行归一化处理。这通常是为了稳定训练过程，确保各层的激活在合适的范围内。
        out_norm = self.last_encoder_norm(out)
        Logits = self.to_logits(out) # 分类的头
        return out, out_norm, Logits
    
    def forward_encoder(self, x):
        # x.shape = (b, c, h, w)
        latents = self.encoder_layers(x) # latents.shape = (b, emb_dim, h, w)
        # 通过self.last_norm对最后一层进行归一化处理。这通常是为了稳定训练过程，确保各层的激活在合适的范围内。
        encoder = self.last_encoder_norm(latents)
        Logits = self.to_logits(latents) # 分类的头
        return latents, encoder, Logits
    
    def forward_decoder(self, x):
        return self.decoder_layers(x) # x.shape = (N,3,h,w)
    
    def forward(self, x):
        out, _, _ = self.forward_stn(x)
        # _, encoder, _ = self.forward_encoder(x)
        # decoder = self.forward_decoder(encoder)

        return out

def net(args, pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """ 
    model = STNCvT(args, num_classes=0)
    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model