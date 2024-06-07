import os
import random
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor
from models.net import STNCvT, net
from losses import CosLoss, averCosineSimilatiry
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
# device = torch.device('cpu')
device = torch.device('cuda:0' if use_cuda else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--mode', type=str, 
                                  default='train')
    parser.add_argument('--obj', type=str, 
                                 default='PCB2') #类别
    parser.add_argument('--data_path', type=str, 
                                       default='./PCB') #数据集路径
    parser.add_argument('--epochs', type=int, 
                                    default=100, # 600 
                                    help='maximum training epochs') #最大训练轮数
    parser.add_argument('--batch_size', type=int, 
                                        default=4) # 分批 #batch_size
    parser.add_argument('--img_size', type=int, 
                                      default=384) # 384
    parser.add_argument('--lr', type=float, 
                                default=0.0001, 
                                help='learning rate of others in SGD') #学习率
    parser.add_argument('--momentum', type=float, 
                                      default=0.9, 
                                      help='momentum of SGD') #动量
    parser.add_argument('--seed', type=int, 
                                  default=668, 
                                  help='manual seed') #随机种子
    parser.add_argument('--shot', type=int, 
                                  default=2, 
                                  help='shot count') #样本数(k-shot)
    parser.add_argument('--inferences', type=int, 
                                        default=10, 
                                        help='number of rounds per inference') #每轮推断次数
    parser.add_argument('--stn_mode', type=str, 
                                      default='rotation_scale',
                                      help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args = parser.parse_args()
    args.input_channel = 3 #输入通道数  

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str() #日志文件
    args.save_dir = './logs_pcb/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_model_dir = './logs_pcb/' + args.stn_mode + '/' + str(args.shot) + '/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}.txt'.format(str(args.shot),args.obj)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    # 加载模型
    STN = net(args, pretrained=False).to(device) #经过迭代器
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # 第一种
    CKPT_name = f'logs_pcb/rotation_scale/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'
    model_CKPT = torch.load(CKPT_name, map_location=device)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    # 第二种
    # STN.load_state_dict(torch.load('logs_pcb/rotation_scale/1/PCB2/PCB2_1_rotation_scale_model.pt')['STN'], strict=False) # strict=False 时，意味着加载过程中可以忽略一些不匹配的键（keys）和形状（shapes）
    # ENC.load_state_dict(torch.load('logs_pcb/rotation_scale/1/PCB2/PCB2_1_rotation_scale_model.pt')['ENC'], strict=False) 
    # PRED.load_state_dict(torch.load('logs_pcb/rotation_scale/1/PCB2/PCB2_1_rotation_scale_model.pt')['PRED'], strict=False) 

    print(STN)

    # 模型优化器
    STN_optimizer = optim.SGD(STN.parameters(), lr=args.lr, momentum=args.momentum)
    ENC_optimizer = optim.SGD(ENC.parameters(), lr=args.lr, momentum=args.momentum)
    PRED_optimizer = optim.SGD(PRED.parameters(), lr=args.lr, momentum=args.momentum)
    models = [STN, ENC, PRED]   #用于传递到下方的model
    optimizers = [STN_optimizer, ENC_optimizer, PRED_optimizer] #用于传递到下方的optimizer
    init_lrs = [args.lr, args.lr, args.lr]

    # 加载数据集
    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=args.shot, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs) # 加载一批，因为dataset已经加载好了
    # test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.stn_mode))
    epoch_time = AverageMeter()

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizers, init_lrs, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        train(models, epoch, train_loader, optimizers, log)
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
        if epoch % 10 == 0:
            state = {'STN': STN.state_dict(), 'ENC': ENC.state_dict(), 'PRED':PRED.state_dict()}
            torch.save(state, save_name)  # 每10轮保存一次模型
        
    log.close()

def train(models, epoch, train_loader, optimizers, log):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN_optimizer = optimizers[0]
    ENC_optimizer = optimizers[1]
    PRED_optimizer = optimizers[2]

    STN.train() #空间变换网络 --不改变shape
    ENC.train()
    PRED.train()

    total_losses = AverageMeter() # 初始化一个计算平均值的类

    for (query_img, support_img_list, _) in tqdm(train_loader):
        STN_optimizer.zero_grad() #清零梯度，
        ENC_optimizer.zero_grad() #清零梯度，
        PRED_optimizer.zero_grad() #清零梯度，

        # query_img: (类别/批次,Batch,3,224,224) support_img_list: (类别/批次,Batch,2,3,224,224)
        # query_img 含义是待检测的图片，support_img_list是支持集
        query_img = query_img.squeeze(0).to(device) # 将query_img从(类别,Batch,3,224,224)转换为(Batch,3,224,224)
        query_feat = STN(query_img) # 将query_img输入到STN中，得到query_feat
        support_img = support_img_list.squeeze(0).to(device) # 将support_img从(类别/批次,Batch,2,3,224,224)转换为(Batch,2,3,224,224)
        B,K,C,H,W = support_img.shape # B是batch_size，K是支持集的个数，C是通道数，H是高度，W是宽度

        support_img = support_img.view(B * K, C, H, W)
        support_feat = STN(support_img)
        support_feat = support_feat / K # 将support_feat除以K，得到平均值

        _, C, H, W = support_feat.shape # 得到support_feat的形状
        support_feat = support_feat.view(B, K, C, H, W) # 将support_feat的形状转换为(B,K,C,H,W)
        support_feat = torch.sum(support_feat, dim=1) # 求和 (B,K,C,H,W) -> (B,C,H,W)

        z1 = ENC(query_feat) # 将query_feat输入到ENC中，得到z1
        p1 = PRED(z1) # 将z1输入到PRED中，得到p1
        
        z2 = ENC(support_feat) # 将support_feat输入到ENC中，得到z2
        p2 = PRED(z2) # 将z2输入到PRED中，得到p2
            
        total_loss = CosLoss(p1,z2, Mean=True)/2 + CosLoss(p2,z1, Mean=True)/2 # 更好
        # total_loss = averCosineSimilatiry(p1,z2, Mean=True)/2 + averCosineSimilatiry(p2,z1, Mean=True)/2

        total_losses.update(total_loss.item(), query_img.size(0)) # 计算平均值 query_img shape (3,224,224) size(0) = 3
        total_loss.backward() # 一个孪生网络，更新参数，防止梯度爆炸

        STN_optimizer.step()
        ENC_optimizer.step()
        PRED_optimizer.step()

    print_log(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)), log)

def adjust_learning_rate(optimizers, init_lrs, epoch, args):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
