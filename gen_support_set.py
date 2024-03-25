import os
import random
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test, FSAD_Dataset_support


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='PCB1')
    parser.add_argument('--data_path', type=str, default='./PCB')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of others in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=5, help='number of rounds per inference')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                        help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args = parser.parse_args()
    args.input_channel = 3
    args.batch_size = args.inferences # 只是个说明

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # {args.obj}/{args.shot}_{args.inferences}.pt'
    args.save_model_dir = './support_set_pcb/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    save_name = os.path.join(args.save_model_dir, '{}_{}.pt'.format(args.shot, args.inferences)) 

    # 加载数据集
    print('Loading Datasets')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    support_dataset = FSAD_Dataset_support(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=args.shot, batch=args.inferences) # 10个为1批
    train_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs) # 每次取1批

    for support_img_list in train_loader:
        torch.save(support_img_list.squeeze(0), save_name)
        fixed_fewshot_list = torch.load(f'./support_set_pcb/{args.obj}/{args.shot}_{args.inferences}.pt') # 这是权重
        break

main()