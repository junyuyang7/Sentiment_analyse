import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from typing import *
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn import metrics

from fgm import FGM
from config import Config
from model import MyModel, IQIYModelLite
from roledataset import RoleDataset, create_dataloader
from utils import JoyOtherLoss, dist_setup

# Use DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as Dsample

args = Config()
gpus = args.device_args['gpus']
# os.environ['CUDA_VISIBLE_DEVICE'] = ','.join(map(str, gpus))

def get_data(args: Config, mode: str='test') -> List[pd.DataFrame]:
    if mode == 'test':
        train = pd.read_csv(args.data_args['train_path'])[:100]
        test = pd.read_csv(args.data_args['test_path'])[:100]
    else:
        train = pd.read_csv(args.data_args['train_path'])
        test = pd.read_csv(args.data_args['test_path'])
    return train, test

def main(rank, world_size, mode: str):
    if torch.cuda.device_count()>1 and args.device_args['use_ddp']:
        # 启动分布式训练环境
        dist_setup(rank, world_size)

    # 1.获取模型
    checkpoint_path = args.model_path['checkpoint_path']
    tokenizer_path = args.model_path['tokenizer_path']
    mymodel = MyModel.build(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, args=args)  
    if torch.cuda.device_count()>1 and args.device_args['use_ddp']:
        mymodel.model = mymodel.model.to(rank)
    else:
        mymodel.model = mymodel.model.cuda(device=gpus[0])

    # if torch.cuda.device_count()>1:
    #     print('使用多卡训练: {}'.format(gpus))
    #     if args.device_args['use_dp']:
    #         mymodel.model = nn.parallel.DataParallel(mymodel.model, device_ids=gpus)
    #     if args.device_args['use_ddp']:
    #         mymodel.model = DDP(mymodel.model, device_ids=[rank], find_unused_parameters=True)
        
    fgm = FGM(mymodel.model)

    # 3.获取数据
    train_all, test = get_data(args, mode=mode)
    train, valid = train_test_split(train_all, test_size=0.2, random_state=42)

    trainset = RoleDataset(train, mymodel, args)
    validset = RoleDataset(valid, mymodel, args)
    testset = RoleDataset(test, mymodel, args)

    train_sample, valid_sample, test_sample = None, None, None
    if torch.cuda.device_count()>1 and args.device_args['use_ddp']:
        # 获取进程数
        num_tasks = dist.get_world_size()
        train_sample = Dsample(trainset, num_replicas=num_tasks, rank=dist.get_rank(), shuffle=True)
        valid_sample = Dsample(trainset, num_replicas=num_tasks, rank=dist.get_rank(), shuffle=False)
        test_sample = Dsample(trainset, num_replicas=num_tasks, rank=dist.get_rank(), shuffle=False)

    train_loader = create_dataloader(trainset, args, train_sample, mode='train')
    valid_loader = create_dataloader(validset, args, valid_sample, mode='test')
    test_loader = create_dataloader(testset, args, test_sample, mode='test')

    # 2.获取优化器，损失函数等
    optimizer = AdamW(mymodel.model.parameters(), lr=args.model_args['lr'], weight_decay=args.model_args['weight_decay']) # correct_bias=False,
    total_steps = len(train_loader) * args.model_args['epoch']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = args.model_args['warm_up_ratio']*total_steps,
        num_training_steps=total_steps
        )

    # criterion_reg = nn.BCEWithLogitsLoss().to(gpus[0])
    # criterion = nn.L1Loss().to(device)
    criterion_reg = nn.MSELoss(reduction="mean").to(gpus[0])
    criterion_cls = nn.CrossEntropyLoss().to(gpus[0])
    myLoss = JoyOtherLoss().to(gpus[0])

    # 开始训练
    mymodel.train(rank, train_loader, valid_loader, criterion_reg, criterion_cls, myLoss, optimizer, scheduler, fgm)
    mymodel.save_pth()

    # 储存最终预测结果
    submit = pd.read_csv('data/submit_example.tsv', sep='\t')
    test_pred = mymodel.predict(test_loader)
    label_preds = []
    for col in args.target_cols:
        preds = test_pred[col]
        label_preds.append(preds)
    print(len(label_preds[0]))
    if mode == 'test':
        sub = submit.copy()[:100]
    else:
        sub = submit.copy()
    sub['emotion'] = np.stack(label_preds, axis=1).tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(max(0, i)) for i in x]))
    output_file = args.data_args['output']
    os.makedirs(output_file, exist_ok=True)
    sub.to_csv('{}result_{}.tsv'.format(output_file, args.model_args['model_name'].split('/')[-1]), sep='\t', index=False)
    sub.head()

if __name__ == '__main__':
    isTest = False
    mode = 'test' if isTest else 'gogogo'

    if args.device_args['use_ddp']:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
        world_size = args.device_args['world_size']  # 进程数，要与cuda_visible_devices的数量一致
        torch.multiprocessing.spawn(main, args=(world_size, mode), nprocs=world_size, join=True)
    else:
        rank = -1
        world_size = -1
        main(rank, world_size, mode)
