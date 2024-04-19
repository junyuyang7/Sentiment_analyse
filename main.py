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

args = Config()
gpus = [4,5,7]
os.environ['CUDA_VISIBLE_DEVICE'] = ','.join(map(str, gpus))

def get_data(args: Config, mode: str='test') -> List[pd.DataFrame]:
    if mode == 'test':
        train = pd.read_csv(args.data_args['train_path'])[:100]
        test = pd.read_csv(args.data_args['test_path'])[:100]
    else:
        train = pd.read_csv(args.data_args['train_path'])
        test = pd.read_csv(args.data_args['test_path'])
    return train, test

def main(mode: str):
    # 1.获取模型
    checkpoint_path = args.model_path['checkpoint_path']
    tokenizer_path = args.model_path['tokenizer_path']
    mymodel = MyModel.build(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, args=args)  

    if torch.cuda.device_count()>1:
        print('使用多卡训练: {}'.format(gpus))
        mymodel.model = nn.DataParallel(mymodel.model, device_ids=[4, 5, 6, 7])
    
    mymodel.model = mymodel.model.cuda(device=gpus[0])
    fgm = FGM(mymodel.model)

    # 3.获取数据
    train_all, test = get_data(args, mode=mode)
    train, valid = train_test_split(train_all, test_size=0.2, random_state=42)

    trainset = RoleDataset(train, mymodel, args)
    train_loader = create_dataloader(trainset, args, mode='train')

    validset = RoleDataset(valid, mymodel, args)
    valid_loader = create_dataloader(validset, args, mode='test')

    testset = RoleDataset(test, mymodel, args)
    test_loader = create_dataloader(testset, args, mode='test')

    # 2.获取优化器，损失函数等
    optimizer = AdamW(mymodel.model.parameters(), lr=args.model_args['lr'], weight_decay=args.model_args['weight_decay']) # correct_bias=False,
    total_steps = len(train_loader) * args.model_args['epoch']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = args.model_args['warm_up_ratio']*total_steps,
        num_training_steps=total_steps
        )

    # criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = nn.L1Loss().to(device)
    criterion_reg = nn.MSELoss(reduction="mean").to(args.model_args['device'])
    criterion_cls = nn.CrossEntropyLoss().to(args.model_args['device'])

    # 开始训练
    mymodel.train(train_loader, valid_loader, criterion_reg, criterion_cls, optimizer, scheduler, fgm)
    mymodel.save_pth()

    # 储存最终预测结果
    submit = pd.read_csv('data/submit_example.tsv', sep='\t')
    test_pred = mymodel.predict(test_loader)
    label_preds = []
    for col in args.target_cols:
        preds = test_pred[col]
        label_preds.append(preds)
    print(len(label_preds[0]))
    if mode == test:
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
    isTest = True
    if isTest:
        main(mode='test')
    else:
        main(mode='gogogo')
