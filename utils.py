import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import math
import pandas as pd
import numpy as np
import re
from typing import *
from config import Config
from tqdm import tqdm

args = Config()

# data_processing.py
def get_labels(df: pd.DataFrame, mode: str='train') -> pd.DataFrame:
    if mode == 'train':
        df['emotions'] = df['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])
        df[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = \
                df['emotions'].values.tolist()
    else:
        df[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] =[0,0,0,0,0,0]
    return df


def get_sort_text(df: pd.DataFrame) -> pd.DataFrame:
    df['script_ids'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['scene_nums'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    df['sentence_nums'] = df['id'].apply(lambda x: int(x.split('_')[3]))
    df.sort_values(by=['script_ids', 'scene_nums', 'sentence_nums'], ascending=True, inplace=True)
    return df


def get_expand_content(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    df_in = df.copy()
    expanded_contents = []
    for index, row in tqdm(df.iterrows()):
        content = row['content']
        start_index = index - 1
        end_index = index + 1
        # 向上扩展
        while 0 < start_index < len(df) - 1 and len(content) < max_len // 2:
            if df_in.loc[start_index, 'content'] != df.loc[start_index+1, 'content']:
                content = df.loc[start_index, 'content'] + content
            start_index -= 1
        
        # 向下扩展
        while 0 < end_index < len(df) - 1 and len(content) < max_len:
            if df.loc[end_index, 'content'] != df.loc[end_index-1, 'content']:
                content += df.loc[end_index, 'content']
            end_index += 1

        if len(content) > max_len:
            content = truncate_content(content, max_len)
        expanded_contents.append(content)
    
    df_in['content'] = expanded_contents
    print('get_expand_content complete')
    return df_in


def get_expand_character(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    df_in = df.copy()
    expanded_contents = []
    for index, row in tqdm(df.iterrows()):
        character = row['character']
        content = row['content']
        start_index = index - 1
        end_index = index + 1
        
        # 向上扩展
        while 0 < start_index < len(df) - 1 and len(content) < max_len // 2:
            if df.loc[start_index, 'content'] != df.loc[start_index+1, 'content'] and \
                character in df.loc[start_index, 'content']:
                content = df.loc[start_index, 'content'] + content
            start_index -= 1

        next_content = ''
        # 向下扩展
        while 0 < end_index < len(df) - 1 and len(next_content) < max_len // 2:
            if df.loc[end_index, 'content'] != df.loc[end_index-1, 'content'] and \
                character in df.loc[end_index, 'content']:
                next_content += df.loc[end_index, 'content']
            end_index += 1

        content += next_content
        if len(content) > max_len:
            content = truncate_content(content, max_len)
        expanded_contents.append(content)
    
    df_in['content'] = expanded_contents
    print('get_expand_character complete')
    return df_in


def truncate_content(content: str, max_len: int) -> str:
    sentences = re.split(r'([，。！？,.!?\s])', content) 
    # sentences = re.split(r'[，。！？,.!?]', content)
    lengths = [len(sentence) for sentence in sentences]
    sumn, n = sum(lengths), len(lengths)
    end_idx = n-1   
    for i in range(n-1, -1, -1):
        if sumn < max_len:
            end_idx = i
            break
        sumn -= lengths[i]
    return ''.join(sentences[:end_idx])


def get_prompt(content:str, character:str, args: Config, mode='prompt1') -> str:
    return args.prompt[mode].format(content, character)


def get_processing_data(df: pd.DataFrame, args: Config, mode:str ='train') -> List[pd.DataFrame]:
    df = get_labels(df, mode)
    df_content = get_expand_content(df, args.model_args['max_len'])
    df_character = get_expand_character(df, args.model_args['max_len'])
    df_content['content'] = df_content.apply(lambda x: get_prompt(x['content'], x['character'], args, mode=args.data_args['prompt_name']), axis=1)
    df_character['content'] = df_character.apply(lambda x: get_prompt(x['content'], x['character'], args, mode=args.data_args['prompt_name']), axis=1)
    return df_content, df_character


# 构建GAT
from torch_geometric.data import Data
from config import Config
import re
import networkx as nx   
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

def get_role(args: Config) -> list:
    train = pd.read_csv(args.data_args['train_fill_path'])
    test = pd.read_csv(args.data_args['test_fill_path'])
    roles_train = train['character'].unique().tolist()
    roles_test = test['character'].unique().tolist()
    roles_train.extend(roles_test)
    roles = list(set(roles_train))
    roles = [role for role in roles if role != '' or pd.notna(role)]
    return roles

def create_graph(text, embeddings):
    num_nodes = len(text)
    roles = get_role(args)
    x = embeddings
    edge_index = None
    for i in range(num_nodes):
        for j in range(i):
            exist_roles1 = [role for role in roles if re.search(role, text[i])]
            exist_roles2 = [role for role in roles if re.search(role, text[j])]
            if set(exist_roles1).intersection(set(exist_roles2)):  # 交集不为空
                if edge_index is None:
                    edge_index = torch.tensor([[i, j]])
                else:
                    # 无向图
                    edge_index = torch.cat([edge_index, torch.tensor([[i, j]])], dim=0)
                    edge_index = torch.cat([edge_index, torch.tensor([[j, i]])], dim=0)
    try:
        edge_index = edge_index.t().contiguous()  
    # 考虑没有边的情况
    except:
        # pass
        # print(len(text))
        # print(text)
        edge_index = torch.tensor([[], []], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.to(args.model_args['device']))

    return data

def draw_graph(edge_index, name=None):
    G = nx.Graph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        G.add_edge(i, j)
    plt.figure(figsize=(8, 8)) # 设置画布的大小
    nx.draw_networkx(G)
    plt.show()

# Loss构建
class JoyOtherLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joy, other):
        loss = torch.clamp(abs(joy * other), min=0)
        return torch.mean(loss)
    

# DDP的构建
import os
import torch.distributed as dist
import torch

def dist_setup(global_rank, world_size):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = '172.31.76.134'
    os.environ['MASTER_PORT'] = '6756'

    # # 根据local_rank来设定当前使用哪块GPU
    # torch.cuda.set_device(global_rank)
    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    dist.init_process_group(backend="nccl", init_method='env://', rank=global_rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
