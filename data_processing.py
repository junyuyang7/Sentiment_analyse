import pandas as pd
import numpy as np
from config import Config
from typing import *
import os

################################################
# 数据扩充的步骤
# 1.先将emotion中的六个情感分为6列
# 2.将剧本按id进行排序(script_ids, scene_nums, sentence_nums)
# 3.进行上下文的拼接，两种方式（按剧本内容拼接、从角色出发进行拼接）
# 4.输入文本的设计（[CLS] or [MASK]）
################################################
from utils import get_processing_data

args = Config()
output_data_file = args.data_args['data_save_file']
prompt = args.data_args['prompt_name']
os.makedirs(output_data_file, exist_ok=True)

def get_data(args: Config) -> List[pd.DataFrame]:
    train = pd.read_csv(args.data_args['train_fill_path'])
    test = pd.read_csv(args.data_args['test_fill_path'])
    return train, test

train, test = get_data(args)


if __name__ == '__main__':
    isTest = True
    if isTest:
        train_content, train_character = get_processing_data(train[:200], args, mode='train')
        test_content, test_character = get_processing_data(test[:100], args, mode='test')
        print(train_content)
        print(test_content)
    else:
        train_content, train_character = get_processing_data(train, args, mode='train')
        test_content, test_character = get_processing_data(test, args, mode='test')

        train_content.to_csv(output_data_file + 'train_content_' + prompt + '.csv', \
            columns=['id', 'content', 'character', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep=',', index=False)
        train_character.to_csv(output_data_file + 'train_character_' + prompt + '.csv', \
            columns=['id', 'content', 'character', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep=',', index=False)
        test_content.to_csv(output_data_file + 'test_content_' + prompt + '.csv', \
            columns=['id', 'content', 'character', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep=',', index=False)
        test_character.to_csv(output_data_file + 'test_character_' + prompt + '.csv', \
            columns=['id', 'content', 'character', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep=',', index=False)