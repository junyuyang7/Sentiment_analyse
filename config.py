import torch
from dataclasses import dataclass

class Config:
    model_args: dict = {
        'num_labels': 6,
        'model_name': 'models/chinese-roberta-wwm-ext',
        'epoch': 2,
        'weight_decay': 0.005,
        'warmup_proportion': 0.0,
        'batch_size': 32,
        'lr': 2e-5,
        'warm_up_ratio': 0,
        'max_len': 256,
        'token_max_len': 512,
        'model_save_file': "checkpoints/",
        'device': torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
        # 'device': 'cpu',
        'load_model': False,
        'save_model': True,
        'n_class': 1,
        'use_gat': True,
        'use_cls': False,
        'use_mlp': False,
        'use_mask': True,
        'use_role': False,
        'use_fgm': False,
        'use_loss': True
    }
    data_args: dict = {
        'data_path': 'data',
        'data_save_file': 'data_after_process/',
        'train_fill_path': 'data/train_filled.csv',
        'test_fill_path': 'data/test_filled.csv',
        'train_path': 'data_after_process/train_content_prompt2.csv',
        'test_path': 'data_after_process/test_content_prompt2.csv',
        'output': 'result/',
        'prompt_name': 'prompt2'
    }
    prompt = {
        'prompt1': '{}，目标角色是 {}',
        'prompt2': '{}，对于目标角色{}，love的分数是 [MASK]，joy的分数是 [MASK]， fright的分数是 [MASK]，anger的分数是 [MASK]，fear的分数是 [MASK]，sorrow的分数是 [MASK]'
    }
    model_path: dict = {
        'checkpoint_path': 'checkpoints/chinese-roberta-wwm-ext_.pth',
        'tokenizer_path': ''
    }
    weight: dict = {
        'gat_alpha': 0.2,
    }
    target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
    device_args: dict = {
        'gpus': [5,6],
        'use_dp': False,
        'use_ddp': False,
        'use_deepspeed': False,
        'world_size': 4, # gpus几张卡就分成几份

    }
    checkpoint = {
        'init_checkpoint': '', # 路径
    }

