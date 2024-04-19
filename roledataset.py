import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from config import Config
from transformers import AutoModel, AutoTokenizer, RobertaTokenizer
import ipdb
from model import MyModel
# 定义数据集

class RoleDataset(Dataset):
    def __init__(self, data, model: MyModel, args: Config):
        super().__init__()
        self.target_cols = args.target_cols
        self.data = data
        self.texts=self.data['content'].tolist()
        self.labels=self.data[self.target_cols].to_dict('records')
        self.character = self.data['character'].tolist()
        self.tokenizer: RobertaTokenizer = model.tokenizer
        self.max_len = args.model_args['token_max_len']
        
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        character = self.character[index]

        encoding = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_token_type_ids=True,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_tensors='pt')

        mask_pos, role_pos = None, None
        try:
            vocab_index = self.tokenizer.convert_tokens_to_ids(character)
            role_pos = encoding['input_ids'][0].tolist().index(vocab_index) # 找出该角色的位置
        except:
            print(text, character)
            
        try:
            # 找出所有MASK的位置
            mask_pos = [i for i, token_id in enumerate(encoding['input_ids'][0]) if token_id == self.tokenizer.mask_token_id]
            mask_pos = torch.tensor(mask_pos)
        except:
            print(text, '---并没有设置MASK')

        sample = {
            'character': character,
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'role_pos': role_pos,
            'mask_pos': mask_pos,
        }

        for label_col in self.target_cols:
            # 0-3 映射到 0-1 之间
            sample[label_col] = torch.tensor(label[label_col]/3.0, dtype=torch.float)
        return sample
    
    def __len__(self):
        return len(self.texts)


# 创建dataloader
def create_dataloader(dataset, args: Config, mode='train'):
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=args.model_args['batch_size'], shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=args.model_args['batch_size'], shuffle=shuffle)
    return data_loader

