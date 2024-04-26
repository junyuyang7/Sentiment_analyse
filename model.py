import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from gat import GAT
from fgm import FGM
from config import Config
from utils import create_graph, draw_graph, get_role
from tqdm import tqdm
import os
import time
import logging

import torch.distributed as dist
from utils import cleanup

args = Config()
gpus = args.device_args['gpus']
# os.environ['CUDA_VISIBLE_DEVICE'] = ','.join(map(str, gpus))

# 模型构建
# 对给定模型列表中的参数进行Xavier均匀初始化，以提升模型的训练效果和收敛速度。
def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return
        

class IQIYModelLite(nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.device = gpus[0]
        config = AutoConfig.from_pretrained(args.model_args['model_name'])
        config.update({"output_hidden_states": True,
                        "hidden_dropout_prob": 0.0,
                        "layer_norm_eps": 1e-7})
        self.base = AutoModel.from_pretrained(args.model_args['model_name'], config=config)
        dim = 1024 if 'large' in args.model_args['model_name'] else 768
        self.gatnet = GAT(dim, dim)


        # 添加一个门控计算last_hidden_state 中各个token的权重
        self.gate = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
            # F.silu()
        )

        self.out_love = nn.Sequential(nn.Linear(dim, args.model_args['n_class']))
        self.out_joy = nn.Sequential(nn.Linear(dim , args.model_args['n_class']))
        self.out_fright = nn.Sequential(nn.Linear(dim, args.model_args['n_class']))
        self.out_anger = nn.Sequential(nn.Linear(dim, args.model_args['n_class']))
        self.out_fear = nn.Sequential(nn.Linear(dim, args.model_args['n_class']))
        self.out_sorrow = nn.Sequential(nn.Linear(dim, args.model_args['n_class']))

        init_params([self.out_love, self.out_joy, self.out_fright, self.out_anger,
                     self.out_fear,  self.out_sorrow, self.gate])

    def forward(self, input_ids, attention_mask, texts, mask_pos, role_pos):
        num_labels = self.args.model_args['num_labels']
        gat_alpha = self.args.weight['gat_alpha']
        # last_hidden_state：(batch_size, sequence_length, hidden_size)
        # pooled_output: (batch_size, hidden_size)
        output = self.base(input_ids=input_ids,
                                   attention_mask=attention_mask)
        last_hidden_state, pooled_output = output[0], output[1]
        emotions_embedding = [pooled_output] * num_labels

        # CLS embediing
        cls_embedding = last_hidden_state[:, 0, :].squeeze(1)
        
        # MLP embedding
        weights = self.gate(last_hidden_state[:, 1:, :])
        mlp_embedding = torch.sum(weights * last_hidden_state[:, 1:, :], dim=1)
         
        # MASK embedding
        mask_embeddings = []
        for mix in mask_pos.flatten():
            mask_embeddings.append(last_hidden_state[:, mix, :])
            
        # Role embedding
        role_embedding = torch.mean(last_hidden_state[:, role_pos, :], dim=1).squeeze(1)
        
        data = create_graph(texts, pooled_output)
        if self.args.model_args['use_role']:
            emotions_embedding = [role_embedding for _ in range(num_labels)]
            data = create_graph(texts, role_embedding)
            
        if self.args.model_args['use_cls']:
            emotions_embedding = [cls_embedding for _ in range(num_labels)]
            data = create_graph(texts, cls_embedding)
            
        if self.args.model_args['use_mask']:
            emotions_embedding = [msk_emb for msk_emb in mask_embeddings]
            
        if self.args.model_args['use_mlp']:
            emotions_embedding = [mlp_embedding for _ in range(num_labels)]
            data = create_graph(texts, mlp_embedding)
            
        # GAT embedding
        gat_embedding = self.gatnet(data.x, data.edge_index)  # 还是用batch中的所有句子

        if self.args.model_args['use_gat']:
            emotions_embedding = [gat_alpha*gat_embedding + (1-gat_alpha)*emb for emb in emotions_embedding]

        love = self.out_love(emotions_embedding[0])
        joy = self.out_joy(emotions_embedding[1])
        fright = self.out_fright(emotions_embedding[2])
        anger = self.out_anger(emotions_embedding[3])
        fear = self.out_fear(emotions_embedding[4])
        sorrow = self.out_sorrow(emotions_embedding[5])

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
        
        
class MyModel(nn.Module):
    def __init__(self, model: IQIYModelLite, tokenizer: AutoTokenizer, args: Config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = gpus[0]
        self.target_cols = args.target_cols
        self.adv = args.model_args['use_fgm']
        
    @staticmethod
    def build(checkpoint_path: str, tokenizer_path: str, args: Config):
        model = IQIYModelLite(args)
        tokenizer = AutoTokenizer.from_pretrained(args.model_args['model_name'])
        # 需要加入角色名字新词，否则 'h3' 会被分成 'h' 和 '3'
        roles = get_role(args)
        num_added_toks = tokenizer.add_tokens(roles)
        print('加入{}个新词，分别是：{}'.format(num_added_toks, roles))
        # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
        model.base.resize_token_embeddings(len(tokenizer)) 

        if args.model_args['load_model']:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
        args = args
        return MyModel(model, tokenizer, args)
    
    def predict(self, data_loader):
        from collections import defaultdict
        
        test_pred = defaultdict(list)
        self.model.eval()
        self.model.to(self.device)
        for batch in tqdm(data_loader):
            b_input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            texts = batch["texts"]
            mask_pos = batch['mask_pos'].to(self.device)
            role_pos = batch['role_pos'].to(self.device)
            with torch.no_grad():
                logists = self.model(input_ids=b_input_ids, 
                                     attention_mask=attention_mask, 
                                     texts=texts,
                                     mask_pos=mask_pos,
                                     role_pos=role_pos,)
                for col in self.target_cols:
                    out2 = logists[col].squeeze(1)*3.0  # 这个3要乘回去
                    out2 = torch.clamp(out2, min=0, max=3)
                    test_pred[col].extend(out2.cpu().numpy().tolist())
                    
        return test_pred
    
    def test(self, valid_loader, criterion_reg, criterion_cls, myloss, epoch):
        self.model.train()
        tic_train = time.time()
        losses = []
        log_steps = 100
        for step, batch in enumerate(tqdm(valid_loader)):
            b_input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            texts = batch["texts"]
            mask_pos = batch['mask_pos'].to(self.device)
            role_pos = batch['role_pos'].to(self.device)
            with torch.no_grad():
                logists = self.model(input_ids=b_input_ids, 
                                        attention_mask=attention_mask, 
                                        texts=texts,
                                        mask_pos=mask_pos,
                                        role_pos=role_pos,)

            # Reg_loss
            loss_love_reg = criterion_reg(logists['love'], batch['love'].view(-1, 1).to(self.device))
            loss_joy_reg = criterion_reg(logists['joy'], batch['joy'].view(-1, 1).to(self.device))
            loss_fright_reg = criterion_reg(logists['fright'], batch['fright'].view(-1, 1).to(self.device))
            loss_anger_reg = criterion_reg(logists['anger'], batch['anger'].view(-1, 1).to(self.device))
            loss_fear_reg = criterion_reg(logists['fear'], batch['fear'].view(-1, 1).to(self.device))
            loss_sorrow_reg = criterion_reg(logists['sorrow'], batch['sorrow'].view(-1, 1).to(self.device))
            reg_loss = loss_love_reg + loss_joy_reg + loss_fright_reg + loss_anger_reg + loss_fear_reg + loss_sorrow_reg

            if self.args.model_args['use_loss']:
                    loss_anger_joy = myloss(logists['joy'], logists['anger'])
                    loss_sorrow_joy = myloss(logists['joy'], logists['sorrow'])
                    reg_loss = reg_loss + loss_anger_joy + loss_sorrow_joy

            loss = reg_loss
            losses.append(loss.item())
            
            if step % log_steps == 0:
                print("epoch: {}, step: {}, loss: {:.15f}, speed: {:.2f} step/s".format(epoch, step, np.mean(losses), step / (time.time() - tic_train)))
                # logging.info("epoch: {}, step: {}, loss: {:.15f}, speed: {:.2f} step/s".format(epoch, step, np.mean(losses), step / (time.time() - tic_train)))
    
    def train(self, rank, train_loader, valid_loader, criterion_reg, criterion_cls, myloss, optimizer, scheduler, fgm: FGM):
        self.model.train()
        global_step = 0
        tic_train = time.time()
        log_steps = 100
        
        if self.args.device_args['use_ddp']:
            # ddp_loss是为了收集不同进程返回的loss，
            # 以便我们记录并展示所有进程的平均loss，来看loss的下降趋势
            ddp_loss = torch.zeros(1).to(rank)

        for epoch in range(self.args.model_args['epoch']):
            losses = []
            if self.adv: # 加入对抗训练
                losses_adv = [] 
            for step, batch in enumerate(tqdm(train_loader)):
                b_input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                texts = batch["texts"]
                mask_pos = batch['mask_pos'].to(self.device)
                role_pos = batch['role_pos'].to(self.device)
                
                logists = self.model(input_ids=b_input_ids, 
                                     attention_mask=attention_mask, 
                                     texts=texts,
                                     mask_pos=mask_pos,
                                     role_pos=role_pos,)

                # Reg_loss
                loss_love_reg = criterion_reg(logists['love'], batch['love'].view(-1, 1).to(self.device))
                loss_joy_reg = criterion_reg(logists['joy'], batch['joy'].view(-1, 1).to(self.device))
                loss_fright_reg = criterion_reg(logists['fright'], batch['fright'].view(-1, 1).to(self.device))
                loss_anger_reg = criterion_reg(logists['anger'], batch['anger'].view(-1, 1).to(self.device))
                loss_fear_reg = criterion_reg(logists['fear'], batch['fear'].view(-1, 1).to(self.device))
                loss_sorrow_reg = criterion_reg(logists['sorrow'], batch['sorrow'].view(-1, 1).to(self.device))
                reg_loss = loss_love_reg + loss_joy_reg + loss_fright_reg + loss_anger_reg + loss_fear_reg + loss_sorrow_reg

                if self.args.model_args['use_loss']:
                    loss_anger_joy = myloss(logists['joy'], logists['anger'])
                    loss_sorrow_joy = myloss(logists['joy'], logists['sorrow'])
                    reg_loss = reg_loss + loss_anger_joy + loss_sorrow_joy

                loss = reg_loss
                losses.append(loss.item())
                loss.backward()

                # 2. 加入对抗训练
                if self.adv:
                    loss_adv = 0
                    emb_name = 'word_embeddings'

                    fgm.attack(epsilon=0.3, emb_name=emb_name) # 只攻击 word_embeddings
                    outputs = self.model(input_ids=b_input_ids, 
                                        attention_mask=attention_mask, 
                                        texts=texts,
                                        mask_pos=mask_pos,
                                        role_pos=role_pos,)
                    loss_adv = torch.sqrt(criterion_reg(outputs, batch["labels"].to(self.device)))
                    losses_adv.append(loss_adv.item())
                    loss_adv.backward()
                    fgm.restore(emb_name=emb_name) # 恢复Embedding的参数

                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if self.args.device_args['use_ddp']:
                    ddp_loss[0] += loss.item()

                global_step += 1
                if global_step % log_steps == 0:
                    batch_loss = -1
                    if self.args.device_args['use_ddp']:
                        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                        size = dist.get_world_size()
                        batch_loss = ddp_loss[0].item() / (10 * size)  # 求平均
                        ddp_loss = torch.zeros(1).to(rank)

                        print("global step {}, epoch: {}, batch: {}, loss: {:.6f}, batch_loss: {:.6f}, speed: {:.2f} step/s, lr: {:.10f}"
                            .format(global_step, epoch, step, np.mean(losses), batch_loss, global_step / (time.time() - tic_train), 
                                float(scheduler.get_last_lr()[0])))
                    else:
                        print("global step {}, epoch: {}, batch: {}, loss: {:.6f}, speed: {:.2f} step/s, lr: {:.10f}"
                            .format(global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train), 
                                float(scheduler.get_last_lr()[0])))

                    # logging.info("global step {}, epoch: {}, batch: {}, loss: {:.15f}, speed: {:.2f} step/s, lr: {:.10f}"
                        # .format(global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train), 
                        #     float(scheduler.get_last_lr()[0])))

            self.test(valid_loader, criterion_reg, criterion_cls, myloss, epoch)
            if self.args.device_args['use_ddp']:
                # 等待所有进程一轮训练结束，类似于join
                dist.barrier()

        if self.args.device_args['use_ddp']:
            # 训练结束后关闭分布式环境
            cleanup()

    
    def save_pth(self):
        output_file = self.args.model_path['checkpoint_path'].rsplit('/', 1)[0]
        model_name = self.args.model_args['model_name'].split('/')[-1]
        
        use_gat = 'Ugat_' if self.args.model_args['use_gat'] else ''
        use_cls = 'Ucls_' if self.args.model_args['use_cls'] else ''
        use_mlp = 'Umlp_' if self.args.model_args['use_mlp'] else ''
        use_mask = 'Umask_' if self.args.model_args['use_mask'] else ''
        use_role = 'Urole_' if self.args.model_args['use_role'] else ''

        os.makedirs(output_file, exist_ok=True)
        torch.save(self.model.state_dict(), f'{output_file}/{model_name}_{use_gat}{use_cls}{use_mlp}{use_mask}{use_role}.pth')
            