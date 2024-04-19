import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from config import Config
import re
import networkx as nx   
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
import pandas as pd
import numpy as np

from utils import draw_graph

args = Config()


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        # num_features: Alias for num_node_features.
        self.conv1 = GATConv(in_channels, 64, heads=8, dropout=0.1)
        # self.conv1_1 = GATConv(8*8, 8, heads=8, dropout=0.1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(64 * 8, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        # x_copy = x.clone()
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.elu(self.conv1_1(x, edge_index))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        # return x + x_copy  # Residual connection, 避免孤立节点变成全0
        
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    model = GAT(3, 6)

    x = torch.tensor([[1.2,2.2,3.1], [0,1.1,1], [1,2.5,3]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    draw_graph(data.edge_index)
    print(data)
    pred = model(data.x, data.edge_index)
    print(pred.shape)  # [torch.FloatTensor of size (3, 6)]