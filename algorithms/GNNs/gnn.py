import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.nn import GATConv


class DummyGNN(nn.Module):
    def __init__(self, ):
        super(DummyGNN, self).__init__()

    def forward(self, input):
        return input


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, g):
        super(GNN, self).__init__()
        self.node_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.g = g

        self.conv1 = GATConv(input_dim, hidden_dim//2, 2)
        self.conv2 = GATConv(hidden_dim, hidden_dim, 1)
        self.conv3 = GATConv(hidden_dim, out_dim, 1)

    def forward(self, input):
        # input: [B, N, x ]
        for i in range(input.size(0)):
            h = self.conv1(self.g, input[i].clone()).flatten(1, 2)
            h = F.relu(self.conv2(self.g, h)).squeeze()
            h = self.conv3(self.g, h).squeeze()
            if i == 0:
                x = [h]
            else:
                x.append(h)
        return torch.stack(x, dim=0)
