import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Q_Network(nn.Module):
    def __init__(self, input_size, act_size, args):
        super(Q_Network, self).__init__()
        self.hidden_size = args.q_net_hidden_size
        self.mlp_in_layer = nn.Linear(input_size, args.q_net_out[0])
        self.mlp_out_layer = nn.Linear(args.q_net_hidden_size, act_size)
        self.GRU_layer = nn.GRUCell(args.q_net_out[0], args.q_net_hidden_size)
        self.ReLU = nn.ReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_in_layer.weight)
        nn.init.xavier_uniform_(self.mlp_out_layer.weight)

    def init_hidden(self, args):
        return self.mlp_in_layer.weight.new(1, args.q_net_hidden_size).zero_()

    def forward(self, obs_a_cat, hidden_last):
        # x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        gru_out = self.GRU_layer(x, hidden_last)
        output = self.mlp_out_layer(gru_out)
        return output, gru_out
