# Code adapted from https://github.com/DKuan/sc2_QMIX

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.nn import GATConv

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

class DummyGNN(nn.Module):
    def __init__(self, ):
        super(DummyGNN, self).__init__()

    def forward(self, input):
        return input


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
        #x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        gru_out = self.GRU_layer(x, hidden_last)
        output = self.mlp_out_layer(gru_out)
        return output, gru_out
    
class Hyper_Network(nn.Module):
    def __init__(self, shape_state, shape_hyper_net, args):
        super(Hyper_Network, self).__init__()
        self.hyper_net_pars = shape_hyper_net
        self.w1_layer = nn.Linear(shape_state, shape_hyper_net['w1_size'])
        self.w2_layer = nn.Linear(shape_state, shape_hyper_net['w2_size'])
        self.b1_layer = nn.Linear(shape_state, shape_hyper_net['b1_size'])
        self.b2_layer_i = nn.Linear(shape_state, args.shape_hyper_b2_hidden)
        self.b2_layer_h = nn.Linear(args.shape_hyper_b2_hidden, shape_hyper_net['b2_size'])
        self.LReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer_i.weight)
        nn.init.xavier_uniform_(self.b2_layer_h.weight)

    def forward(self, state):
        w1_shape = self.hyper_net_pars['w1_shape']
        w2_shape = self.hyper_net_pars['w2_shape']
        w1 = torch.abs(self.w1_layer(state)).view(-1, w1_shape[0], w1_shape[1])
        w2 = torch.abs(self.w2_layer(state)).view(-1, w2_shape[0], w2_shape[1])
        b1 = self.b1_layer(state).view(-1, 1, self.hyper_net_pars['b1_shape'][0])
        #x = self.LReLU(self.b2_layer_i(state))
        x = self.ReLU(self.b2_layer_i(state))
        b2 = self.b2_layer_h(x).view(-1, 1, self.hyper_net_pars['b2_shape'][0])
        return {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
        
class Mixing_Network(nn.Module):
    def __init__(self, action_size, num_agents, args):
        super(Mixing_Network, self).__init__()
        # action_size * num_agents = the num of Q values
        self.w1_shape = torch.Size((num_agents, args.mix_net_out[0]))
        self.b1_shape = torch.Size((args.mix_net_out[0], ))
        self.w2_shape = torch.Size((args.mix_net_out[0], args.mix_net_out[1]))
        self.b2_shape = torch.Size((args.mix_net_out[1], ))
        self.w1_size = self.w1_shape[0] * self.w1_shape[1]
        self.b1_size = self.b1_shape[0]
        self.w2_size = self.w2_shape[0] * self.w2_shape[1]
        self.b2_size = self.b2_shape[0]
        self.pars = {'w1_shape':self.w1_shape, 'w1_size':self.w1_size, \
                'w2_shape':self.w2_shape, 'w2_size':self.w2_size, \
                'b1_shape':self.b1_shape, 'b1_size':self.b1_size, \
                'b2_shape':self.b2_shape, 'b2_size':self.b2_size, }
        self.LReLU = nn.LeakyReLU(0.001)
        self.ReLU = nn.ReLU()
    
    def forward(self, q_values, hyper_pars):
        # q_values: [B * T, 1, n_sensors]
        x = self.ReLU(torch.bmm(q_values, hyper_pars['w1']) + hyper_pars['b1'])
        output = torch.bmm(x, hyper_pars['w2']) + hyper_pars['b2']
        return output.view(-1)

class IQN(nn.Module):
    def __init__(self):
        super(IQN, self).__init__()


    def forward(self, q_values, state):
        # q_values: [B * T, 1, n_sensors]
        return torch.squeeze(q_values)

class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()


    def forward(self, q_values, state):
        # q_values: [B * T, 1, n_sensors]
        return torch.sum(q_values, dim=2)


class Qmixer(nn.Module):
    def __init__(self, num_agents, shape_state, args):
        super(Qmixer, self).__init__()
        # action_size * num_agents = the num of Q values
        self.w1_shape = torch.Size((num_agents, args.mix_net_out[0]))
        self.b1_shape = torch.Size((args.mix_net_out[0],))
        self.w2_shape = torch.Size((args.mix_net_out[0], args.mix_net_out[1]))
        self.b2_shape = torch.Size((args.mix_net_out[1],))
        self.w1_size = self.w1_shape[0] * self.w1_shape[1]
        self.b1_size = self.b1_shape[0]
        self.w2_size = self.w2_shape[0] * self.w2_shape[1]
        self.b2_size = self.b2_shape[0]
        pars = {'w1_shape':self.w1_shape, 'w1_size':self.w1_size, \
                'w2_shape':self.w2_shape, 'w2_size':self.w2_size, \
                'b1_shape':self.b1_shape, 'b1_size':self.b1_size, \
                'b2_shape':self.b2_shape, 'b2_size':self.b2_size, }
        self.LReLU = nn.LeakyReLU(0.001)
        self.ReLU = nn.ReLU()
        self.hyper_net = Hyper_Network(shape_state, pars, args)

    def forward(self, q_values, state):
        # q_values: [B * T, 1, n_sensors]
        hyper_pars = self.hyper_net(state)
        x = self.ReLU(torch.bmm(q_values, hyper_pars['w1']) + hyper_pars['b1'])
        output = torch.bmm(x, hyper_pars['w2']) + hyper_pars['b2']
        return output.view(-1, 1)