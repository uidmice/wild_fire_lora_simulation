# Code adapted from https://github.com/DKuan/sc2_QMIX

import os
import sys, itertools

import time
import torch
import random
import numpy as np
import dgl
import torch.nn.functional as F
import torch.nn as nn

from replay_buffer import ReplayBuffer
from .baselines import Base_Agent
from .mixers import Qmixer, VDN, IQN
from .graphmix import GraphMixer
from .GNNs.gnn import DummyGNN
from .q_network import Q_Network

class QMIX_Agent(Base_Agent):
    def __init__(self, shape_obs, shape_state, num_agents, num_actions_set, args):
        super(QMIX_Agent, self).__init__()
        self.args = args
        self.epsilon = args.epsilon 
        self.shape_obs = shape_obs
        self.shape_state = shape_state
        self.num_agents = num_agents
        self.num_actions_set = num_actions_set
        self.last_cnt4update = 0
        self.last_epi_cnt = 0
        self.memory = ReplayBuffer(args.memory_size)
        self.learned_cnt = 0

    def init_trainers(self):
        # self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
        # if self.args.run == 'GNNQmix':
        #     self.shape_emb = self.args.gnn_out_dim
        #     edges = list(itertools.product(range(self.num_agents), range(self.num_agents)))
        #     u, v = torch.tensor([e[0] for e in edges]), torch.tensor([e[1] for e in edges])
        #     g = dgl.add_self_loop(dgl.graph((u, v))).to(self.args.device)
        #     self.embedding_tar = GNN(self.shape_obs + max(self.num_actions_set), self.args.gnn_hidden_dim, self.shape_emb, g).to(self.args.device)
        #     self.embedding_cur = GNN(self.shape_obs + max(self.num_actions_set), self.args.gnn_hidden_dim, self.shape_emb, g).to(self.args.device)
        #
        # else:
        self.shape_emb = self.shape_obs + max(self.num_actions_set)
        self.embedding_tar = DummyGNN().to(self.args.device)
        self.embedding_cur = DummyGNN().to(self.args.device)

        self.embedding_tar.load_state_dict(self.embedding_cur.state_dict())
        self.q_net_tar = Q_Network(self.shape_emb, max(self.num_actions_set), self.args).to(self.args.device)
        self.q_net_cur = Q_Network(self.shape_emb, max(self.num_actions_set), self.args).to(self.args.device)
        self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par

        if self.args.mixer == 'QMIX':
            self.mixer_tar = Qmixer(self.num_agents, self.shape_state, self.args).to(self.args.device)
            self.mixer_cur = Qmixer(self.num_agents, self.shape_state, self.args).to(self.args.device)
        elif self.args.mixer == 'VDN':
            self.mixer_tar = VDN().to(self.args.device)
            self.mixer_cur = VDN().to(self.args.device)
        elif self.args.mixer == 'GraphMix':
            self.mixer_tar = GraphMixer(self.num_agents, self.shape_state, self.shape_obs, self.args)
            self.mixer_cur = GraphMixer(self.num_agents, self.shape_state, self.shape_obs, self.args)
        if self.args.mixer == 'NONE':
            self.mixer_tar = IQN().to(self.args.device)
            self.mixer_cur = IQN().to(self.args.device)

        self.mixer_tar.load_state_dict(self.mixer_cur.state_dict())
        self.optimizer = torch.optim.RMSprop([{'params':self.q_net_cur.parameters()},
                                              {'params':self.mixer_cur.parameters()},
                                              {'params':self.embedding_cur.parameters()}
            ], lr=self.args.lr)
    
    def enjoy_trainers(self, args):
        self.q_net_cur = torch.load(args.old_model_name+'q_net.pkl', map_location=args.device)
        self.mixer_cur =  torch.load(args.old_model_name+'hyper_net.pkl', map_location=args.device)

    def select_actions(self, avail_actions, obs, actions_last, hidden_last, args, eval_flag=False, fb=None, burning=None):
        """
        Note:epsilon-greedy to choose the action
        """
        """ step1: get the q_values """
        obs_a = torch.from_numpy(np.hstack([obs, actions_last])).to(args.device, dtype=torch.float).unsqueeze(0)

        emb = self.embedding_cur(obs_a).reshape(-1,  self.shape_emb)

        q_values, hidden = self.q_net_cur(emb, torch.from_numpy(hidden_last).to(args.device, dtype=torch.float))

        """ step2: mask the q_values"""
        mask = torch.from_numpy(avail_actions).to(args.device) # mask the actions
        q_values[mask==0] = float('-inf')

        """ choose action by e-greedy """
        avail_act_idxs = [list(np.where(avail_actions[idx]==1)[0]) for idx in range(self.num_agents)]

        avail_actions_random = torch.tensor([random.sample(avail_act_idxs[i], 1) \
            for i in range(self.num_agents)], device=args.device) # all random actions
        avail_actions_random = avail_actions_random.reshape(-1)


        max_actions = torch.max(q_values, dim=1)[1] # all max actions

        epsilons_choice = torch.rand(max_actions.shape) < self.epsilon # e-greedy choose the idx

        max_actions[epsilons_choice] = avail_actions_random[epsilons_choice] if eval_flag == False else \
            max_actions[epsilons_choice]# exchange the data

        return max_actions.detach().cpu().numpy(), hidden.detach().cpu().numpy()

    def cal_totq_values(self, batch_data, args):
        """step1: split the batch data and change the numpy data to tensor data """
        obs_and_u_last_n, state_n, u_n, new_avail_act_n, \
            obs_new_n, state_new_n, r_n, done_n, _ =  batch_data # obs_n obs_numpy
        obs_and_u_last_t_b = torch.from_numpy(obs_and_u_last_n).to(args.device, dtype=torch.float) # obs_tensor_batch 
        state_t_b = torch.from_numpy(state_n).to(args.device, dtype=torch.float) 
        u_t_b = torch.from_numpy(u_n).to(args.device, dtype=torch.long)
        new_obs_and_u_t_b = torch.from_numpy(obs_new_n).to(args.device, dtype=torch.float)
        new_avail_act_t_b = (torch.from_numpy(new_avail_act_n)>0).to(args.device)
        state_new_t_b = torch.from_numpy(state_new_n).to(args.device, dtype=torch.float)
        r_t_b = torch.sum(torch.from_numpy(r_n).to(args.device, dtype=torch.float) , dim=-1) - 1
        done_t_b = torch.from_numpy(1-done_n).to(args.device, dtype=torch.float) # be careful for this action
        max_episode_len = state_new_n[0].shape[0]

        """step2: cal the totq_values """
        q_cur = None # record the low level out values
        q_tar = None

        # cal the q_cur and q_tar
        q_net_input_size = self.shape_obs + max(self.num_actions_set)
        hidden_cur = torch.zeros((args.batch_size*self.num_agents, args.q_net_hidden_size), device=args.device)
        hidden_tar = torch.zeros((args.batch_size*self.num_agents, args.q_net_hidden_size), device=args.device)
        hidden_states_cur = []
        hidden_states_tar = []
        for episode_step in range(max_episode_len):
            # obs_and_u_last_t_b --> [B, T, N, n_actions+obs_dim]
            # input --> [B, N, nkkk]
            input1 = torch.index_select(obs_and_u_last_t_b, 1, torch.tensor([episode_step], device=args.device)).view(-1, self.num_agents, q_net_input_size)
            input2 = torch.index_select(new_obs_and_u_t_b, 1, torch.tensor([episode_step], device=args.device)).view(-1, self.num_agents, q_net_input_size)

            input1 = self.embedding_cur(input1).reshape(-1,  self.shape_emb)
            input2 = self.embedding_tar(input2).reshape(-1,  self.shape_emb)
            q_values_cur, hidden_cur = self.q_net_cur(input1, hidden_cur)
            q_values_tar, hidden_tar = self.q_net_tar(input2, hidden_tar)
            hidden_states_cur.append(hidden_cur)
            hidden_states_tar.append(hidden_tar)
            if episode_step == 0:
                q_cur = [q_values_cur.view(args.batch_size, self.num_agents, -1)]
                q_tar = [q_values_tar.view(args.batch_size, self.num_agents, -1)]
            else:
                q_cur.append(q_values_cur.view(args.batch_size, self.num_agents, -1))
                q_tar.append(q_values_tar.view(args.batch_size, self.num_agents, -1))

        hidden_states_tar = torch.stack(hidden_states_tar, dim=1)
        hidden_states_cur = torch.stack(hidden_states_cur, dim=1)
        # B, T, N, hidden_dim

        q_cur_per_action = torch.stack(q_cur, dim=1)
        q_cur = torch.gather(q_cur_per_action, -1, torch.transpose(u_t_b, -1, -2))
        q_cur = torch.squeeze(q_cur).view(-1, 1, self.num_agents)

        q_tar_per_action = torch.stack(q_tar, dim=1)

        q_tar_per_action[~new_avail_act_t_b] = float('-inf')

        if args.double_q:
            q1 = q_cur_per_action.clone().detach()
            q1[~new_avail_act_t_b] = float('-inf')
            q1_tar = q1.max(dim=-1, keepdim=True)[1]
            q_tar = torch.gather(q_tar_per_action, -1, q1_tar).view(-1, 1, self.num_agents)
            # if torch.isinf(q_tar).any():
            #     idx = torch.isinf(q_tar.view(args.batch_size, -1, self.num_agents)).nonzero(as_tuple=True)
            #     print(f'Got inf number at {torch.isinf(q_tar.view(args.batch_size, -1, self.num_agents)).nonzero(as_tuple=False)}')
            #     print(f'Corresponding action mask is {~new_avail_act_t_b[idx]}')
            #     print(f'Corresponding numpy mask is {new_avail_act_n[idx]}')
            #     raise ValueError

        else:
            q_tar = torch.max(q_tar_per_action, dim=-1)[0].detach().view(-1, 1, self.num_agents)
        # [B * T, 1, N]


        """step3 cal the qtot_cur and qtot_tar by hyper_network"""
        if self.args.mixer != 'GraphMix':
            qtot_cur = self.mixer_cur(q_cur, state_t_b.view(args.batch_size*max_episode_len, -1))
            qtot_tar = self.mixer_tar(q_tar, state_new_t_b.view(args.batch_size*max_episode_len, -1) )
            local_td_error = None
        else:
            # [B * T, -1]
            qtot_cur, local_rewards = self.mixer_cur(q_cur, state_t_b.view(args.batch_size*max_episode_len, -1),
                                                  hidden_states=hidden_states_cur,
                                                  team_rewards=r_t_b.view(args.batch_size*max_episode_len, -1),
                                                  alive_agents=done_t_b.int())
            qtot_tar = self.mixer_cur(q_tar, state_new_t_b.view(args.batch_size*max_episode_len, -1),
                                                  hidden_states=hidden_states_tar,
                                                  alive_agents=done_t_b.int())[0]
            local_targets = local_rewards + args.gamma * done_t_b.view(args.batch_size*max_episode_len,-1) * q_tar.view(args.batch_size*max_episode_len,-1).detach()
            local_td_error = (q_cur.view(args.batch_size*max_episode_len,-1).clone() - local_targets)

        qtot_tar = r_t_b.view(args.batch_size*max_episode_len, -1) + args.gamma * qtot_tar * done_t_b.view(args.batch_size*max_episode_len,-1)

        return qtot_cur, qtot_tar, max_episode_len, local_td_error
        
    def learn(self, epi_cnt, args, logdir=None):
        print(epi_cnt)
        if epi_cnt < args.learning_start_episode:
            return
        self.last_epi_cnt = epi_cnt

        if self.epsilon > 0.01 :
            self.epsilon -= args.anneal_par
        if epi_cnt % args.learning_fre != 0:
            return

        self.learned_cnt += 1
        print('Start to learn! ' + str(self.learned_cnt))

        """ step1: get the batch data from the memory and change to tensor"""
        batch_data, num_diff_lens = self.memory.sample(args.batch_size) # obs_n obs_numpy
        # [B * T, 1], [B * T, N]
        q, q_, T, local_td_error= self.cal_totq_values(batch_data, args)

        """ step2: cal the loss by bellman equation """
        q = q.view(args.batch_size, T, -1)
        q_ = q_.view(args.batch_size, T, -1)


        # delete the loss created by 0_padding data
        for batch_cnt in range(args.batch_size):
            if num_diff_lens[batch_cnt]:
                idx = -num_diff_lens[batch_cnt]
            else:
                idx = T
            q_cur = q[batch_cnt][:idx] if batch_cnt == 0 else \
                torch.cat((q_cur, q[batch_cnt][:idx]), dim=0)
            q_tar = q_[batch_cnt][:idx] if batch_cnt == 0 else \
                torch.cat((q_tar, q_[batch_cnt][:idx]), dim=0)


        td_error = q_cur  - q_tar.detach()
        loss = (td_error ** 2).mean()

        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError('td_error is infinite')

        if args.mixer == 'GraphMix':
            local_td_error = local_td_error.view(args.batch_size, T, -1)
            # delete the loss created by 0_padding data
            for batch_cnt in range(args.batch_size):
                if num_diff_lens[batch_cnt]:
                    idx = -num_diff_lens[batch_cnt]
                else:
                    idx = T
                local_error = local_td_error[batch_cnt][:idx] if batch_cnt == 0 else \
                    torch.cat((local_error, local_td_error[batch_cnt][:idx]), dim=0)
            local_loss = (local_error**2).mean()
            loss += args.lambda_local * local_loss

            if torch.isinf(loss) or torch.isnan(loss):
                raise ValueError('GraphMIX: local_error is infinite')

        """ step3: loss backward """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.embedding_cur.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.q_net_cur.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.mixer_cur.parameters(), args.max_grad_norm)

        self.optimizer.step()

        """ step4: update the tar and cur network """
        if self.learned_cnt > self.last_cnt4update and \
            (self.learned_cnt - self.last_cnt4update)%args.tar_net_update_fre == 0:
            self.last_cnt4update = self.learned_cnt
            self.embedding_tar.load_state_dict(self.embedding_cur.state_dict())
            self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par
            self.mixer_tar.load_state_dict(self.mixer_cur.state_dict())
            print('Update')

        """ step6: save the model """
        if self.learned_cnt >= args.start_save_model and (self.learned_cnt - args.start_save_model) % args.fre4save_model == 0:
            if not logdir:
                logdir = 'models'
            else:
                if not os.path.exists(logdir):
                    os.mkdir(logdir)
                logdir = os.path.join(logdir, 'models')
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            torch.save(self.q_net_tar, os.path.join(logdir, f'q_net_{self.learned_cnt}.pkl'))
            torch.save(self.embedding_tar, os.path.join(logdir, f'emb_net_{self.learned_cnt}.pkl'))
            torch.save(self.mixer_tar, os.path.join(logdir, f'hyper_net_{self.learned_cnt}.pkl'))
            print('store!')

        return loss
