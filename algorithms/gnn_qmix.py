# Code adapted from https://github.com/DKuan/sc2_QMIX

import os
import sys

import time
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from replay_buffer import ReplayBuffer
from .qmix import QMIX_Agent
from .models import Q_Network, Qmixer, VDN, IQN

class GNNQMIX_Agent(QMIX_Agent):
    def __init__(self, shape_obs, shape_state, num_agents, num_actions_set, args):
        super(GNNQMIX_Agent, self).__init__( shape_obs, shape_state, num_agents, num_actions_set, args)

    def init_trainers(self):
        # self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
        super(GNNQMIX_Agent, self).init_trainers()
        self.q_net_tar = Q_Network(self.shape_obs, max(self.num_actions_set), self.args).to(self.args.device)
        self.q_net_cur = Q_Network(self.shape_obs, max(self.num_actions_set), self.args).to(self.args.device)
        self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par

        if self.args.mixer == 'QMIX':
            self.mixer_tar = Qmixer(self.num_agents, self.shape_state, self.args).to(self.args.device)
            self.mixer_cur = Qmixer(self.num_agents, self.shape_state, self.args).to(self.args.device)
        elif self.args.mixer == 'VDN':
            self.mixer_tar = VDN().to(self.args.device)
            self.mixer_cur = VDN().to(self.args.device)
        if self.args.mixer == 'NONE':
            self.mixer_tar = IQN().to(self.args.device)
            self.mixer_cur = IQN().to(self.args.device)

        self.mixer_tar.load_state_dict(self.mixer_cur.state_dict())
        self.optimizer = torch.optim.RMSprop([{'params':self.q_net_cur.parameters()},
                                                {'params':self.mixer_cur.parameters()},
            ], lr=self.args.lr)
    
    def enjoy_trainers(self, args):
        # self.mixing_net = Mixing_Network(max(self.num_actions_set), self.num_agents, args).to(args.device)
        self.q_net_cur = torch.load(args.old_model_name+'q_net.pkl', map_location=args.device)
        # self.hyper_net_cur = torch.load(args.old_model_name+'hyper_net.pkl', map_location=args.device)
        self.mixer_cur =  torch.load(args.old_model_name+'hyper_net.pkl', map_location=args.device)

    def select_actions(self, avail_actions, obs, actions_last, hidden_last, args, eval_flag=False, fb=None):
        """
        Note:epsilon-greedy to choose the action
        """
        """ step1: get the q_values """

        q_values, hidden = self.q_net_cur(torch.from_numpy( \
            np.hstack([obs, actions_last])).to(args.device, dtype=torch.float), \
            torch.from_numpy(hidden_last).to(args.device, dtype=torch.float))

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
        new_avail_act_t_b = torch.from_numpy(new_avail_act_n).to(args.device, dtype=torch.long)
        state_new_t_b = torch.from_numpy(state_new_n).to(args.device, dtype=torch.float) 
        r_t_b = torch.from_numpy(r_n).to(args.device, dtype=torch.float) 
        done_t_b = torch.from_numpy(1-done_n).to(args.device, dtype=torch.float) # be careful for this action
        max_episode_len = state_new_n[0].shape[0]

        # print("MAx len: " + str(max_episode_len))

        """step2: cal the totq_values """
        q_cur = None # record the low level out values
        q_tar = None
        step_cnt = 0

        # cal the q_cur and q_tar
        q_net_input_size = self.shape_obs + max(self.num_actions_set)
        hidden_cur = torch.zeros((args.batch_size*self.num_agents, args.q_net_hidden_size), device=args.device)
        hidden_tar = torch.zeros((args.batch_size*self.num_agents, args.q_net_hidden_size), device=args.device)
        for episode_step in range(max_episode_len):
            input1 = torch.index_select(obs_and_u_last_t_b, 1, torch.tensor([episode_step], device=args.device)).reshape(-1, q_net_input_size)
            input2 = torch.index_select(new_obs_and_u_t_b, 1, torch.tensor([episode_step], device=args.device)).reshape(-1, q_net_input_size)
            q_values_cur, hidden_cur = self.q_net_cur(input1, hidden_cur)
            q_values_tar, hidden_tar = self.q_net_tar(input2, hidden_tar)
            if episode_step == 0:
                q_cur = [q_values_cur.view(args.batch_size, self.num_agents, -1)]
                q_tar = [q_values_tar.view(args.batch_size, self.num_agents, -1)]
            else:
                q_cur.append(q_values_cur.view(args.batch_size, self.num_agents, -1))
                q_tar.append(q_values_tar.view(args.batch_size, self.num_agents, -1))

        q_cur = torch.stack(q_cur, dim=1)
        q_cur = torch.gather(q_cur, -1, torch.transpose(u_t_b, -1, -2))
        q_cur = torch.squeeze(q_cur).view(-1, 1, self.num_agents)

        q_tar = torch.stack(q_tar, dim=1)

        q_tar[~new_avail_act_t_b] = float('-inf')
        q_tar = torch.max(q_tar, dim=-1)[0].detach().view(-1, 1, self.num_agents)
        # [B * T, 1, N]


        """step3 cal the qtot_cur and qtot_tar by hyper_network"""
        # qtot_cur = self.mixing_net(q_cur, self.hyper_net_cur(state_t_b.view(args.batch_size*max_episode_len, -1))).view(-1, 1)
        # qtot_tar = self.mixing_net( q_tar, self.hyper_net_tar(state_new_t_b.view(args.batch_size*max_episode_len, -1)) ).view(-1, 1) # the net is no par
        qtot_cur = self.mixer_cur(q_cur, state_t_b.view(args.batch_size*max_episode_len, -1))
        qtot_tar = self.mixer_tar(q_tar, state_new_t_b.view(args.batch_size*max_episode_len, -1) )
        qtot_tar = r_t_b.view(args.batch_size*max_episode_len, -1) + args.gamma * qtot_tar * done_t_b.view(args.batch_size*max_episode_len,-1)


        return qtot_cur, qtot_tar, max_episode_len
        
    def learn(self, epi_cnt, args, logdir=None):
        loss = 0.0
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
        q, q_, T = self.cal_totq_values(batch_data, args)

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

        # loss = self.mse_loss(q_tar.detach(), q_cur)
        td_error = q_cur  - q_tar.detach()
        loss = (td_error ** 2).mean()
        """ step3: loss backward """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net_cur.parameters(), args.max_grad_norm)
        # nn.utils.clip_grad_norm_(self.hyper_net_cur.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.mixer_cur.hyper_net.parameters(), args.max_grad_norm)

        self.optimizer.step()

        """ step4: update the tar and cur network """
        if epi_cnt > args.learning_start_episode and \
            epi_cnt > self.last_cnt4update and \
            (epi_cnt - self.last_cnt4update)%args.tar_net_update_fre == 0:
            self.last_cnt4update = epi_cnt
            # self.hyper_net_tar.load_state_dict(self.hyper_net_cur.state_dict()) # update the tar net par
            self.q_net_tar.load_state_dict(self.q_net_cur.state_dict()) # update the tar net par
            self.mixer_tar.load_state_dict(self.mixer_cur.state_dict())

        """ step6: save the model """
        if self.learned_cnt > args.start_save_model and self.learned_cnt % args.fre4save_model == 0:
            if not logdir:
                logdir = 'models'
            else:
                logdir = os.path.join(logdir, 'models')
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            torch.save(self.q_net_tar, os.path.join(logdir, f'q_net_{self.learned_cnt}.pkl'))
            # torch.save(self.hyper_net_tar, os.path.join(logdir, f'hyper_net_{self.learned_cnt}.pkl'))
            torch.save(self.mixer_tar, os.path.join(logdir, f'hyper_net_{self.learned_cnt}.pkl'))

        return loss
