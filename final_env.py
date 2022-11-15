import math
import pickle, datetime, json

import torch

from config import *

env_init()

from Region import Region
from framework.LoRaCommunication import LoRaCommunication
from framework.GRASS import SOURCE_NAME
from framework.Environment import Environment
from framework.utils import *
import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

# from pettingzoo.utils import parallel_to_aec

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import random

from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple, MultiBinary

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

FUTURE_STEPS = 5
NUM_CHANNELS = 2

COMM_PENALTY = -20
PRED_PENALTY = -10
CORRECTION_REWARD = 10
FUTILE_COMM = -10

WINDVS_LEVELS = 4
WINDTH_LEVELS = 8

AGENT_STATE = {
    'PREDICTION': FUTURE_STEPS + 2,
    'BURNING': 1,
    'WIND': 2,
    'LOCATION': 2,
    'CHANNEL': NUM_CHANNELS,
    'SF': 4
}


def agent_obs(state_composition):
    return Box(0.0, np.inf, shape=(sum([AGENT_STATE[s] for s in state_composition]),))


def env_obs(n_agents, state_composition):
    return Box(0.0, np.inf, shape=(agent_obs(state_composition).shape[0] * n_agents,))
    # return Tuple([agent_obs(binary, wind_in_state)]*n_agents)
#
# def agent_obs_one_hot(wind_in_state=False):
#     if wind_in_state:
#         return Box(np.array([0.0]*(FUTURE_STEPS * 2 + 6)), np.array([1.0]*(FUTURE_STEPS * 2 + 4)+[np.inf, np.inf]))
#     return Box(0.0, 1.0, shape=(FUTURE_STEPS * 2 + 4,))
#
#
# def env_obs_one_hot(n_agents, wind_in_state=False):
#     return Box(0.0, np.inf, shape=(agent_obs_one_hot(wind_in_state).shape[0] * n_agents,))
#     # return MultiBinary(agent_obs_one_hot(wind_in_state).shape[0] * n_agents)
#     # return Tuple([agent_obs_one_hot(wind_in_state)]*n_agents)

class g_env(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config, args):
        super().__init__()
        self.action_space = g_env.action_space
        self.state = None
        self.n_agents = env_config.get("n_agents", args.n_agents)
        self.args = args
        self._skip_env_checking = True

        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.with_state = env_config.get("separate_state_space", False)
        self.state_composition = env_config.get('state_composition', AGENT_STATE.keys())
        self.state_dim = agent_obs(self.state_composition).shape[0]
        # self.binary = env_config.get("binary", True)
        # if args.wind_in_state:
        #     self.binary = False
        self._agent_ids = {i for i in range(self.n_agents)}

        if self.with_state:
            self.observation_space = Dict(
                {
                    'obs': agent_obs(self.state_composition),
                    ENV_STATE: env_obs(self.n_agents, self.state_composition)
                }
            )
        else:
            self.observation_space = agent_obs(self.state_composition)

        # if self.one_hot_state_encoding:
        #     if self.with_state:
        #         self.observation_space = Dict(
        #             {
        #                 "obs": agent_obs_one_hot(self.args.wind_in_state),
        #                 ENV_STATE: env_obs_one_hot(self.n_agents, self.args.wind_in_state),
        #             }
        #         )
        #     else:
        #         self.observation_space = agent_obs_one_hot()
        # else:
        #     if self.with_state:
        #         self.observation_space = Dict(
        #             {
        #                 "obs": agent_obs(self.binary, args.wind_in_state),
        #                 ENV_STATE: env_obs(self.n_agents, self.binary, args.wind_in_state),
        #             }
        #         )
        #     else:
        #         self.observation_space = agent_obs(self.binary, args.wind_in_state)

        self.step_size = args.step_size
        self.T = self.step_size * args.steps_per_episodes
        self.spotting = args.spotting
        bound = Bound(57992, 54747, -14955, -11471)

        if self.args.alternating_wind:
            self.wind_vs = [450, 400]
            self.wind_th = [200, 100]
            self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', self.wind_vs,
                                       self.wind_th, 'dem', gisdb, location, mapset, 10)
        else:
            self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs',
                                           'sampleth', 'dem', gisdb, location, mapset, 10)
            self.wind_vs = [self.environment.vs]
            self.wind_th = [self.environment.th]
        self.environment.print_region()

        self.n_sensors = self.n_agents
        step_time = 6000
        offset = 2000

        np.random.seed(args.seed)
        self.row_idx = np.random.choice(self.environment.rows, self.n_sensors)
        self.col_idx = np.random.choice(self.environment.cols, self.n_sensors)
        node_indexes = [[self.row_idx[i], self.col_idx[i]]
                        for i in range(self.n_sensors)]

        self.seeds = np.random.choice(100000, 100000)
        self.communication = LoRaCommunication(node_indexes, [[self.environment.rows // 2, self.environment.cols // 2]],
                                               step_time, self.environment, 1, no_channels=NUM_CHANNELS, use_adr=True,
                                               offset=offset)
        self.communication.reset()

        self.region = Region(node_indexes, self.environment)
        self.dead_sensors = set()

        self.I = 0
        self.eps = 0
        self.records = []
        self.epi_data = {}
        result_path = os.path.join(root, 'results')

        SF = 'RS' if self.args.random_source else 'FS'

        dir_name = '{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.run, SF)

        if self.args.alternating_wind:
            dir_name = dir_name + f'_alter{args.alternating_wind}'
        if args.suffix:
            dir_name = dir_name + f'_{args.suffix}'

        self.logdir = os.path.join(result_path, dir_name)
        os.mkdir(self.logdir)
        json.dump(args.__dict__, open(os.path.join(self.logdir, 'args.txt'), 'w'))
        self.logging = os.path.join(self.logdir, 'runs.txt')
        json.dump(self.records, open(self.logging, 'w'))

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
        self.state = np.zeros(env_obs(self.n_agents, self.state_composition).shape[0]).astype(int)
        self.last_p = np.zeros(self.n_agents)
        self.records = []
        if self.args.random_source or self.eps == 1:
            self.seed(self.seeds[self.eps])
            a = np.random.choice(self.n_sensors)
            self.environment.set_source(self.region.cell_set[a])
        if self.args.alternating_wind:
            self.environment.generate_wildfire_alternate(self.T+FUTURE_STEPS*self.step_size, self.step_size*self.args.alternating_wind, spotting=self.spotting)
        else:
            self.environment.generate_wildfire(self.T+FUTURE_STEPS*self.step_size, spotting=self.spotting)
        self.epi_data = {'tcfire': self.environment.ground_truth,
                         'tcwind_vs': self.environment.vs,
                         'tcwind_th': self.environment.th,
                         'pcwind_vs': [],
                         'pcwind_th': [],
                         'pfire': [],
                         'tfire': []
                         }
        self.dead_sensors = set()
        return self._obs([0 for _ in range(self.n_agents)])

    def take_action(self, action_dict):
        if self.actions_are_logits:
            action_dict = {
                k: np.random.choice([0, 1], p=v) for k, v in action_dict.items()
            }
        for n in self.dead_sensors:
            action_dict[n] = 0
        print('----------EPS {}, ROUND {}----------'.format(self.eps, self.I))
        send_index, received = self.communication.step(list(action_dict.values()), False)
        if self.I == 0:
            if len(received) == 0:
                received.append(np.random.randint(self.n_agents))
            corrected, vs, th = self.region.model_update(received, self.I, SOURCE_NAME)
            future_predict, future_b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a) for a in range(FUTURE_STEPS + 1)])
            predict, b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size, 'predict', spotting=self.spotting)
        else:
            corrected, vs, th = self.region.model_update(received, self.I, 'predict')
            future_predict, future_b = self.region.predict(
                'predict', self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a) for a in range(FUTURE_STEPS + 1)])

            predict, b = self.region.predict(
                'predict', self.I, self.step_size, 'predict', spotting=self.spotting)
        # on_fire = self.environment.get_on_fire(self.I)
        # acc = 1 - np.sum(abs(on_fire - predict))/(self.region.cols*self.region.rows)
        current_state = [self.region.get_state(i, self.I)[-1] for i in range(self.n_agents)]
        on_fire = np.array(current_state)
        for i in range(self.n_agents):
            if on_fire[i] and np.random.random() < 0:
                self.dead_sensors.add(i)
        diff = np.absolute(on_fire - self.last_p)
        acc = 1 - np.dot(diff, np.array(self.region.area)/np.sum(self.region.area))

        rewards = {}
        if self.args.single_reward:
            avg = acc/max(len(send_index),1)
            rewards = {i: avg for i in range(self.n_agents)}
        else:
            for i in range(self.n_agents):
                if i in self.dead_sensors:
                    rewards[i] = 0
                elif i in send_index:
                    if i in corrected:
                        rewards[i] = CORRECTION_REWARD
                    elif i not in received:
                        rewards[i] = COMM_PENALTY
                    else:
                        rewards[i] = FUTILE_COMM
                elif diff[i]:
                    rewards[i] = PRED_PENALTY
                else:
                    rewards[i] = 0

        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        print(f'Predict area: {np.sum(self.last_p)}')
        print(f'Burning area: {np.sum(on_fire)}')
        print(f'accuracy: {acc}')
        print('reward: {}'.format(np.average(list(rewards.values()))))


        fb = (FUTURE_STEPS + 1 - np.sum(future_b, axis=0)).tolist()
        self.state = np.array([j for i in range(self.n_agents) for j in self.get_agent_obs(i, fb[i])])
        # self.state = np.array([self.get_agent_obs(i, fb[i]) for i in range(self.n_agents)])

        obs = self._obs(fb)

        self.records.append({
            'sent': len(send_index),
            'received': len(received),
            'rewards': sum(rewards.values()) / self.n_agents,
            'acc': acc,
            'sent_nodes': send_index,
            'received_nodes': received,
            'reward_nodes': [float(r) for r in rewards.values()],
            'burning_state': current_state
        })

        if hasattr(self.args, 'store') and self.args.store:
            self.epi_data['pfire'].append(fb)
            self.epi_data['tfire'].append(current_state)
            self.epi_data['pcwind_vs'].append(vs)
            self.epi_data['pcwind_th'].append(th)

        done = ((self.I >= self.T) or (np.sum(on_fire) > 0.3 * self.n_agents)) and (self.I > 10*self.step_size)
        if done:
            predata = json.load(open(self.logging, 'r'))
            predata.append(self.records)
            json.dump(predata, open(self.logging, 'w'))
            if hasattr(self.args, 'store') and self.args.store:
                pickle.dump(self.epi_data, open(os.path.join(self.logdir, f'data_{self.eps:02}.pk'), 'wb'))

        dones = {"__all__": done}
        infos = {}
        if self.args.as_test:
            infos = {'predict': predict, 'received': received, 'sent': send_index}

        self.I += self.step_size
        self.last_p = b

        return obs, rewards, dones, infos, fb


    def step(self, action_dict):
        obs, rewards, dones, infos, fb = self.take_action(action_dict)
        return obs, rewards, dones, infos

    def step_heuristic(self, action_dict):
        obs, rewards, dones, infos, fb = self.take_action(action_dict)
        return fb, dones["__all__"]

    def _obs(self, predict_dt):
        if self.with_state:
            return {
                i: self.get_agent_obs_with_state(i, predict_dt[i]) for i in range(self.n_agents)
            }
        else:
            return {
                i: self.get_agent_obs(i, predict_dt[i]) for i in range(self.n_agents)
            }

    def get_agent_obs(self, i, predict_bt):
        if i in self.dead_sensors:
            return np.zeros(self.state_dim)
        s = self.region.get_state(i, self.I)
        channel, sf = self.communication.get_sensor_para(i)

        state = [0 for _ in range(FUTURE_STEPS + 2)] + \
                [s[-1], s[0]/max([np.max(a) for a in self.wind_th]), s[1]/360]
        state[predict_bt] = 1

        if 'LOCATION' in self.state_composition:
            state += [self.row_idx[i]/self.environment.rows, self.col_idx[i]/self.environment.cols]

        if 'CHANNEL' in self.state_composition:
            comm_state = [0 for _ in range(NUM_CHANNELS + 4)]
            comm_state[sf] = 1
            comm_state[4 + channel] = 1
            state += comm_state

        return np.array(state)

    def get_agent_obs_with_state(self, i, predict_bt=0):
        return {"obs": self.get_agent_obs(i, predict_bt), ENV_STATE: self.state}