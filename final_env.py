import math
import pickle, datetime, json
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

#from pettingzoo.utils import parallel_to_aec

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import random

from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple, MultiBinary


from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

FUTURE_STEPS = 5
COMM_PENALTY = -20
PRED_PENALTY = -5
CORRECTION_REWARD = 5


def agent_obs(binary=True):
    if binary:
        return MultiBinary(FUTURE_STEPS + 3)
    return Box(0.0, 1.0, shape=(FUTURE_STEPS + 3,))


def env_obs(n_agents, binary=True):
    if binary:
        return MultiBinary(agent_obs().shape[0] * n_agents)
    return Box(0.0, 1.0, shape=(agent_obs().shape[0]* n_agents,))


def agent_obs_one_hot():
    return MultiBinary(FUTURE_STEPS*2 + 4)


def env_obs_one_hot(n_agents):
    return MultiBinary(agent_obs_one_hot().shape[0] * n_agents)


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
        self.one_hot_state_encoding = env_config.get("one_hot_state_encoding", False)
        self.with_state = env_config.get("separate_state_space", False)
        self.binary = env_config.get("binary", True)
        self._agent_ids = {i for i in range(self.n_agents)}

        if self.one_hot_state_encoding:
            if self.with_state:
                self.observation_space = Dict(
                    {
                        "obs": agent_obs_one_hot(),
                        ENV_STATE: env_obs_one_hot(self.n_agents),
                    }
                )
            else:
                self.observation_space = agent_obs_one_hot()
        else:
            if self.with_state:
                self.observation_space = Dict(
                    {
                        "obs": agent_obs(self.binary),
                        ENV_STATE: env_obs(self.n_agents, self.binary),
                    }
                )
            else:
                self.observation_space = agent_obs(self.binary)

        self.step_size = args.step_size
        self.T = self.step_size * args.steps_per_episodes
        self.spotting = args.spotting
        bound = Bound(57992, 54747, -14955, -11471)
        self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs',
                                       'sampleth', 'dem', gisdb, location, mapset, 10)
        self.environment.print_region()

        self.n_sensors = self.n_agents
        step_time = 6000
        offset = 2000

        np.random.seed(0)
        self.row_idx = np.random.choice(self.environment.rows, self.n_sensors)
        self.col_idx = np.random.choice(self.environment.cols, self.n_sensors)
        node_indexes = [[self.row_idx[i], self.col_idx[i]]
                        for i in range(self.n_sensors)]

        self.seeds = np.random.choice(100000, 1000)
        self.communication = LoRaCommunication(node_indexes, [[self.environment.rows//2, self.environment.cols//2]],
                                               step_time, self.environment, 1, no_channels=2, use_adr=True, offset=offset)
        self.communication.reset()

        self.region = Region(node_indexes, self.environment)


        self.I = 0
        self.eps = 0
        self.records = []
        result_path = os.path.join(root, 'results')
        dir_name =  '{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.run)
        if args.suffix:
            dir_name = dir_name + f'_{args.suffix}'
        dir_name = os.path.join(result_path, dir_name)
        os.mkdir(dir_name)
        json.dump(args.__dict__, open(os.path.join(dir_name, 'args.txt'), 'w'))
        self.logging = os.path.join(dir_name, 'runs.txt')
        json.dump(self.records, open(self.logging, 'w'))


    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
        if self.one_hot_state_encoding:
            self.state = np.array([0 for i in range(env_obs_one_hot(self.n_agents).shape[0])])
        else:
            self.state = np.zeros(env_obs(self.n_agents).shape[0]).astype(int)
        self.last_p = np.zeros(self.n_agents)
        self.records = []
        if self.args.random_source or self.eps == 1:
            np.random.seed(self.seeds[self.eps])
            a = np.random.choice(self.n_sensors)
            self.environment.set_source(self.region.cell_set[a])
        self.environment.generate_wildfire(self.T, spotting=self.spotting)
        return self._obs([0 for _ in range(self.n_agents)])

    def step(self, action_dict):
        if self.actions_are_logits:
            action_dict = {
                k: np.random.choice([0, 1], p=v) for k, v in action_dict.items()
            }
        print('----------EPS {}, ROUND {}----------'.format(self.eps,self.I))
        send_index, received = self.communication.step(list(action_dict.values()), False)
        if self.I == 0:
            if len(received) == 0:
                received.append(np.random.randint(self.n_agents))
            corrected = self.region.model_update(received, self.I, SOURCE_NAME)
            future_predict, future_b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a) for a in range(FUTURE_STEPS+1)])
            predict, b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size, 'predict', spotting=self.spotting)
        else:
            corrected = self.region.model_update(received, self.I, 'predict')
            future_predict, future_b = self.region.predict(
                'predict', self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a ) for a in range(FUTURE_STEPS + 1)])

            predict, b = self.region.predict(
                'predict', self.I, self.step_size, 'predict', spotting=self.spotting)
        # on_fire = self.environment.get_on_fire(self.I)
        # acc = 1 - np.sum(abs(on_fire - predict))/(self.region.cols*self.region.rows)
        on_fire = np.array([self.region.get_state(i, self.I)[-1] for i in range(self.n_agents)])
        acc = np.absolute(on_fire - self.last_p)
        rewards = {i: acc[i] * PRED_PENALTY for i in range(self.n_agents)}
        for i in corrected:
            rewards[i] = CORRECTION_REWARD
        for i in send_index:
            if i not in received:
                rewards[i] = COMM_PENALTY
        acc = 1 - np.average(acc)
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        print(f'Predict area: {np.sum(self.last_p)}')
        print(f'Burning area: {np.sum(on_fire)}')
        print(f'accuracy: {acc}')
        print('reward: {}\t{}'.format(np.average(list(rewards.values())),list(rewards.values())))

        fb = np.sum(future_b, axis=0).tolist()
        self.state = np.array([j for i in range(self.n_agents) for j in self.get_agent_obs(i, fb[i])])
        obs = self._obs(fb)

        self.records.append({
            'sent': len(send_index),
            'received': len(received),
            'rewards': sum(rewards.values())/self.n_agents,
            'acc': acc
        })

        done = self.I >= self.T
        if done:
            predata = json.load(open(self.logging, 'r'))
            predata.append(self.records)
            json.dump(predata, open(self.logging, 'w'))

        dones = {"__all__": done}
        # infos = {'predict': predict, 'received': received, 'sent': send_index}
        infos = {}
        self.I +=  self.step_size
        self.last_p = b
        
        return obs, rewards, dones, infos

    def step_heuristic(self, action):
        print('----------EPS {}, ROUND {}----------'.format(self.eps, self.I))
        send_index, received = self.communication.step(action, False)
        if self.I == 0:
            if len(received) == 0:
                received.append(np.random.randint(self.n_agents))
            corrected = self.region.model_update(received, self.I, SOURCE_NAME)
            future_predict, future_b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a) for a in range(FUTURE_STEPS + 1)])
            predict, b = self.region.predict(
                SOURCE_NAME, self.I, self.step_size, 'predict', spotting=self.spotting)
        else:
            corrected = self.region.model_update(received, self.I, 'predict')
            future_predict, future_b = self.region.predict(
                'predict', self.I, self.step_size * FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                middle_state=[self.I + self.step_size * (a) for a in range(FUTURE_STEPS + 1)])

            predict, b = self.region.predict(
                'predict', self.I, self.step_size, 'predict', spotting=self.spotting)

        on_fire = np.array([self.region.get_state(i, self.I)[-1] for i in range(self.n_agents)])
        acc = np.absolute(on_fire - self.last_p)
        rewards = {i: acc[i] * PRED_PENALTY for i in range(self.n_agents)}
        for i in corrected:
            rewards[i] = CORRECTION_REWARD
        for i in send_index:
            if i not in received:
                rewards[i] = COMM_PENALTY
        acc = 1 - np.average(acc)
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        print(f'Predict area: {np.sum(self.last_p)}')
        print(f'Burning area: {np.sum(on_fire)}')
        print(f'accuracy: {acc}')
        print('reward: {}\t{}'.format(np.average(list(rewards.values())), list(rewards.values())))

        fb = FUTURE_STEPS + 1 - np.sum(future_b, axis=0)

        self.records.append({
            'sent': len(send_index),
            'received': len(received),
            'rewards': sum(rewards.values()) / self.n_agents,
            'acc': acc
        })

        done = self.I >= self.T
        if done:
            predata = json.load(open(self.logging, 'r'))
            predata.append(self.records)
            json.dump(predata, open(self.logging, 'w'))


        self.I += self.step_size
        self.last_p = b

        return fb, done

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
        if self.one_hot_state_encoding:
            state = [0 for _ in range(FUTURE_STEPS * 2 + 4)]
            idx = int(predict_bt + self.region.get_state(i, self.I)[-1] * (FUTURE_STEPS + 2))
            state[idx] = 1
        else:
            state = [0 for _ in range(FUTURE_STEPS + 2)] + [self.region.get_state(i, self.I)[-1]]
            state[predict_bt] = 1
        return np.array(state)

    def get_agent_obs_with_state(self, i, predict_bt=0):
        return {"obs": self.get_agent_obs(i, predict_bt), ENV_STATE: self.state}