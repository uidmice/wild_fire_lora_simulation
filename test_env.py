import math
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

from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class g_env(MultiAgentEnv):
    action_space = Discrete(2)
    def __init__(self, env_config):
        super().__init__()
        self.action_space = Discrete(2)
        self.state = None
        self.n_agents = env_config.get("n_agents", 5)
        self._skip_env_checking = True
        # MADDPG emits action logits instead of actual discrete actions
        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.one_hot_state_encoding = env_config.get("one_hot_state_encoding", False)
        self.with_state = env_config.get("separate_state_space", False)
        self._agent_ids = {i for i in range(self.n_agents)}
        if not self.one_hot_state_encoding:
            self.observation_space = MultiDiscrete([2 for i in range(self.n_agents)])
            self.with_state = False
        else:
            # Each agent gets the full state (one-hot encoding of which of the
            # three states are active) as input with the receiving agent's
            # ID (1 or 2) concatenated onto the end.
            if self.with_state:
                self.observation_space = Dict(
                    {
                        "obs": Discrete(2),
                        ENV_STATE: MultiDiscrete([2 for i in range(self.n_agents)]),
                    }
                )
            else:
                self.observation_space = MultiDiscrete([2 for i in range(self.n_agents)])
                
        self.T = 600
        bound = Bound(57992, 54747, -14955, -11471)
        source = (56978.3098189104, -12406.60548812005)
        self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs',
                                       'sampleth', 'dem', source, gisdb, location, mapset, 10)
        self.environment.print_region()
        true_p = self.environment.generate_wildfire(self.T)
        plt.imshow(true_p)

        self.n_sensors = self.n_agents
        step_time = 6000
        offset = 2000

        np.random.seed(0)
        self.row_idx = np.random.choice(self.environment.rows, self.n_sensors)
        self.col_idx = np.random.choice(self.environment.cols, self.n_sensors)
        node_indexes = [[self.row_idx[i], self.col_idx[i]]
                        for i in range(self.n_sensors)]
        self.communication = LoRaCommunication(node_indexes, [[self.environment.rows//2, self.environment.cols//2]],
                                               step_time, self.environment, 1, no_channels=2, use_adr=True, offset=offset)
        self.communication.reset()

        self.region = Region(node_indexes, self.environment)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.region.sub_regions)
        plt.scatter(self.col_idx, self.row_idx, c='r', marker='D', s=10)
        plt.show()
        self.I = 0
        self.step_size = 15
        self.eps = 0

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
        self.state = np.array([0 for i in range(self.n_agents)])
        return self._obs()

    def step(self, action_dict):
        if self.actions_are_logits:
            action_dict = {
                k: np.random.choice([0, 1], p=v) for k, v in action_dict.items()
            }
        print('----------EPS {}, ROUND {}----------'.format(self.eps,self.I))
        send_index, received = self.communication.step(list(action_dict.values()), False)
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')

        rewards = {i: len(send_index) for i in range(self.n_agents)}
        print('reward: {}'.format(len(send_index)))
        # rewards = acc
        done = self.I >= self.T
        dones = {"__all__": done}
        infos = {}
        
        self.I +=  self.step_size
        self.state = [self.region.get_state(i, self.I)[-1] for i in range(self.n_agents)]
        obs = self._obs()
        
        return obs, rewards, dones, infos
       
    def _obs(self):
        if self.with_state:
            return {
                i: self.obs_agent(i) for i in range(self.n_agents)
            }
        else:
            return 
    
    def obs_agent(self, i):
        return {"obs": self.region.get_state(i, self.I)[-1], ENV_STATE: self.state}