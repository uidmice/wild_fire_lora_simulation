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


def agent_obs_t(state_composition):
    return Box(0.0, np.inf, shape=(sum([AGENT_STATE[s] for s in state_composition]),))


def env_obs_t(n_agents, state_composition):
    return Box(0.0, np.inf, shape=(agent_obs_t(state_composition).shape[0] * n_agents,))
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

class g_env_t(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config, args):
        super().__init__()
        self.action_space = g_env_t.action_space
        self.state = None
        self.n_agents = env_config.get("n_agents", args.n_agents)
        self.args = args
        self._skip_env_checking = True

        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.with_state = env_config.get("separate_state_space", False)
        self.state_composition = env_config.get('state_composition', AGENT_STATE.keys())
        self.state_dim = agent_obs_t(self.state_composition).shape[0]
        # self.binary = env_config.get("binary", True)
        # if args.wind_in_state:
        #     self.binary = False
        self._agent_ids = {i for i in range(self.n_agents)}

        if self.with_state:
            self.observation_space = Dict(
                {
                    'obs': agent_obs_t(self.state_composition),
                    ENV_STATE: env_obs_t(self.n_agents, self.state_composition)
                }
            )
        else:
            self.observation_space = agent_obs_t(self.state_composition)

        self.step_size = args.step_size
        self.T = self.step_size * args.steps_per_episodes
        self.spotting = args.spotting
        bound = Bound(57992, 54747, -14955, -11471)



        self.I = 0
        self.eps = 0
        self.records = []

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
        self.state = np.zeros(env_obs_t(self.n_agents, self.state_composition).shape[0]).astype(int)
        self.last_p = np.zeros(self.n_agents)
        self.records = []
        return self._obs([0 for _ in range(self.n_agents)])

    def take_action(self, action_dict):
        if self.actions_are_logits:
            action_dict = {
                k: np.random.choice([0, 1], p=v) for k, v in action_dict.items()
            }
        print('----------EPS {}, ROUND {}----------'.format(self.eps, self.I))

        rewards = {i: np.random.choice(9) for i in range(self.n_agents)}
        fb = [0 for i in range(self.n_agents)]

        self.state = np.array([j for i in range(self.n_agents) for j in self.get_agent_obs(i, fb[i])])
        # self.state = np.array([self.get_agent_obs(i, fb[i]) for i in range(self.n_agents)])
        obs = self._obs(fb)


        done = self.I > 240

        dones = {"__all__": done}
        infos = {}

        self.I += self.step_size

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
        return np.zeros(self.state_dim)


    def get_agent_obs_with_state(self, i, predict_bt=0):
        return {"obs": self.get_agent_obs(i, predict_bt), ENV_STATE: self.state}