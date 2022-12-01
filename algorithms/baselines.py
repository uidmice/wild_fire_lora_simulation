import random
import numpy as np
import torch
from replay_buffer import ReplayBuffer


class Base_Agent:
    def __init__(self):
        self.memory = ReplayBuffer(1)

    def init_trainers(self, *args):
        return

    def select_actions(self, avail_actions, *args, **kwargs):
        return np.array([0 for _ in range(len(avail_actions))]), None

    def save_memory(self, obs_and_u_last, state, u, new_avail_actions, new_obs_and_u, state_new, r, dones, done):
        done = np.array([done])[np.newaxis, :]
        self.memory.add((obs_and_u_last[np.newaxis, :], state[np.newaxis, :], u[np.newaxis, :], \
                         new_avail_actions[np.newaxis, :], new_obs_and_u[np.newaxis, :], state_new[np.newaxis, :],
                         r[np.newaxis, :], dones, done))

    def learn(self, *args):
        return


class Random_Agent(Base_Agent):
    def __init__(self, p, num_actions, num_agents):
        super(Random_Agent, self).__init__()
        self.p = np.array(p)
        self.num_actions = num_actions
        assert len(self.p) == self.num_actions
        self.num_agents = num_agents

    def select_actions(self, avail_actions, *args, **kwargs):
        p = avail_actions * self.p
        p = p / p.sum(axis=1)[:, np.newaxis]
        return np.array([np.random.choice(self.num_actions, p=p[i]) for i in range(self.num_agents)]), None


class Heuristic_Agent(Base_Agent):
    def __init__(self):
        super(Heuristic_Agent, self).__init__()

    def select_actions(self, avail_actions, *args, **kwargs):
        fb = kwargs['fb']
        burning = kwargs['burning']
        if not fb:
            return np.array([0 for _ in range(len(avail_actions))])

        arg = kwargs['args']
        if arg.run == 'HEURISTIC1':
            return np.array([int(a == 2) for a in fb]), None

        elif arg.run == 'HEURISTIC2':
            return np.array([int((a > 0) and b) for a, b in zip(fb, burning)]), None

        elif arg.run == 'HEURISTIC3':
            return np.array([int((a > 0) and b) + int(not a and not b) for a, b in zip(fb, burning)]), None

        elif arg.run == 'HEURISTIC4':
            temp = np.array([int((a > 0) and b) + int(not a and not b) for a, b in zip(fb, burning)])
            return temp * np.random.choice([0,1], size=len(fb), replace=True, p=[1-arg.random_prob, arg.random_prob]), None
        else:
            raise ValueError




class Heuristic_Agent2(Base_Agent):
    def __init__(self):
        super(Heuristic_Agent2, self).__init__()

    def select_actions(self, avail_actions, *args, **kwargs):
        fb = kwargs['fb']
        burning = kwargs['burning']
        if not fb:
            return np.array([0 for _ in range(len(avail_actions))])

        return np.array([int((a > 0) and b) for a, b in zip(fb, burning)]), None

        # return np.array([int((a > 0) and b) + int(not a and not b) for a, b in zip(fb, burning)]), None

class Heuristic_Agent3(Base_Agent):
    def __init__(self):
        super(Heuristic_Agent3, self).__init__()

    def select_actions(self, avail_actions, *args, **kwargs):
        fb = kwargs['fb']
        burning = kwargs['burning']
        if not fb:
            return np.array([0 for _ in range(len(avail_actions))])

        return np.array([int((a > 0) and b) + int(not a and not b) for a, b in zip(fb, burning)]), None