from config import *
env_init()

from Region import Region
from framework.LoRaCommunication import LoRaCommunication
from framework.GRASS import SOURCE_NAME
from framework.Environment import Environment
from framework.utils import *
import functools
from gym.spaces import Discrete
from pettingzoo import ParallelEnv

#from pettingzoo.utils import parallel_to_aec

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import random

#from config import *
#env_init()


ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


# def env():
#     """
#     The env function often wraps the environment in wrappers by default.
#     You can find full documentation for these methods
#     elsewhere in the developer documentation.
#     """
#     env = raw_env()
#     # This wrapper is only for environments which print results to the terminal
#     env = wrappers.CaptureStdoutWrapper(env)
#     # this wrapper helps error handling for discrete action spaces
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     # Provides a wide vareity of helpful user errors
#     # Strongly recommended
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env


# def raw_env():
#     """
#     To support the AEC API, the raw_env() function just uses the from_parallel
#     function to convert from a ParallelEnv to an AEC env
#     """
#     env = parallel_env()
#     env = parallel_to_aec(env)
#     return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        self.T = 600
        bound = Bound(57992, 54747, -14955, -11471)
        source = (56978.3098189104, -12406.60548812005)
        self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs',
                                       'sampleth', 'dem', source, gisdb, location, mapset, 10)
        self.environment.print_region()
        true_p = self.environment.generate_wildfire(self.T)
        plt.imshow(true_p)

        self.n_sensors = 100
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
        self.grid_input_features = ["temp", "wind"]
        self.agents_name = [str(i) for i in range(self.n_sensors)]
        self.I = 0
        self.step = 15
        self.observation_space = 2
        self.action_space = 2
        #self.parallel_env = to_parallel(self.environment)
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]
                      ], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: NONE for agent in self.agents}
        self.I = 0
        return observations
    
    def get_env_info(self) -> dict:
        map_info = {
            'grid_input_shape': [0, len(self.grid_input_features), self.environment.rows, self.environment.cols],
            'n_agents': self.n_sensors,
            'agents_name': self.agents_name,
            'obs_space': self.observation_space,
            'state_space': self.n_sensors*self.observation_space
        }
        map_info['n_actions'] = 2
        map_info['action_dim'] = 1
        map_info['action_space'] = self.action_space
        return map_info
    
    def state(self):
        return [i for i in range(self.n_sensors*self.observation_space)]
    
    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        on_fire = self.environment.get_on_fire(self.I)
        fire_zone = np.where(on_fire[-1] > 0)
        fire_zone = set(zip(fire_zone[0], fire_zone[1]))
        print(f'On-fire area: {len(fire_zone)}')
        send_index, received = self.communication.step(actions, False)

        n_send = len(send_index)
        n_received = len(received)
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        update_nodes = [False for j in range(self.n_sensors)]

        for j in received:
            update_nodes[j] = True

        points = [[col, row]
                  for col, row, a in zip(self.col_idx, self.row_idx, actions) if a]
        update_points = [[col, row] for col, row,
                         a in zip(self.col_idx, self.row_idx, update_nodes) if a]

        if self.I == 0:
            update_nodes = True
            self.region.model_update(update_nodes, self.I, SOURCE_NAME)
            predict, ros = self.region.predict(
                SOURCE_NAME, self.I, self.step, 'predict')
        else:
            self.region.model_update(update_nodes, self.I, 'predict')
            predict, ros = self.region.predict(
                'predict', self.I, self.step, 'predict')

        print(f'Predict area: {(predict>0).sum()}')

        rewards = {abs(on_fire - predict) for j in range(self.n_sensors)}
        self.I += 1

        self.num_moves += 1
        env_done = self.num_moves >= self.T
        dones = {agent: env_done for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: int(actions[self.agents[1 - i]])
            for i in range(len(self.agents))
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos
