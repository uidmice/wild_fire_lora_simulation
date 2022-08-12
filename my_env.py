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

OBSERVATIONS = {'vs': 1, 'th': 1, 'burning': 1}
FEEDBACK_DIM = 0
NONE = 0

class g_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, n_agent=5):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["sensor_" + str(r) for r in range(n_agent)]
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

        self.n_sensors = n_agent
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
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(2) for agent in self.possible_agents
        }

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
        pass


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
        self.observations = {agent: NONE for agent in self.agents}
        self.I = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        return self.observations
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observations[agent]

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

        send_index, received = self.communication.step(list(actions.values()), False)


        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')


        if self.I == 0:
            # received = [i for i in range(self.n_sensors)]
            self.region.model_update(received, self.I, SOURCE_NAME)
            predict, ros = self.region.predict(
                SOURCE_NAME, self.I, self.step_size, 'predict')
        else:
            self.region.model_update(received, self.I, 'predict')
            predict, ros = self.region.predict(
                'predict', self.I, self.step_size, 'predict')

        print(f'Predict area: {(predict>0).sum()}')


        self.I +=  self.step_size

        on_fire = self.environment.get_on_fire(self.I)
        
        acc = 1 - np.sum(abs(on_fire - predict))/(self.region.cols * self.region.rows)
        
        rewards = {agent: acc for agent in self.agents}
        # rewards = acc
        env_done = self.I >= self.T
        dones = {agent: env_done for agent in self.agents}

        self.observations = {self.agents[i]: self.region.get_state(i, self.I)[-1] for i in range(self.n_sensors)}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return self.observations, rewards, env_done, infos
