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

import numpy as np
import random



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


class g_env:
    def __init__(self,  args):
        self.args = args
        self.n_sensors = args.n_agents

        self.num_actions = 2
        self.state_composition = {'DATA': ['PREDICTION', "BURNING", "WIND", "LOCATION"],
                                  'COMM': ["CHANNEL", 'SF']}
        self.obs_data_dim = sum([AGENT_STATE[feat] for feat in self.state_composition['DATA']])
        self.obs_comm_dim = sum([AGENT_STATE[feat] for feat in self.state_composition['COMM']])
        self.obs_dim = self.obs_data_dim + self.obs_comm_dim
        self.state_dim = self.n_sensors * self.obs_dim
        self.obs = None
        self.state = None


        self.step_size = args.wind_step_size
        self.T = self.step_size * (args.per_episode_max_len+1)
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

        step_time = 6000
        offset = 2000

        self.seed(args.seed)
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
        result_path = args.save_dir

        SF = 'RS' if self.args.random_source else 'FS'


        dir_name = '{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.run, SF)
        if  hasattr(args, 'mixer') and args.mixer != 'QMIX':
            dir_name = dir_name + f'_{args.mixer}'

        if self.args.alternating_wind:
            dir_name = dir_name + f'_alter{args.alternating_wind}'
        if args.suffix:
            dir_name = dir_name + f'_{args.suffix}'

        self.logdir = os.path.join(result_path, dir_name)
        os.mkdir(self.logdir)
        json.dump(args.__dict__, open(os.path.join(self.logdir, 'args.txt'), 'w'))
        self.logging = os.path.join(self.logdir, 'runs.txt')
        json.dump(self.records, open(self.logging, 'w'))

    def seed(self, s):
        s = int(s)
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)

    def reset(self):
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
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
        self._obs([0 for _ in range(self.n_sensors)])
        self.last_p = np.zeros(self.n_sensors)
        self.last_action = np.array([[1,0] for i in range(self.n_sensors)])



    def take_action(self, actions):
        for n in self.dead_sensors:
            assert not actions[n], "dead sensor should not act"
        print('----------EPS {}, ROUND {}----------'.format(self.eps, self.I))
        send_index, received = self.communication.step(actions, False)
        if self.I == 0:
            if len(received) == 0:
                received.append(np.random.randint(self.n_sensors))
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

        for i in range(self.n_sensors):
            if self.burning[i] and np.random.random() < 0:
                self.dead_sensors.add(i)
        diff = np.absolute(self.burning - self.last_p)
        acc = 1 - np.dot(diff, np.array(self.region.area)/np.sum(self.region.area))
        rewards = np.ones(self.n_sensors)
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        print(f'Predict area: {np.sum(self.last_p)}')
        print(f'Burning area: {np.sum(self.burning)}')
        print(f'accuracy: {acc} ')
        print(f'dead sensors: {self.dead_sensors}')
        if self.args.single_reward:
            rewards *= acc / max(len(send_index), 1)
            print(f'reward: {np.average(rewards)}')
        else:
            for i in range(self.n_sensors):
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
            print(f'reward: {np.average(rewards)}, {rewards}')


        fb = (FUTURE_STEPS + 1 - np.sum(future_b, axis=0)).tolist()

        done = (np.sum(self.burning) > 0.3 * self.n_sensors) and (self.I > 10*self.step_size)
        if done:
            dones = np.ones(self.n_sensors)
        else:
            dones = np.zeros(self.n_sensors)
            for n in self.dead_sensors:
                dones[n] = 1

        self.records.append({
            'sent': len(send_index),
            'received': len(received),
            'rewards': np.average(rewards),
            'acc': acc,
            'sent_nodes': send_index,
            'received_nodes': received,
            'reward_nodes': [float(r) for r in rewards],
            'burning_state': self.burning.tolist()
        })

        if hasattr(self.args, 'store') and self.args.store:
            self.epi_data['pfire'].append(fb)
            self.epi_data['tfire'].append(self.burning)
            self.epi_data['pcwind_vs'].append(vs)
            self.epi_data['pcwind_th'].append(th)

        if done:
            predata = json.load(open(self.logging, 'r'))
            predata.append(self.records)
            json.dump(predata, open(self.logging, 'w'))
            if hasattr(self.args, 'store') and self.args.store:
                pickle.dump(self.epi_data, open(os.path.join(self.logdir, f'data_{self.eps:02}.pk'), 'wb'))


        self.I += self.step_size
        self.last_p = b
        self.last_action = np.eye(self.num_actions)[actions]

        self._obs(fb)
        return rewards, dones, done, fb


    def step(self, actions):
        return self.take_action(actions)


    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def _obs(self, predict_dt):
        self.obs = np.concatenate((np.array([self.get_agent_obs_data(i, predict_dt[i]) for i in range(self.n_sensors)]), np.array([self.get_agent_obs_comm(i) for i in range(self.n_sensors)])), axis=1)
        self.state = self.obs.flatten()
        # self.burning = np.array([self.region.get_state(i, self.I)[-1] for i in range(self.n_sensors)])
        self.burning = self.obs[:, FUTURE_STEPS + 2].copy()
        for i in self.dead_sensors:
            self.obs[i,:] = 0
        return

    def get_avail_actions(self):
        actions = np.ones((self.n_sensors, self.num_actions))
        for n in self.dead_sensors:
            actions[n,1:] = 0
        return actions


    def get_agent_obs_data(self, i, predict_bt):

        s = self.region.get_state(i, self.I)

        state = [0 for _ in range(FUTURE_STEPS + 2)] + [s[-1]]
        state[predict_bt] = 1

        if 'WIND' in self.state_composition['DATA']:
            state += [s[0] / max([np.max(a) for a in self.wind_th]), s[1] / 360]

        if 'LOCATION' in self.state_composition['DATA']:
            state += [self.row_idx[i]/self.environment.rows, self.col_idx[i]/self.environment.cols]
        return state

    def get_agent_obs_comm(self, i):
        channel, sf = self.communication.get_sensor_para(i)
        comm_state = [0 for _ in range(NUM_CHANNELS + 4)]
        comm_state[sf] = 1
        comm_state[4 + channel] = 1
        return comm_state


class test_env:
    def __init__(self,  args):
        self.args = args
        self.n_sensors = args.n_agents

        self.num_actions = 2
        self.state_composition = {'DATA': ['PREDICTION', "BURNING", "WIND", "LOCATION"],
                                  'COMM': ["CHANNEL", 'SF']}
        self.obs_data_dim = sum([AGENT_STATE[feat] for feat in self.state_composition['DATA']])
        self.obs_comm_dim = sum([AGENT_STATE[feat] for feat in self.state_composition['COMM']])
        self.obs_dim = self.obs_data_dim + self.obs_comm_dim
        self.state_dim = self.n_sensors * self.obs_dim
        self.obs = None
        self.state = None


        self.seed(args.seed)

        self.seeds = np.random.choice(100000, 1000)

        self.dead_sensors = set()
        self.logdir = 'test_log'

        self.I = 0
        self.eps = 0

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def reset(self):
        self.eps += 1
        self.I = 0
        self.records = []

        self.dead_sensors = set()
        self._obs([0 for _ in range(self.n_sensors)])
        self.last_action = np.array([[1, 0] for i in range(self.n_sensors)])

    def take_action(self, actions):
        for n in self.dead_sensors:
            assert not actions[n], "dead sensor should not act"

        rewards = np.random.random(self.n_sensors) * 10

        fb = np.random.random_integers(0, 6, size=self.n_sensors).tolist()
        done = self.I > 4
        dones = np.ones(self.n_sensors) * int(done)

        self.I += 1
        self.last_action = np.eye(self.num_actions)[actions]

        self._obs(fb)
        return rewards, dones, done, fb

    def step(self, actions):
        return self.take_action(actions)

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def _obs(self, predict_dt):
        self.obs = np.random.random((self.n_sensors, self.obs_dim))
        self.state = self.obs.flatten()
        return

    def get_avail_actions(self):
        actions = np.ones((self.n_sensors, self.num_actions))
        actions[0, 1] = 0
        return actions
