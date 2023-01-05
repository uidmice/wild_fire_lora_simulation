import time
import pickle, datetime, json, os
from itertools import product
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


PROP_DATA = 'data'


NUM_CHANNELS = 2

COMM_PENALTY = -20
PRED_PENALTY = -10
CORRECTION_REWARD = 10
FUTILE_COMM = -10

WINDVS_LEVELS = 4
WINDTH_LEVELS = 8

COMM_RATIO = -5


def loadall(filename):
    rt = []
    with open(filename, "rb") as f:
        while True:
            try:
                rt.append(pickle.load(f))
            except EOFError:
                break
    return rt


class g_env:
    FUTURE_STEPS = 2
    AGENT_STATE = {
        'PREDICTION': FUTURE_STEPS + 2,
        'BURNING': 1,
        'WIND': 2,
        'LOCATION': 2,
        'CHANNEL': NUM_CHANNELS,
        'SF': 4
    }

    def __init__(self,  args):
        self.args = args
        if args.disable_prediction:
            g_env.FUTURE_STEPS = 0
            g_env.AGENT_STATE['PREDICTION'] = 0

        self.n_sensors = args.n_agents

        self.num_actions = 2
        if self.args.disable_prediction:
            self.args.simplified_state = True

        if self.args.simplified_state:
            self.state_composition = {'DATA': ["BURNING", "WIND", "LOCATION"],
                                      'COMM': ['SF', 'CHANNEL']}
        else:
            self.state_composition = {'DATA': ['PREDICTION', "BURNING", "WIND", "LOCATION"],
                                  'COMM': ['SF', 'CHANNEL']}
        self.obs_data_dim = sum([g_env.AGENT_STATE[feat] for feat in self.state_composition['DATA']])
        self.obs_comm_dim = sum([g_env.AGENT_STATE[feat] for feat in self.state_composition['COMM']])
        if not self.args.synchronous:
            self.obs_comm_dim += 1
        self.obs_dim = self.obs_data_dim + self.obs_comm_dim
        self.state_dim = self.n_sensors * self.obs_dim
        self.obs = None
        self.state = None


        self.step_size = args.wind_step_size
        self.T = self.step_size * (args.per_episode_max_len+1)
        self.spotting = args.spotting
        bound = Bound(57992, 54747, -14955, -11471)

        if hasattr(self.args, 'uneven_wind') and self.args.uneven_wind:
            self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', None,
                                          None, 'dem', gisdb, location, args.mapset, 10, self.args.uneven_wind)
        elif self.args.alternating_wind:
            self.wind_vs = [600, 600, 700, 500, 600, 300, 450, 500, 400, 400]
            self.wind_th = [200, 100, 20, 20, 20, 20, 100, 90, 90, 100]
            self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', self.wind_vs,
                                       self.wind_th, 'dem', gisdb, location, args.mapset, 10)
        else:
            self.environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs',
                                           'sampleth', 'dem', gisdb, location, args.mapset, 10)
            self.wind_vs = [self.environment.vs]
            self.wind_th = [self.environment.th]

        self.environment.print_region()
        self.environment.step_size = self.step_size*self.args.alternating_wind
        self.environment.simulation_time = self.T+g_env.FUTURE_STEPS*self.step_size

        step_time = 4000
        offset = 500

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

        self.dead_sensors = set()
        self.burning = np.zeros(self.n_sensors)


        self.I = 0
        self.eps = 0
        self.records = {}
        self.last_update = np.zeros(self.n_sensors)

        result_path = args.save_dir

        SF = 'RS' if self.args.random_source else 'FS'
        syn = "SYNC" if self.args.synchronous else 'ASYNC'


        dir_name = '{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), SF, args.run)
        if  hasattr(args, 'mixer') and args.run in ["QMIX", 'GNNQmix']:
            dir_name = dir_name + f'_{args.mixer}'
            if args.double_q:
                dir_name = dir_name + f'_dq'
            if args.simplified_state:
                dir_name = dir_name + '_simplified'
            dir_name = dir_name + '_' + self.args.data_reward
        dir_name = dir_name + f'_{syn}'

        # if self.args.alternating_wind:
        #     dir_name = dir_name + f'_alter{args.alternating_wind}'
        if args.suffix:
            dir_name = dir_name + f'_{args.suffix}'

        self.logdir = os.path.join(result_path, dir_name)
        os.mkdir(self.logdir)
        json.dump(args.__dict__, open(os.path.join(self.logdir, 'args.txt'), 'w'))
        self.logging = os.path.join(self.logdir, 'runs.pk')

        self.region = Region(node_indexes, self.environment, self.logdir, self.args)

        # json.dump(self.records, open(self.logging, 'w'))

        fn = PROP_DATA + f'/prop_ground_{self.n_sensors}_{self.step_size}'
        if self.args.limit_observation:
            fn += '_limit'

        if hasattr(self.args, 'uneven_wind') and self.args.uneven_wind:
                fn += '_uneven'

        self.gt_data = fn

        self.bad_episodes = []

        if not os.path.exists(fn):
            os.mkdir(fn)
            for i in range(self.n_sensors):
                print(i)
                self.environment.set_source(self.region.cell_set[i])
                if self.args.alternating_wind:
                    if hasattr(self.args, 'uneven_wind') and self.args.uneven_wind:
                        self.environment.generate_wildfire_uneven_wind(self.T + g_env.FUTURE_STEPS * self.step_size,
                                                                 self.step_size * self.args.alternating_wind,
                                                                self.logdir, spotting=self.spotting)
                    else:
                        self.environment.generate_wildfire_alternate(self.T + g_env.FUTURE_STEPS * self.step_size,
                                                                 self.step_size * self.args.alternating_wind,
                                                                 spotting=self.spotting)
                else:
                    self.environment.generate_wildfire(self.T + g_env.FUTURE_STEPS * self.step_size, spotting=self.spotting)

                pickle.dump([self.environment.ground_truth, self.environment.vs, self.environment.th], open(f'{fn}/wind_{i}.pk', 'wb'))
                if np.sum(self.environment.ground_truth) < 1000:
                    self.bad_episodes.append(i)

            json.dump(self.bad_episodes, open(f'{fn}/bad.txt', 'w'))
        else:
            self.bad_episodes = json.load(open(f'{fn}/bad.txt', 'r'))

        if self.args.disable_prediction:
            self.args.simplified_state = True



    def seed(self, s):
        s = int(s)
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)

    def reset(self, a=None):
        self.region.reset()
        print('----------EPS {}----------'.format(self.eps))
        self.eps += 1
        self.I = 0
        if self.args.random_source or self.eps == 1:
            self.seed(self.seeds[self.eps])
            if a is None:
                a = np.random.choice(self.n_sensors)
            while a in self.bad_episodes:
                a = np.random.choice(self.n_sensors)
            if self.args.unknown_source:
                random_source = random.sample(list(product(range(self.environment.rows), range(self.environment.cols))),
                                              self.environment.rows * self.environment.cols//10000)
                self.environment.set_source(random_source)
            else:
                self.environment.set_source(self.region.cell_set[a])
            self.environment.ground_truth, self.environment.vs, self.environment.th = pickle.load(open(f'{self.gt_data}/wind_{a}.pk', 'rb'))
            if self.args.uneven_wind:
                self.wind_vs = self.environment.vs
                self.wind_th = self.environment.th

        self.records = {
            'source': a,
            'sent': [],
            'received':[],
            'rewards': [],
            'acc': [],
            'track_acc': [],
            'sent_nodes': [],
            'received_nodes': [],
            'rewards_nodes': [],
            'burning_state': [],
            'sf': [],
            'update_time': [],
            'predict_time': [],
            'num_corrected': [],
            'num_correction_required': []
        }

        self.dead_sensors = set()
        self.fb = [g_env.FUTURE_STEPS + 1 for _ in range(self.n_sensors)]

        self.last_p = np.zeros(self.n_sensors)
        self.last_p[a] = 1

        self.update_state(self.fb, self.last_p, self.region.masks[a].copy(), [0 for i in range(self.n_sensors)])
        self.last_update = np.zeros(self.n_sensors)

        self.disregard_area = np.zeros((self.environment.rows, self.environment.cols))
        if self.args.disable_prediction:
            self.disregard_area = np.zeros(self.n_sensors)

    def take_action(self, actions, step_size):
        for n in self.dead_sensors:
            assert not actions[n], "dead sensor should not act"
        print('----------EPS {}, ROUND {}----------'.format(self.eps, self.I))
        send_index, received = self.communication.step(actions, False)

        print(send_index)

        if self.args.disable_prediction:
            predict = self.last_p_cell.copy()
            b = self.last_p.copy()
            corrected = []
            for i in received:
                if b[i] and not self.burning[i]:
                    corrected.append(i)
                    b[i] = 0
                    predict = predict - self.region.masks[i]
                elif not b[i] and self.burning[i]:
                    corrected.append(i)
                    b[i] = 1
                    predict = predict + self.region.masks[i]

            future_b = np.array([b.copy()])

            diff = np.absolute(self.last_p - self.burning)
            blank = (1 - self.last_p) * (1 - self.burning)
            acc = 1 - np.sum(diff * np.array(self.region.area))/(self.environment.cols * self.environment.rows)

            ca = np.ones(self.n_sensors) - self.disregard_area - blank
            track_acc = 1
            if np.sum(ca) > 0:
                track_acc = 1 - np.sum(diff * np.array(self.region.area))/np.sum(ca * np.array(self.region.area))

            self.disregard_area = np.where(self.burning * self.last_p + self.disregard_area > 0, 1, 0)

        else:
            if self.I == 0:
                if len(received) == 0:
                    received.append(np.random.randint(self.n_sensors))
                corrected, vs, th = self.region.model_update(received, self.I, SOURCE_NAME)
                future_predict, future_b = self.region.predict(
                    SOURCE_NAME, self.I, step_size * g_env.FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                    middle_state=[self.I + step_size * (a) for a in range(g_env.FUTURE_STEPS + 1)])

                predict, b = self.region.predict(
                    SOURCE_NAME, self.I, step_size, 'predict', spotting=self.spotting)
            else:
                corrected, vs, th = self.region.model_update(received, self.I, 'predict')
                future_predict, future_b = self.region.predict(
                    'predict', self.I, step_size * g_env.FUTURE_STEPS, 'predict_future', spotting=self.spotting,
                    middle_state=[self.I + step_size * (a) for a in range(g_env.FUTURE_STEPS + 1)])

                predict, b = self.region.predict(
                    'predict', self.I, step_size, 'predict', spotting=self.spotting)

            diff = np.absolute(self.burning_cell - self.last_p_cell)
            blank = (1 - self.burning_cell) * (1 - self.last_p_cell)
            acc = 1 - np.sum(diff) / (self.environment.cols * self.environment.rows)

            ca = self.environment.cols * self.environment.rows - np.sum(self.disregard_area) - np.sum(blank)
            track_acc = 1
            if ca > 0:
                track_acc = 1 - np.sum(diff) / ca

            self.disregard_area = np.where(self.burning_cell * self.last_p_cell + self.disregard_area > 0, 1, 0)

        for i in range(self.n_sensors):
            if self.burning[i] and np.random.random() < 0:
                self.dead_sensors.add(i)

        correction_eff = len(corrected)/10
        require_correction = set(np.where(self.burning != self.last_p)[0])
        # print(require_correction)

        # if len(require_correction):
        #     # assert set(corrected).issubset(require_correction), 'something wrong'
        #     # correction_eff = len(corrected)/len(require_correction)
        #     correction_eff = len(corrected)/10

        rewards = np.ones((self.n_sensors, 2))
        print(f' # of sent: {len(send_index)}')
        print(f' # of received: {len(received)}')
        print(f'Predict area: {np.sum(self.last_p)}')
        print(f'Burning area: {np.sum(self.burning)}')
        print(f'accuracy: {acc} ')
        print(f'# of corrections: {len(corrected)}')
        print(f'# of required corrections: {len(require_correction)}')

        # print(f'dead sensors: {self.dead_sensors}')
        if self.args.single_reward:
            # rewards[:, 0] *= acc + COMM_RATIO * len(send_index)/self.n_sensors
            if self.args.data_reward == 'track_acc':
                rewards[:, 0] = track_acc
            elif self.args.data_reward == 'acc':
                rewards[:, 0] = acc
            else:
                rewards[:, 0] = correction_eff
            rewards[:, 1] = 1- len(send_index)/self.n_sensors
            print(f'reward: {np.average(rewards, axis=0)}')
        else:
            diff_p = np.absolute(self.burning - self.last_p)
            for i in range(self.n_sensors):
                if i in self.dead_sensors:
                    rewards[i] = 0
                else:
                    if i in send_index:
                        if i not in received:
                            rewards[i, 1] = -1
                        else:
                            rewards[i, 1] = 0

                    if i in corrected:
                        rewards[i, 0] = 1
                    elif diff_p[i]:
                        rewards[i, 0] = -1
                    else:
                        rewards[i, 0] = 0
            print(f'reward: {np.average(rewards, axis=0)}\n {rewards[:,0]}\n {rewards[:,1]} ')

        fb = (g_env.FUTURE_STEPS + 1 - np.sum(future_b, axis=0)).tolist()

        if self.args.synchronous or self.I == 0:
            self.fb = fb
        else:
            for node in range(self.n_sensors):
                if node in received:
                    self.fb[node] = fb[node]
                    self.last_update[node] = 0
                else:
                    self.last_update[node] += 1
                    if self.fb[node] < g_env.FUTURE_STEPS + 1:
                        if self.fb[node] > 0:
                            self.fb[node] -= 1

        done = (np.sum(self.burning) > 0.5 * self.n_sensors) and (self.I > 10*self.step_size) and (np.sum(ca) <= 0)


        dones = np.ones(self.n_sensors) * done
        for n in self.dead_sensors:
            dones[n] = 1


        self.records['sent'].append(len(send_index))
        self.records['received'].append(len(received))
        self.records['rewards'].append(np.average(rewards,  axis=0))
        self.records['acc'].append(acc)
        self.records['track_acc'].append(track_acc)
        self.records['sent_nodes'].append(send_index)
        self.records['received_nodes'].append(received)
        self.records['rewards_nodes'].append(rewards)
        self.records['burning_state'].append(self.burning.copy())
        self.records['sf'].append([self.communication.get_sensor_para(i)[1] for i in range(self.n_sensors)])
        self.records['num_corrected'].append(len(corrected))
        self.records['num_correction_required'].append(len(require_correction))
        # self.records['update_time'].append(t1 - start)
        # self.records['predict_time'].append(t2 - t1)

        # self.records['last_status_cell'] = burning_cell
        # self.records['last_predict_cell'] = self.last_p_cell


        if self.args.visualize:
            infos = {'burning': self.burning_cell, 'predict': self.last_p_cell.copy(), 'received': received, 'sent': send_index}
            if self.args.disable_prediction:
                infos['disregard'] = np.sum([self.region.masks[i] for i in range(self.n_sensors) if self.disregard_area[i]>0], axis=0)
            else:
                infos['disregard'] =  self.disregard_area.copy()
        else:
            infos = None

        self.I += self.step_size
        self.update_state(self.fb, b, predict, actions)

        return rewards, dones, done, self.fb, infos


    def step(self, actions):
        return self.take_action(actions,  self.step_size // 10)


    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def update_state(self, predict_dt, b, predict, actions):
        self.last_p = b
        self.last_p_cell = predict
        self.last_action = np.eye(self.num_actions)[actions]

        self.burning_cell = self.environment.get_on_fire(self.I)
        self.burning = self.region.subregion_firing(self.burning_cell)

        self.obs = np.concatenate((np.array([self.get_agent_obs_data(i, predict_dt[i]) for i in range(self.n_sensors)]),
                                   np.array([self.get_agent_obs_comm(i) for i in range(self.n_sensors)])), axis=1)
        if not self.args.synchronous:
            self.obs = np.hstack([self.obs, np.exp(-self.last_update/3).reshape((self.n_sensors, 1))])
        self.state = self.obs.flatten()

        for i in self.dead_sensors:
            self.obs[i,:] = 0
        return

    def get_avail_actions(self):
        actions = np.ones((self.n_sensors, self.num_actions), dtype=int)
        for n in self.dead_sensors:
            actions[n,1:] = 0
        return actions


    def get_agent_obs_data(self, i, predict_bt):
        s = self.region.get_state(i, self.I)
        if self.args.simplified_state:
            r = int(s[-1])
            p = int(not predict_bt > 0)
            state = [1]
            if r != p:
                state = [0]
        else:
            state = [0 for _ in range(g_env.FUTURE_STEPS + 2)]
            state[predict_bt] = 1
            state = state + [s[-1]]

        if 'WIND' in self.state_composition['DATA']:
            state += [s[0] / max([np.max(a) for a in self.wind_vs]), s[1] / 360]

        if 'LOCATION' in self.state_composition['DATA']:
            state += [self.row_idx[i]/self.environment.rows, self.col_idx[i]/self.environment.cols]
        return state

    def get_agent_obs_comm(self, i):
        channel, sf = self.communication.get_sensor_para(i)
        comm_state = [0 for _ in range(4 + NUM_CHANNELS)]
        # comm_state = [0 for _ in range(4)]
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
        self.obs_data_dim = sum([g_env.AGENT_STATE[feat] for feat in self.state_composition['DATA']])
        self.obs_comm_dim = sum([g_env.AGENT_STATE[feat] for feat in self.state_composition['COMM']])
        self.obs_dim = self.obs_data_dim + self.obs_comm_dim
        self.state_dim = self.n_sensors * self.obs_dim
        self.obs = None
        self.state = None


        self.seed(args.seed)

        self.seeds = np.random.choice(100000, 1000)

        self.dead_sensors = set()
        self.logdir = 'test_log'
        self.burning = np.zeros(self.n_sensors)

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

        rewards = np.random.random((self.n_sensors, 2))

        fb = np.random.random_integers(0, 6, size=self.n_sensors).tolist()
        done = (self.I > 4) or (self.I > 0 and np.random.random() < 0.2)
        dones = np.ones(self.n_sensors) * int(done)
        for n in np.random.choice(self.n_sensors, 1):
            self.dead_sensors.add(n)
            dones[n] = 1

        self.I += 1
        self.last_action = np.eye(self.num_actions)[actions]

        self._obs(fb)
        print(f'dead sensors: {self.dead_sensors}')

        return rewards, dones, done, fb, None

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
        actions = np.ones((self.n_sensors, self.num_actions), dtype=int)
        for n in self.dead_sensors:
            actions[n,1:] = 0
        return actions
