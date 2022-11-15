import numpy as np
import random
from itertools import product
from framework.GRASS import *

OBSERVATION_RANGE = 2

class Region:
    def __init__(self, node_indices, env, logdir, args, subregion_fire_percent=0.5):
        self.env = env
        self.logdir = logdir
        self.cols = env.cols
        self.rows = env.rows

        self.n_points = len(node_indices)

        self.point_set = node_indices
        self.point_loc = [[env.grass_r.north-idx[0]*env.grass_r.nsres, idx[1]*env.grass_r.ewres + env.grass_r.west] for idx in self.point_set]

        if not args.limit_observation:
            sub_regions = (-1 * np.ones((self.rows, self.cols))).astype(int)
            temp = np.zeros((self.rows, self.cols, len(self.point_set)))

            for i in range(self.rows):
                for j in range(self.cols):
                    for k, p in enumerate(self.point_set):
                        temp[i, j, k] = (p[0] - i)**2 + (p[1] - j)**2
                    sub_regions[i,j] = np.argmin(temp[i, j])
            self.masks = [np.where(sub_regions == i, 1, 0) for i in range(self.n_points)]
            self.cell_set = [np.transpose(np.where(sub_regions == i)).tolist() for i in range(self.n_points)]
            self.cell_set = [set([tuple(i) for i in l]) for l in self.cell_set]

        else:
            self.masks = [np.zeros((self.rows, self.cols)) for i in range(self.n_points)]
            self.cell_set = []
            for k, idx in enumerate(self.point_set):
                xmin = max(0, idx[0]-OBSERVATION_RANGE)
                xmax = min(self.rows, idx[0]+OBSERVATION_RANGE+1)
                ymin = max(0, idx[1]-OBSERVATION_RANGE)
                ymax = min(self.cols, idx[1]+OBSERVATION_RANGE+1)
                self.masks[k][xmin:xmax, ymin:ymax] = 1
                self.cell_set.append(set(product(range(xmin, xmax), range(ymin, ymax))))
                assert np.sum(self.masks[k]) == len(self.cell_set[-1])


        self.area = [np.sum(ma) for ma in self.masks]

        self.percent_fire = subregion_fire_percent
        self.fire_threshold = [a * self.percent_fire for a in self.area]

        self.reset()

    def reset(self):
        self.data = {
            'vs': [],
            'th': [],
            'loc': []
        }
        self.idx = 0
        self.history = 0


    def model_update(self, received, time, source_name, suffix=''):
        predict = raster.raster2numpy(source_name).astype('int32')
        p = np.where((predict <= time) & (predict > 0), 1, 0)

        predict_state = [np.sum(p * self.masks[i])>= self.fire_threshold[i] for i in range(self.n_points)]
        # rast = raster.RasterSegment(source_name)
        # rast.open('rw', overwrite=True)
        corrected = []

        for i in received:
            sensed = self.env.sense_region(self.point_set[i][0], self.point_set[i][1], self.masks[i], time)
            if self.history < 10:
                self.data['vs'].append(sensed['vs'])
                self.data['th'].append(sensed['th'])
                self.data['loc'].append(i)
                self.history += 1
            else:
                self.data['vs'][self.idx] = sensed['vs']
                self.data['th'][self.idx] = sensed['th']
                self.data['loc'][self.idx] = i

            self.idx = (self.idx + 1) % 10

            n = int(self.fire_threshold[i])
            report_state = sensed['fire_area'] >= n
            if report_state and not predict_state[i]:
                t = np.where((p==0) & (sensed['firing']==1))
                wrong_cells = set(zip(t[0], t[1]))
                whole_cells = self.cell_set[i]
                assert wrong_cells.issubset(whole_cells)
                # n_correct = n - (self.area[i] - len(wrong_cells))
                # random_set = set(random.sample(wrong_cells, n_correct))
                # for c in wrong_cells:
                #     rast[c[0], c[1]] = time
                corrected.append(i)
                predict[(p==0) & (sensed['firing']==1)] = time


            if not report_state and predict_state[i]:
                t = np.where((p == 1) & (self.masks[i] == 1))
                wrong_cells = set(zip(t[0], t[1]))
                whole_cells = self.cell_set[i]
                assert wrong_cells.issubset(whole_cells)
                random_set = set(random.sample(list(wrong_cells),  len(wrong_cells)-n))
                for c in random_set:
                    predict[c[0], c[1]] = 0
                corrected.append(i)
        if len(corrected):
            # buffer = raster.Buffer(predict.shape, buffer = predict)
            rast = raster.RasterRow(source_name)
            rast.open('w', overwrite=True)
            for i in range(self.rows):
                rast.put_row(raster.Buffer(shape=(self.cols, ), buffer=predict[i]))
            rast.close()

        y = [self.point_loc[i][1] for i in self.data['loc']]
        x = [self.point_loc[i][0] for i in self.data['loc']]
        vs = self.data['vs']
        th = self.data['th']

        with open(os.path.join(self.logdir, "temp.txt"), "w") as f:
            for c in zip(y, x, vs, th):
                f.write('|'.join([str(a) for a in c]) + '\n')

        script.run_command('v.in.ascii', input=os.path.join(self.logdir, "temp.txt"), output='sample', overwrite=True,
                           columns='x double precision, y double precision, vsfpm double precision, mean double precision', quiet=True)
        vs, th = caldata(REGION_SAVE_NAME, PREDICTION_SUFFIX + suffix, self.env.res, samplevs='sample', sampleth='sample')
        return corrected, vs, th




    def predict(self, source, init_time, lag, output, suffix='', spotting=False, middle_state=None):
        pre, ros = self.env.propogate(source, init_time, lag, suffix=PREDICTION_SUFFIX + suffix, ros_out='pre_out',
                                      spread_out=output, spotting=spotting, middle_state=middle_state)
        if not middle_state:
            b = np.array(self.subregion_firing(pre))
        else:
            b = np.array([self.subregion_firing(p) for p in pre])

        return pre, b

    def subregion_firing(self, firing_state):
        burning_a = []
        for ma in self.masks:
            masked = firing_state * ma
            burning_a.append(np.sum(masked))
        # burning_a = [np.sum(firing_state * ma) for ma in self.masks]
        burning = [int(a > am) for a, am in zip(burning_a, self.fire_threshold)]
        return burning

    def get_state(self, idx, time):
        sensed = self.env.sense_region(self.point_set[idx][0], self.point_set[idx][1], self.masks[idx], time)
        n = int(self.area[idx] * 0.5)
        return  [sensed['vs'], sensed['th'], int(sensed['fire_area'] >= n)]













