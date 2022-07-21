import numpy as np
import random

from framework.GRASS import *

class Region:
    def __init__(self, node_indices, env):
        self.env = env
        self.cols = env.cols
        self.rows = env.rows

        self.n_points = len(node_indices)

        self.point_set = node_indices
        self.point_loc = [[env.grass_r.north-idx[0]*env.grass_r.nsres, idx[1]*env.grass_r.ewres + env.grass_r.west] for idx in self.point_set]
        self.sub_regions = (-1 * np.ones((self.rows, self.cols))).astype(int)

        temp = np.zeros((self.rows, self.cols, len(self.point_set)))

        for i in range(self.rows):
            for j in range(self.cols):
                for k, p in enumerate(self.point_set):
                    temp[i, j, k] = (p[0] - i)**2 + (p[1] - j)**2
                self.sub_regions[i,j] = np.argmin(temp[i, j])

        self.masks = [np.where(self.sub_regions==i, 1, 0) for i in range(self.n_points)]
        self.cell_set = [np.transpose(np.where(self.sub_regions==i)).tolist() for i in range(self.n_points)]
        self.cell_set = [set([tuple(i) for i in l]) for l in self.cell_set ]
        self.area = [np.sum(ma) for ma in self.masks]

        self.reset()

    def reset(self):
        self.data = {
            'vs': np.ones(self.n_points) * (-1),
            'th': np.ones(self.n_points) * (-1)
        }


    # def step(self, actions, time):
    #     self.model_update(actions, time)
    #     self.predict


    def model_update(self, actions, time, source_name, suffix='', percent=0.5):
        predict = raster.raster2numpy(source_name).astype(float)
        predict_state = [np.sum(np.where((predict<= time) & (predict >0), 1, 0) * self.masks[i])>= self.area[i]*percent for i in range(self.n_points)]
        rast = raster.RasterSegment(source_name)
        rast.open('rw', overwrite=True)

        send_index = [idx for idx, send in enumerate(actions) if send]
        for i in send_index:
            sensed = self.env.sense_region(self.point_set[i][0], self.point_set[i][1], self.masks[i], time)
            self.data['vs'][i] = sensed['vs']
            self.data['th'][i] = sensed['th']
            n = int(self.area[i]*percent)
            report_state = sensed['fire_area'] >= n
            if report_state and not predict_state[i]:
                p = np.where((predict<= time) & (predict>0), 1, 0)
                t = np.where((p==0) & (self.masks[i]==1))
                wrong_cells = set(zip(t[0], t[1]))
                whole_cells = self.cell_set[i]
                assert wrong_cells.issubset(whole_cells)
                n_correct = n - (self.area[i] - len(wrong_cells))
                random_set = set(random.sample(wrong_cells, n_correct))
                for c in random_set:
                    rast[c[0], c[1]] = time


            if not report_state and predict_state[i]:
                p = np.where((predict<= time) & (predict>0), 1, 0)
                t = np.where((p == 1) & (self.masks[i] == 1))
                wrong_cells = set(zip(t[0], t[1]))
                whole_cells = self.cell_set[i]
                assert wrong_cells.issubset(whole_cells)
                random_set = set(random.sample(wrong_cells,  len(wrong_cells)-n))
                for c in random_set:
                    rast[c[0], c[1]] = 0

        rast.close()
        predict = raster.raster2numpy(source_name).astype(float)
        print((predict>0).sum())



        data_idx = np.where(self.data['vs'] > 0)[0]
        y = [self.point_loc[i][1] for i in data_idx]
        x = [self.point_loc[i][0] for i in data_idx]
        vs = self.data['vs'][data_idx]
        th = self.data['th'][data_idx]

        with open("data/temp.txt", "w") as f:
            for c in zip(y, x, vs, th):
                f.write('|'.join([str(a) for a in c]) + '\n')

        script.run_command('v.in.ascii', input='data/temp.txt', output='sample', overwrite=True,
                           columns='x double precision, y double precision, vsfpm double precision, mean double precision')
        caldata(REGION_SAVE_NAME, PREDICTION_SUFFIX + suffix, self.env.res, samplevs='sample', sampleth='sample')




    def predict(self, source, init_time, lag, output, suffix='', spotting=False, middle_state=None):
        pre, ros = self.env.propogate(source, init_time, lag, suffix=PREDICTION_SUFFIX + suffix, ros_out='pre_out',
                                      spread_out=output, spotting=spotting, middle_state=middle_state)

        return pre, ros











