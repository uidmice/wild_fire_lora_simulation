import numpy as np
from config import *
import random

from .GRASS import *

class FieldReconstructor:
    def __init__(self, environment):
        self.data = {}
        self.env = environment
        self.update_required = False

    def update(self, info):
        if info.node_id not in self.data:
            self.data[info.node_id] = [ info.payload['y'], info.payload['x'], info.payload['vs'], info.payload['th']]
            with open("data/temp.txt", "a") as f:
                f.write('|'.join([str(a) for a in self.data[info.node_id]])+ '\n')
            self.update_required = True

    def model_update(self, suffix=''):
        script.run_command('v.in.ascii', input='data/temp.txt', output='sample', overwrite=True, columns='x double precision, y double precision, vsfpm double precision, mean double precision')
        caldata(REGION_SAVE_NAME, PREDICTION_SUFFIX+suffix, self.env.res, samplevs='sample', sampleth='sample')
        self.vs = raster.raster2numpy(WIND_SPEED + PREDICTION_SUFFIX+suffix)
        self.th = raster.raster2numpy(WIND_DIR + PREDICTION_SUFFIX+suffix)
        self.update_required = False


    def field_reconstruct(self, source, init_time, lag, output, suffix='', spotting=False):
        if self.update_required:
            self.model_update(suffix)
        pre, ros = self.env.propogate(source, init_time, lag, suffix=PREDICTION_SUFFIX+suffix, ros_out='pre_out', spread_out=output, spotting=spotting)
        return pre, ros

    def reset(self):
        self.data = {}
        open('data/temp.txt', 'w').close()




