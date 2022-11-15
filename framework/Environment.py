import numpy as np
import logging, os

from .GRASS import *
from config import root

class Environment:
    def __init__(self, bound, fuel, samplefm, evi, samplevs, sampleth, dem,
                 gisbd ,
                 location,
                 mapset,
                 res=10):

        self.bound = bound
        self.fuel = fuel
        self.dem = dem
        self.samplevs = samplevs
        self.sampleth = sampleth
        self.samplefm = samplefm
        self.evi = evi
        self.varying_wind = isinstance(self.sampleth, list) or isinstance(self.samplevs, list)
        self.step_size = None
        self.res = res

        self.grass_proc = grass_init(gisbd, location, mapset)

        g.region(raster=fuel)
        g.region(s=bound.s, n=bound.n, w=bound.w, e=bound.e, res=self.res, save=REGION_SAVE_NAME, overwrite=True)
        g.region(region=REGION_SAVE_NAME, res=res)


        self.grass_r = Region()
        self.grass_r.get_current()

        self.cols = self.grass_r.cols
        self.rows = self.grass_r.rows
        if isinstance(samplevs, list):
            self.vs = [np.ones((self.rows, self.cols)) * a for a in samplevs]
            self.th = [np.ones((self.rows, self.cols)) * a for a in sampleth]
        else:
            self.vs, self.th = caldata(REGION_SAVE_NAME, GROUND_TRUTH_SUFFIX, self.res, dem=self.dem, samplefm100=samplefm,samplevs=samplevs, sampleth=sampleth, evi=evi)


        # row, col
        # dir_tmp = os.path.join(root, "data/source.txt")
        # with open(dir_tmp, "w") as f:
        #     f.write('|'.join([str(a) for a in (56978.3098189104, -12406.60548812005)]))
        # script.run_command('v.in.ascii', input=dir_tmp, output=SOURCE_NAME, overwrite=True,
        #                    columns='x double precision, y double precision', quiet=True)
        # script.run_command('v.to.rast', input=SOURCE_NAME, output=SOURCE_NAME, type='point', use='cat', overwrite=True, quiet=True)
        #
        # self.source = raster.raster2numpy(SOURCE_NAME)
        # self.source[self.source<0] = 0

        self.simulation_time = 0

    def set_source(self, cells):
        # m = raster.Buffer((self.rows, self.cols), buffer = np.zeros((self.rows, self.cols)).astype(int))
        m = np.zeros((self.rows, self.cols)).astype('int32')
        for c in cells:
            m[c[0], c[1]] = 1

        s = raster.RasterRow(SOURCE_NAME)
        s.open('w', overwrite=True)
        for i in range(self.rows):
            # s.put_row(m[i])
            s.put_row(raster.Buffer(shape=(self.cols, ), buffer=m[i]))
        # for i in range(self.rows):
        #     for j in range(self.cols):
        #         s[i, j] = 0
        # for c in cells:
        #     m[c[0], c[1]] = 1
        s.close()
        self.source = raster.raster2numpy(SOURCE_NAME)
        self.source[self.source < 0] = 0


    def propogate(self, source, init_time, lag, suffix=GROUND_TRUTH_SUFFIX, ros_out='gt_out', spread_out='gt_spread',
                  spotting=False, middle_state=None, sv_suffix=None, th_suffix=None):

        calculate_ros(suffix, ros_out, sv_suffix=sv_suffix, th_suffix=th_suffix)
        calculate_spread(ros_out, suffix, source, spread_out, init_time=init_time, lag=lag, spotting=spotting, sv_suffix=sv_suffix)
        c = raster.raster2numpy(spread_out)
        if not middle_state:
            pre = np.where(c+ self.source> 0, 1, 0)
        else:
            try:
                pre = [np.where((c + self.source> 0) & (c<= a), 1, 0) for a in middle_state]
            except:
                raise ValueError(f"middle_state should be a list, but is {middle_state} instead")
        return pre, {'base': raster.raster2numpy(ros_out+'.base'), 'max': raster.raster2numpy(ros_out+'.max'), 'dir': raster.raster2numpy(ros_out+'.dir'), 'spotting': raster.raster2numpy(ros_out+'.spotting')}


    def generate_wildfire(self, lag, spotting=False):
        out_name = 'gt_spread'+'_'+str(lag)
        self.vs, self.th = caldata(REGION_SAVE_NAME, GROUND_TRUTH_SUFFIX, self.res, dem=self.dem, samplefm100=self.samplefm,
                                   samplevs=self.samplevs, sampleth=self.sampleth, evi=self.evi)

        self.propogate(SOURCE_NAME, 0, lag, spread_out=out_name, spotting=spotting)
        self.ground_truth = raster.raster2numpy(out_name).astype(float) + self.source
        self.simulation_time = lag
        return self.ground_truth

    def generate_wildfire_alternate(self, lag, step_size, spotting=False):
        t = 0
        i = 0
        self.step_size = step_size
        self.vs, self.th = caldata(REGION_SAVE_NAME, GROUND_TRUTH_SUFFIX, self.res, dem=self.dem, samplefm100=self.samplefm,
                                   samplevs=self.samplevs, sampleth=self.sampleth, evi=self.evi)
        while t < lag:
            nt = t + step_size
            nt = min(nt, lag)
            out_name = 'gt_spread'
            vs = i%len(self.samplevs)
            th = i%len(self.sampleth)
            # print(f't={t}, nt={nt}, outname={out_name}')
            if t == 0:
                self.propogate(SOURCE_NAME, t, step_size, suffix=GROUND_TRUTH_SUFFIX, spread_out=out_name,
                               spotting=spotting, sv_suffix=GROUND_TRUTH_SUFFIX + str(vs),
                               th_suffix=GROUND_TRUTH_SUFFIX + str(th))
            else:
                self.propogate(out_name, t, step_size, suffix=GROUND_TRUTH_SUFFIX,
                               spread_out=out_name, spotting=spotting,
                               sv_suffix=GROUND_TRUTH_SUFFIX + str(vs), th_suffix=GROUND_TRUTH_SUFFIX + str(th))
            # print(np.max(raster.raster2numpy(out_name)))
            t = nt
            i += 1
        self.ground_truth = raster.raster2numpy(out_name).astype(float) + self.source
        self.simulation_time = lag
        return self.ground_truth

    def print_region(self):
        print(g.region(flags='p'))

    def end_grass(self):
        os.remove(self.grass_proc)

    def sense(self, row, col, time):
        if self.varying_wind:
            return {'vs': self.vs[(time//self.step_size)%len(self.samplevs)][row, col], 'th': self.th[(time//self.step_size)%len(self.samplevs)][row, col], 'fire': self.ground_truth[row, col] < time}
        return {'vs': self.vs[row, col], 'th': self.th[row, col], 'fire': self.ground_truth[row, col] < time}

    def sense_region(self, row, col, mask, time):
        firing = np.where ((self.ground_truth <= time) & (self.ground_truth > 0), 1, 0)
        masked_result = firing * mask
        if self.varying_wind:
            r = {'vs': self.vs[(time//self.step_size)%len(self.samplevs)][row, col], 'th': self.th[(time//self.step_size)%len(self.sampleth)][row, col],
                    'fire_area': np.sum(masked_result), 'firing': masked_result}
            return r
        return {'vs': self.vs[row, col], 'th': self.th[row, col], 'fire_area': np.sum(masked_result), 'firing': masked_result}

    def get_on_fire(self, time):
        if time > self.simulation_time:
            print(f'Wildfire data is only generated until {self.simulation_time}, but get_on_fire is called for time {time}.')
        return np.where((self.ground_truth <= time) & (self.ground_truth > 0), 1, 0)

