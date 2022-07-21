import numpy as np

from .GRASS import *

class Environment:
    def __init__(self, bound, fuel, samplefm, evi, samplevs, sampleth, dem, source,
                 gisbd ,
                 location,
                 mapset,
                 res=10):

        self.bound = bound
        self.fuel = fuel
        self.dem = dem

        self.res = res

        self.grass_proc = grass_init(gisbd, location, mapset)

        g.region(raster=fuel)
        g.region(s=bound.s, n=bound.n, w=bound.w, e=bound.e, res=self.res, save=REGION_SAVE_NAME, overwrite=True)
        self.print_region()

        caldata(REGION_SAVE_NAME, GROUND_TRUTH_SUFFIX, self.res, dem=self.dem, samplefm100=samplefm,samplevs=samplevs, sampleth=sampleth, evi=evi)

        self.grass_r = Region()
        self.grass_r.get_current()

        self.cols = self.grass_r.cols
        self.rows = self.grass_r.rows

        # row, col

        with open("data/source.txt", "w") as f:
            f.write('|'.join([str(a) for a in source]))
        script.run_command('v.in.ascii', input='data/source.txt', output=SOURCE_NAME, overwrite=True,
                           columns='x double precision, y double precision')
        script.run_command('v.to.rast', input=SOURCE_NAME, output=SOURCE_NAME, type='point', use='cat', overwrite=True)
        self.source = raster.raster2numpy(SOURCE_NAME)
        self.source[self.source<0] = 0


        self.vs = raster.raster2numpy(WIND_SPEED+GROUND_TRUTH_SUFFIX)
        self.th = raster.raster2numpy(WIND_DIR+GROUND_TRUTH_SUFFIX)
        #

        self.simulation_time = 0


    def propogate(self, source, init_time, lag, suffix=GROUND_TRUTH_SUFFIX, ros_out='gt_out', spread_out='gt_spread', spotting=False, middle_state=None):
        calculate_ros(suffix, ros_out)
        calculate_spread(ros_out, suffix, source, spread_out, init_time=init_time, lag=lag, spotting=spotting)
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
        self.propogate(SOURCE_NAME, 0, lag, spread_out=out_name, spotting=spotting)
        self.ground_truth = raster.raster2numpy(out_name).astype(float) + self.source
        self.simulation_time = lag
        return self.ground_truth

    def print_region(self):
        print(g.region(flags='p'))

    def end_grass(self):
        os.remove(self.grass_proc)

    def sense(self, row, col, time):
        return {'vs': self.vs[row, col], 'th': self.th[row, col], 'fire': self.ground_truth[row, col] < time}

    def sense_region(self, row, col, mask, time):
        firing = np.where ((self.ground_truth <= time) & (self.ground_truth > 0), 1, 0)
        masked_result = firing * mask
        return {'vs': self.vs[row, col], 'th': self.th[row, col], 'fire_area': np.sum(masked_result)}

    def get_on_fire(self, time):
        flag = np.where(self.ground_truth > 0, self.ground_truth, np.Inf)
        if time > self.simulation_time:
            print(f'Wildfire data is only generated until {self.simulation_time}, but get_on_fire is called for time {time}.')
        return np.where(flag < time, 1, 0)

