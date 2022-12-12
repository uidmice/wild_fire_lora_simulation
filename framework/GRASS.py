import os, sys

# import grass gis package
import grass.script as gs
import grass.script.setup as gsetup
from grass import script
from grass.pygrass.modules.shortcuts import raster as r, vector as v, general as g, display as d
from grass.pygrass.gis.region import Region
from grass.pygrass import raster, vector
from grass.pygrass.vector.geometry import Point

WIND_SPEED = 'wind_speed'
WIND_DIR = 'wind_dir'
GROUND_TRUTH_SUFFIX='_gt'
PREDICTION_SUFFIX = '_pre'
REGION_SAVE_NAME='current_r'
SOURCE_NAME = 'source'


def grass_init(gisdb, location, mapset):
    gsetup.init(gisdb, location, "PERMANENT")

    try:
        g.mapset(flags='c', mapset=mapset, location=location)
    except:
        print('mapset error')

    cfile = gsetup.init(gisdb, location, mapset)

    gs.message("Current GRASS GIS 8 environment:")
    print(gs.gisenv())
    script.run_command('g.gisenv', set="DEBUG=0")
    print(gs.verbosity())

    return cfile

def uniformMap(regname, value, output_name, res):
    g.region(region=regname, res=res)
    cur_region = Region()
    cur_region.get_current()

    new = vector.Vector('vsuniform')
    cols = [('cat',       'INTEGER PRIMARY KEY'),  ('value',      'double precision')]
    new.open('w', tab_name='vsuniform', tab_cols=cols, overwrite=True)
    p = Point((cur_region.east + cur_region.west)/2,(cur_region.south + cur_region.north)/2 )
    new.write(p, cat=1, attrs=(value,))
    new.table.conn.commit()
    new.close()
    script.run_command('v.surf.idw', input='vsuniform', output=output_name, column='value', overwrite=True,
                       quiet=True)



def caldata_wind_uneven(regname, suffix, res, locations, samplevs, sampleth, logdir):
    g.region(region=regname, res=res)

    x, y = [loc[0] for loc in locations], [loc[1] for loc in locations]

    with open(os.path.join(logdir, "wind.txt"), "w") as f:
        for c in zip(y, x, samplevs, sampleth):
            f.write('|'.join([str(a) for a in c]) + '\n')

    script.run_command('v.in.ascii', input=os.path.join(logdir, "wind.txt"), output='wind', overwrite=True,
                       columns='x double precision, y double precision, vsfpm double precision, mean double precision',
                       quiet=True)
    vs, th = caldata_wind(regname, suffix, res, samplevs='wind',sampleth='wind')
    return vs, th

def caldata_wind(regname, suffix, res, samplevs='samplevs', sampleth='sampleth'):
    g.region(region=regname, res=res)

    if isinstance(samplevs, (int, float)):
        uniformMap(regname, samplevs, WIND_SPEED + suffix, res)
        vs = raster.raster2numpy(WIND_SPEED + suffix)
    elif isinstance(samplevs, list):
        vs = []
        for i in range(len(samplevs)):
            uniformMap(regname, samplevs[i], WIND_SPEED + suffix + str(i), res)
            vs.append(raster.raster2numpy(WIND_SPEED + suffix + str(i)))
    else:
        script.run_command('v.surf.idw', input=samplevs, output=WIND_SPEED + suffix, column='vsfpm', overwrite=True,
                           quiet=True)
        vs = raster.raster2numpy(WIND_SPEED + suffix)

    if isinstance(sampleth, (int, float)):
        uniformMap(regname, sampleth, WIND_DIR + suffix, res)
        th = raster.raster2numpy(WIND_DIR + suffix)
    elif isinstance(sampleth, list):
        th = []
        for i in range(len(sampleth)):
            uniformMap(regname, sampleth[i], WIND_DIR + suffix + str(i), res)
            th.append(raster.raster2numpy(WIND_DIR + suffix + str(i)))
    else:
        script.run_command('v.surf.idw', input=sampleth, output=WIND_DIR + suffix, column='mean', overwrite=True,
                           quiet=True)
        th = raster.raster2numpy(WIND_DIR + suffix)

    return vs, th


def caldata_base(regname, suffix, res, dem='dem', samplefm100='samplefm100', evi='evi'):
    try:
        g.region(region=regname, res=res)

        script.run_command('r.slope.aspect', elevation=dem, slope='slope' + suffix, aspect='aspect' + suffix, overwrite=True, quiet = True)
        script.run_command('v.surf.idw', input=samplefm100, output='moisture_100h' + suffix, column='mean', overwrite=True, quiet = True)

        ss1 = 'moisture_1h'
        lfm = 'lfm'

        expm = ss1 + suffix + '=' + 'moisture_100h' + suffix + '-2'
        r.mapcalc(expression=expm, overwrite=True, quiet = True)

        # estimating live fuel moisture from evi
        explfm = lfm + suffix + '=(417.602 * '+evi+') + 6.78061'
        r.mapcalc(expression=explfm, overwrite=True, quiet = True)

        # rescale LFM to 0-100
        output = lfm + suffix + '_scaled'
        r.rescale(input='lfm' + suffix, output=output, to=(0, 100), overwrite=True, quiet = True)

    except:
        print("Something went wrong")



def calculate_ros(suffix, output_name, sv_suffix=None, th_suffix=None):
    # generate rate of spread raster map
    if not sv_suffix:
        sv_suffix = suffix
    if not th_suffix:
        th_suffix = suffix
    try:
        r.ros(model='fuel', moisture_1h='moisture_1h'+GROUND_TRUTH_SUFFIX,
            moisture_live='lfm'+GROUND_TRUTH_SUFFIX+'_scaled', velocity=WIND_SPEED+sv_suffix,
            direction=WIND_DIR+th_suffix, slope='slope'+GROUND_TRUTH_SUFFIX, aspect='aspect'+GROUND_TRUTH_SUFFIX,
            elevation='dem', base_ros=output_name+'.base',
            max_ros=output_name+'.max', direction_ros=output_name+'.dir',
            spotting_distance=output_name+'.spotting', overwrite=True, quiet = True)
        return 'ROS successfully calculated'
    except:
        print("Something went wrong")


def calculate_spread(input_name, suffix, source, output_name, init_time=0, lag=60, spotting=False, sv_suffix=None):
    # generate rate of spread raster map
    f = ''
    if not sv_suffix:
        sv_suffix = suffix

    if spotting:
        f += 's'
    if init_time:
        f = 'i' + f
        script.run_command('r.spread', flags=f, base_ros=input_name+'.base', max_ros=input_name+'.max',
            direction_ros=input_name+'.dir', start=source,
            spotting_distance=input_name+'.spotting', wind_speed=WIND_SPEED+sv_suffix,
            fuel_moisture='moisture_1h'+GROUND_TRUTH_SUFFIX, output=output_name, init_time=init_time, lag=lag, overwrite=True, quiet = True)
    else:
        script.run_command('r.spread', flags=f, base_ros=input_name+'.base', max_ros=input_name+'.max',
            direction_ros=input_name+'.dir', start=source,
            spotting_distance=input_name+'.spotting', wind_speed=WIND_SPEED+suffix,
            fuel_moisture='moisture_1h'+GROUND_TRUTH_SUFFIX, output=output_name,  lag=lag, overwrite=True, quiet = True)

    # r.null(map=output_name, setnull=0)
    #
    # r.colors(map=output_name, rules='data/img/fire_colors.txt')

def save_raster(raster, out_path, format='GTiff'):
    cwd = os.getcwd()
    gs.run_command('r.out.gdal', input=raster, output=os.path.join(cwd, out_path+'.tiff'), format=format, overwrite=True)
