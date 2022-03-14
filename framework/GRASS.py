import subprocess
import os

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
    cfile = gsetup.init(gisdb, location, mapset)

    gs.message("Current GRASS GIS 8 environment:")
    print(gs.gisenv())

    return cfile


def caldata(regname, suffix, res, dem='dem', samplefm100='samplefm100',samplevs='samplevs', sampleth='sampleth', evi='evi'):
    try:
        g.region(region=regname, res=res)

        script.run_command('r.slope.aspect', elevation=dem, slope='slope' + suffix, aspect='aspect' + suffix, overwrite=True)
        script.run_command('v.surf.idw', input=samplefm100, output='moisture_100h' + suffix, column='mean', overwrite=True)
        script.run_command('v.surf.idw', input=samplevs, output= WIND_SPEED+ suffix, column='vsfpm', overwrite=True)
        script.run_command('v.surf.idw', input=sampleth, output= WIND_DIR+ suffix, column='mean', overwrite=True)
        ss1 = 'moisture_1h'
        lfm = 'lfm'

        expm = ss1 + suffix + '=' + 'moisture_100h' + suffix + '-2'
        r.mapcalc(expression=expm, overwrite=True)

        # estimating live fuel moisture from evi
        explfm = lfm + suffix + '=(417.602 * '+evi+') + 6.78061'
        r.mapcalc(expression=explfm, overwrite=True)

        # rescale LFM to 0-100
        output = lfm + suffix + '_scaled'
        r.rescale(input='lfm' + suffix, output=output, to=(0, 100), overwrite=True)

        return "successfully calculated"

    except:
        print("Something went wrong")


def calculate_ros(suffix, output_name):
    # generate rate of spread raster map
    try:
        r.ros(model='fuel', moisture_1h='moisture_1h'+suffix,
            moisture_live='lfm'+suffix+'_scaled', velocity=WIND_SPEED+suffix,
            direction=WIND_DIR+suffix, slope='slope'+suffix, aspect='aspect'+suffix,
            elevation='dem', base_ros=output_name+'.base',
            max_ros=output_name+'.max', direction_ros=output_name+'.dir',
            spotting_distance=output_name+'.spotting', overwrite=True)
        return 'ROS successfully calculated'
    except:
        print("Something went wrong")


def calculate_spread(input_name, suffix, source, output_name, init_time=0, lag=60, spotting=False):
    # generate rate of spread raster map
    f = ''
    if spotting:
        f += 's'
    if init_time:
        f = 'i' + f
        script.run_command('r.spread', flags=f, base_ros=input_name+'.base', max_ros=input_name+'.max',
            direction_ros=input_name+'.dir', start=source,
            spotting_distance=input_name+'.spotting', wind_speed=WIND_SPEED+suffix,
            fuel_moisture='moisture_1h'+suffix, output=output_name, init_time=init_time, lag=lag, overwrite=True)
    else:
        script.run_command('r.spread', flags=f, base_ros=input_name+'.base', max_ros=input_name+'.max',
            direction_ros=input_name+'.dir', start=source,
            spotting_distance=input_name+'.spotting', wind_speed=WIND_SPEED+suffix,
            fuel_moisture='moisture_1h'+suffix, output=output_name,  lag=lag, overwrite=True)

    # r.null(map=output_name, setnull=0)
    #
    # r.colors(map=output_name, rules='data/img/fire_colors.txt')

def save_raster(raster, out_path, format='GTiff'):
    cwd = os.getcwd()
    gs.run_command('r.out.gdal', input=raster, output=os.path.join(cwd, out_path+'.tiff'), format=format, overwrite=True)
