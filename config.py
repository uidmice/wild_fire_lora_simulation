import os, sys, subprocess


# IS7 = np.array([1,-8,-9,-9,-9,-9])
# IS8 = np.array([-11,1,-11,-12,-13,-13])
# IS9 = np.array([-15,-13,1,-13,-14,-15])
# IS10 = np.array([-19,-18,-17,1,-17,-18])
# IS11 = np.array([-22,-22,-21,-20,1,-20])
# IS12 = np.array([-25,-25,-25,-24,-23,1])
# IsoThresholds = np.array([IS7,IS8,IS9,IS10,IS11,IS12])

Bandwidth = 125
CodingRate = 1


graphics = 1
DEBUG = 0

DLtime = 1000 # msec
MAX_RETRY = 20

MINUTE_TO_MS = 60000 # minute to ms
UPDATA_RATE = 60 #update external environment every 1 second
BATTERY_ENERGY = 22572000 # 3.3V 1900 mAh battery

UNIT_TIME_GRASS = 1 *  MINUTE_TO_MS # ms
UNIT_TIME_SIMPY = 1 #ms
GRASS_TO_SIMPY_TIME_FACTOR = UNIT_TIME_GRASS/UNIT_TIME_SIMPY
SIMPY_TO_GRASS_TIME_FACTOR = 1/GRASS_TO_SIMPY_TIME_FACTOR


gisdb = '~/grassdata'
location = 'newLocation'
mapset = 'wf'
grass8bin = "/usr/local/bin/grass"


def env_init():
    gisbase = subprocess.check_output([grass8bin, "--config", "path"],
                                      text=True).strip()  # directory where GRASS GIS lives
    os.environ['GISBASE'] = gisbase
    path = os.getenv('LD_LIBRARY_PATH')
    dir  = os.path.join(gisbase, 'lib')
    if path:
        path = dir + os.pathsep + path
    else:
        path = dir
    os.environ['LD_LIBRARY_PATH'] = path
    os.environ['GRASS_VERBOSE'] = '-1'
    sys.path.append(os.path.join(gisbase, "etc", "python"))






