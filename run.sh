#!/bin/bash 

python ray_runner.py 

python ray_runner.py --one-hot-encoding

python ray_runner.py --wind-in-state 

python ray_runner.py --one-hot-encoding --wind-in-state

python ray_runner.py --run RANDOM  --random-prob 0.1 --suffix 10 

python ray_runner.py --run RANDOM  --random-prob 0.05 --suffix 05 

python ray_runner.py --run HEURISTIC


python ray_runner.py --wind-in-state --alternating-wind 10 --enable-store

python ray_runner.py --one-hot-encoding --wind-in-state --alternating-wind 10 --enable-store

python ray_runner.py  --alternating-wind 10 --enable-store

python ray_runner.py --one-hot-encoding  --alternating-wind 10 --enable-store

python ray_runner.py --run RANDOM  --random-prob 0.1 --suffix 10 --alternating-wind 10 --enable-store

python ray_runner.py --run RANDOM  --random-prob 0.05 --suffix 05 --alternating-wind 10 --enable-store

python ray_runner.py --run HEURISTIC  --alternating-wind 10  --enable-store



