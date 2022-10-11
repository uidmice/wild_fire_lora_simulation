#!/bin/bash 


python train_main.py --run GNNQmix --mixer QMIX

python train_main.py --run GNNQmix --mixer NONE

python train_main.py --run BASE
