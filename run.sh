#!/bin/bash
set -e

indivs=020
voxels=128
noise=gauss
seed=00
datafile="data.pickle"
outpref="results/"

python generate_data_synthetic.py \
    --n-voxels=${voxels} \
    --n-indivs=${indivs} \
    --beta-stn=0.10 \
    --noise-dist=${noise} \
    --seed=${seed} \
    --out=${datafile}

python run_irrnn.py $datafile $outpref
