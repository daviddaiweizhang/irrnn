#!/bin/bash
set -e

datafile="demo/data.pickle"
outpref="demo/results/"

python generate_data_synthetic.py \
    --n-voxels=128 \
    --n-indivs=20 \
    --beta-stn=0.1 \
    --noise-dist=gauss \
    --out=${datafile}

python run_irrnn.py ${datafile} ${outpref}
