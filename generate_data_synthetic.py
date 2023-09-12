import argparse
import random

import numpy as np

from design import gen_data
from utils import save_pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-voxels', type=int, default=128)
    parser.add_argument('--n-indivs', type=int, default=20)
    parser.add_argument('--n-features', type=int, default=3)
    parser.add_argument('--beta-stn', type=float, default=0.10)
    parser.add_argument('--omega-stn', type=float, default=0.05)
    parser.add_argument('--noise-dist', type=str, default='gauss')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='data.pickle')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    set_seed(args.seed)

    img_shape = (args.n_voxels, args.n_voxels, args.n_voxels)
    data = gen_data(
            V_out=img_shape, N=args.n_indivs, Q=args.n_features,
            beta_stn=args.beta_stn, omega_stn=args.omega_stn,
            omega_itv=1.0,
            noise_dist=args.noise_dist,
            noise_var='wave', scale=1.0, cut=True)

    save_pickle(data, args.out)


if __name__ == '__main__':
    main()
