import argparse
import random
from time import time

import numpy as np
import torch

from utils import write_text, load_pickle, save_pickle
from irrnn import fit_irrnn, get_coord


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--activation', type=str, default='leaky')
    parser.add_argument('--alpha-threshold', type=float, default=0.05)
    parser.add_argument('--n-permute', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=2)
    parser.add_argument('--n-states', type=int, default=11)
    parser.add_argument('--alpha-states', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-m', '--message', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.seed is not None:
        set_seed(args.seed)

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    write_text(args, args.prefix+'notes.txt', append=True)

    # generate data
    data = load_pickle(args.data)
    datatype = 'synthetic'
    if datatype == 'synthetic':
        y, x = data[:2]
        img_shape = y.shape[1:]
        y = y.reshape(y.shape[0], -1)

        use_free_coords = False
        if use_free_coords:
            s = get_coord(img_shape)
            img_shape = None
        else:
            s = None
    elif datatype == 'cortex':
        y = data['imgs']
        x = data['covariates']
        s = data['coordinates']
        # x = np.concatenate([x, np.ones_like(x[:, :1])], -1)
        img_shape = None
    else:
        y = data['response']
        x = data['covariate']
        s = data['coordinate']
        img_shape = data['img_shape']

    # training parameters
    n_voxels = y.shape[-1]
    batch_size = min(args.batch_size, n_voxels // 16)
    hidden_widths = (args.width,) * args.depth

    print('epochs:', args.epochs)
    print('batch_size:', batch_size)

    print('Fitting irrnn...')
    t0 = time()
    pred = fit_irrnn(
            x=x, y=y, s=s, img_shape=img_shape,
            hidden_widths=hidden_widths,
            activation=args.activation,
            alpha_threshold=args.alpha_threshold,
            n_permute=args.n_permute, lr=args.lr,
            epochs=args.epochs, batch_size=batch_size,
            max_iter=args.max_iter, n_states=args.n_states,
            alpha_states=args.alpha_states,
            prefix=args.prefix, device=device, n_jobs=args.n_jobs)
    print(int(time() - t0), 'sec')

    save_pickle(pred, f'{args.prefix}pred.pickle')
    truth = {
            'x': x, 'y': y, 's': s, 'img_shape': img_shape}
    if datatype == 'synthetic':
        maineff, indiveff, noisevar = data[2:]
        maineff = maineff.reshape(maineff.shape[0], -1)
        indiveff = indiveff.reshape(indiveff.shape[0], -1)
        noisevar = noisevar.reshape(noisevar.shape[0], -1)
        truth['maineff'] = maineff
        truth['indiveff'] = indiveff
        truth['noiselogvar'] = np.log(noisevar)
    save_pickle(truth, f'{args.prefix}true.pickle')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
