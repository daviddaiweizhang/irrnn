from time import time
import os
import pickle
from datetime import datetime
import sys
import random
import string
from PIL import Image

import numpy as np
from scipy.stats import t as tdist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd


def erint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def roc(sig_pred, sig_tru, cutoffs):
    assert sig_pred.shape == sig_tru.shape
    assert sig_tru.dtype == bool
    sig_pred = np.abs(sig_pred.astype(float))
    assert np.ndim(cutoffs) == 1
    assert (np.diff(cutoffs) > 0).all()
    assert np.min(cutoffs) >= 0
    assert np.max(cutoffs) <= 1
    V = sig_tru.shape[-1]
    num_trupos = cumtrupos(sig_tru, sig_pred)
    num_falpos = np.arange(V+1) - num_trupos
    falpos = num_falpos / (V - sig_tru.sum(axis=-1, keepdims=True))
    trupos = num_trupos / sig_tru.sum(axis=-1, keepdims=True)
    falpos = falpos[..., np.newaxis, :]
    cutoffs = np.tile(cutoffs, sig_tru.shape[:-1] + (1,))
    cutoffs = cutoffs[..., np.newaxis]
    indexes = (falpos <= cutoffs).sum(-1) - 1
    trupos = np.take_along_axis(trupos, indexes, axis=-1)
    return trupos


def clusterContiguous(coor):
    assert coor.ndim == 2
    assert np.issubdtype(coor.dtype, np.int)
    clus = np.zeros(coor.shape[0]).astype(int) - 1
    for i in range(clus.size):
        isadj = ((coor[i] - coor[i:])**2).sum(axis=-1) <= 1
        clusadj = np.unique(clus[i:][isadj])
        clusadj = clusadj[clusadj >= 0]
        if clusadj.size > 0:
            clus[np.isin(clus, clusadj)] = clusadj.min()
            clus[i:][isadj] = clusadj.min()
        else:
            clus[i:][isadj] = clus.max() + 1
    clus = np.unique(clus, return_inverse=True)[1]
    return clus


def choose(beta_fit, size):
    Q, V = beta_fit.shape
    beta_chosen = (
            np.abs(beta_fit).argsort().argsort()
            >= V - size.reshape(Q, 1))
    beta_select = beta_fit * beta_chosen
    return beta_select


def cumtrupos(sig_true, sig_pred):
    assert sig_true.shape == sig_pred.shape
    order = sig_pred.argsort()[..., ::-1]
    ctp = np.cumsum(np.take_along_axis(sig_true, order, axis=-1), axis=-1)
    header = np.zeros_like(sig_true[..., 0:1])
    ctp = np.concatenate([header, ctp], axis=-1)
    return ctp


def evaluate(x_pred, x_true, metric, alpha=0.05, decimals=3):
    if np.shape(x_true) != np.shape(x_pred):
        raise ValueError('x_true and x_pred must have the same shape')
    if isinstance(metric, list):
        eva = [evaluate(x_pred, x_true, m) for m in metric]
    else:
        if metric == 'mse':
            eva = np.mean((x_true - x_pred)**2)
        elif metric == 'rsq':
            eva = np.corrcoef(x_true.flatten(), x_pred.flatten())[0, 1]**2
        elif metric == 'pse':
            eva = evaluate(x_pred, x_true, 'mse') / (x_true**2).mean()
        elif metric == 'rse':
            eva = 1 - evaluate(x_pred, x_true, 'rsq')
        elif metric == 'sig':
            sig_true = ~np.isclose(x_true, 0)
            sig_pred = ~np.isclose(x_pred, 0)
            sig00 = np.mean((~sig_true) * (~sig_pred))
            sig01 = np.mean((~sig_true) * (sig_pred))
            sig10 = np.mean((sig_true) * (~sig_pred))
            sig11 = np.mean((sig_true) * (sig_pred))
            sig0_ = sig00 + sig01
            sig1_ = sig10 + sig11
            sig_0 = sig00 + sig10
            sig_1 = sig11 + sig01
            eva = [sig01/sig0_, sig10/sig1_, sig01/sig_1, sig10/sig_0]
        elif metric == 'falpos':
            eva = evaluate(x_pred, x_true, 'sig')[0]
        elif metric == 'trupos':
            eva = 1.0 - evaluate(x_pred, x_true, 'falneg')
        elif metric == 'falneg':
            eva = evaluate(x_pred, x_true, 'sig')[1]
        elif metric == 'adjfalneg':
            V = x_true.shape[-1]
            sig_true = ~np.isclose(x_true, 0)
            num_trupos = cumtrupos(sig_true, np.abs(x_pred))
            num_falpos = np.arange(V+1) - num_trupos
            falpos = num_falpos / (V - sig_true.sum(axis=-1, keepdims=True))
            # falneg = 1 - num_trupos / sig_true.sum(axis=-1, keepdims=True)
            controlled_size = (falpos > alpha).argmax(axis=-1) - 1
            falneg_controlled = (
                    1 - np.take_along_axis(
                        num_trupos, controlled_size.reshape(-1, 1),
                        axis=-1).sum()
                    / sig_true.sum())
            eva = falneg_controlled
        elif metric == 'auc':
            sig_true = ~np.isclose(0, x_true)
            eva = roc_auc_score(sig_true.flatten(), x_pred.flatten())
        else:
            raise ValueError('Metric not recognized')
        eva = np.round(eva, decimals).tolist()
    return eva


def fit_ls(Y, X, alpha=None):
    n, q = X.shape[-2:]
    assert Y.shape[-2] == n
    df = max(1, n - q)
    A = np.swapaxes(X, -1, -2) @ X
    if n < q:
        erint('Warning: n < q. Adding a diagnal matrix')
        evals = np.linalg.eigh(A)[0]
        emin = evals[..., ~np.isclose(evals, 0)].min(-1)
        A += np.eye(q) * emin
    A_inv = np.linalg.inv(A)
    assert (A_inv.diagonal(0, -2, -1) > 0).all()
    beta = A_inv @ np.swapaxes(X, -1, -2) @ Y
    sigmasq = ((Y - X @ beta)**2).sum(-2, keepdims=True) / n
    se = np.sqrt(sigmasq * A_inv.diagonal(0, -2, -1)[..., np.newaxis])
    pval = (1 - tdist.cdf(np.abs(beta / se), df=df))*2
    if alpha is None:
        cutoff = None
    else:
        cutoff = se * tdist.ppf(1 - alpha / 2, df=df)
    return beta, se, pval, cutoff, sigmasq


def fit_baylr(y, x, muze, sigmaze, aze=None, bze=None):
    sample_sigsq = (aze is not None) and (bze is not None)
    N, Q = x.shape[-2:]
    assert y.shape[-2] == N
    K = y.shape[-1]
    assert muze.shape[-2:] == (Q, K)
    assert sigmaze.shape[-3:] == (K, Q, Q)
    if sample_sigsq:
        assert aze.shape[-1] == K
        assert bze.shape[-1] == K
    muze = np.swapaxes(muze, -1, -2)[..., np.newaxis]
    laze = np.linalg.inv(sigmaze)
    la = np.swapaxes(x, -1, -2) @ x
    la = la[..., np.newaxis, :, :] + laze
    sigma = np.linalg.inv(la)
    yx = np.swapaxes(y, -1, -2) @ x
    yx = yx[..., np.newaxis]
    mu = sigma @ (yx + laze @ muze)
    if sample_sigsq:
        yy = (y**2).sum(-2)
        mulamuze = np.swapaxes(muze, -1, -2) @ laze @ muze
        mulamu = np.swapaxes(mu, -1, -2) @ la @ mu
        mulamuze = mulamuze[..., 0, 0]
        mulamu = mulamu[..., 0, 0]
        a = aze + 0.5 * N
        b = bze + 0.5 * (yy + mulamuze - mulamu)
    mu = np.swapaxes(mu[..., 0], -1, -2)
    if sample_sigsq:
        return mu, sigma, a, b
    else:
        return mu, sigma


def savepickle(obj, filepref='', saveid=None):
    if saveid is None:
        saveid = int(time() * 1e6)
    if filepref != '':
        filepref = filepref + '_'
    filename = 'pickles/{}{}.pickle'.format(filepref, saveid)
    os.makedirs('pickles', exist_ok=True)
    # TODO: check file exists
    with open(filename, 'wb') as infile:
        pickle.dump(obj, infile)
    erint('{} saved to {}'.format(filepref[:-1], filename))


def savefig(filepref='', saveid=None):
    if saveid is None:
        saveid = round(time())
    if filepref != '':
        filepref += '_'
    filename = 'imgs/{}{}.png'.format(filepref, saveid)
    os.makedirs('imgs', exist_ok=True)
    plt.savefig(filename, dpi=100)
    plt.close()
    erint('{} saved to {}'.format(filepref[:-1], filename))


def tabsig(sig_fit, sig_tru):
    falpos = np.logical_and(~sig_tru, sig_fit)
    falneg = np.logical_and(sig_tru, ~sig_fit)
    tab = sig_fit.astype(float)
    tab[falpos] = 0.2
    tab[falneg] = 0.8
    return tab


def gen_saveid():
    timenow = datetime.now().strftime("%Y%m%d%H%M%S")
    randstr = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
    saveid = timenow + randstr
    return saveid


def get_lattice(shape, lims):
    shape, lims = np.array(shape), np.array(lims)
    assert shape.ndim == 1
    assert lims.shape == (shape.size, 2)
    assert (lims[:, 1] > lims[:, 0]).all()
    coords = np.array(np.unravel_index(np.arange(shape.prod()), shape)).T
    for j in range(-2, -shape.size-1, -1):
        toflip = coords[:, [j]] % 2
        forward = coords[:, j+1:]
        bacward = coords[::-1, j+1:]
        coords[:, j+1:] = toflip * bacward + (1-toflip) * forward
    wids = lims[:, 1] - lims[:, 0]
    coords = (coords+1) / (shape+1) * wids + lims[:, 0]
    return coords


def alleq(y, x):
    if type(y) != type(x):
        return False
    if isinstance(x, list) or isinstance(x, tuple):
        return (
                len(y) == len(x)
                and all([alleq(ye, xe) for ye, xe in zip(y, x)]))
    elif isinstance(x, np.ndarray):
        return np.array_equal(y, x)
    else:
        return y == x


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def save_image(img, filename, normalization='negpos', cmap='RdBu_r'):

    if normalization != 'none':
        img = img.copy()
        if normalization == 'negpos':
            img /= np.nanmax(np.abs(img)) + 1e-12
            img += 1
            img *= 0.5
        elif normalization == 'zeroone':
            img -= np.nanmin(img)
            img /= np.nanmax(img) + 1e-12

        cmap = plt.get_cmap(cmap)
        img = cmap(img)[..., :3]
        img = (img * 255).astype(np.uint8)
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    print(f'Pickle loaded from {filename}')
    return x


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_text(obj, filename, append=False):
    mkdir(filename)
    mode = 'a' if append else 'w'
    with open(filename, mode) as file:
        print(obj, file=file)
    print(filename)


def load_tsv(infile):
    return pd.read_csv(infile, sep='\t', header=0, index_col=0)
