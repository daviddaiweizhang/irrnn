import multiprocessing

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Adam
import pytorch_lightning as pl

from utils import save_image, save_pickle
from train import get_model as train_load_model


class SemiHardshrink(nn.Module):

    def __init__(self, lambd, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.activation = nn.Hardshrink(lambd)

    def forward(self, x):
        y = self.activation(x)
        if self.alpha is not None and self.training:
            y = x * self.alpha + y * (1-self.alpha)
        return y


class MLP(pl.LightningModule):

    def __init__(self, widths, activation, lr, shrink=False, l2pen=None):
        activation_dict = {
                'relu': nn.ReLU(inplace=True),
                'leaky': nn.LeakyReLU(0.1, inplace=True),
                'leaky0100': nn.LeakyReLU(0.100, inplace=True),
                'leaky0010': nn.LeakyReLU(0.010, inplace=True),
                'swish': nn.SiLU(inplace=True),
                'sigmoid': nn.Sigmoid(),
                }
        activation_func = activation_dict[activation]
        super().__init__()
        layers = []
        for i in range(len(widths)-1):
            n_inp, n_out = widths[i:i+2]
            layers.append(nn.Linear(n_inp, n_out))
            if i < len(widths)-2:
                layers.append(activation_func)
        self.net = nn.Sequential(*layers)
        if shrink:
            self.shrinker = SemiHardshrink(lambd=1.0, alpha=0.1)
        else:
            self.shrinker = None
        self.lr = lr
        self.l2pen = l2pen
        self.save_hyperparameters()

    def forward(self, x, threshold=None):
        x = self.net(x)
        if threshold is not None:
            x = x / threshold
            x = self.shrinker(x)
            x = x * threshold
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, torch.Tensor):
            threshold = None
        else:
            x, threshold = x
        y_pred = self.forward(x, threshold)
        mae = (torch.abs(y_pred - y)).mean()
        loss = mae
        if self.l2pen is not None:
            l2 = (y_pred**2).mean()
            loss += l2 * self.l2pen / x.shape[0]
            self.log('l2', l2, prog_bar=True)
        # l1 = torch.abs(y_pred).mean()
        # loss = mae + l1 * 0.5
        self.log('mae', mae, prog_bar=True)
        # self.log('l1', l1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class ParalleleNet(pl.LightningModule):

    def __init__(self, widths, lr, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList(
                [MLP(widths=wid, lr=lr, **kwargs) for wid in widths])
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x, threshold=None):
        y = [
                net(x, threshold[..., [i]])
                for i, net, in enumerate(self.nets)]
        y = torch.cat(y, -1)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, torch.Tensor):
            threshold = None
        else:
            x, threshold = x
        y_pred = self.forward(x, threshold)
        mae = (torch.abs(y_pred - y)).mean()
        loss = mae
        self.log('mae', mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class ImageDataset(Dataset):

    def __init__(self, value, coord=None, img_shape=None, threshold=None):
        super().__init__()
        if coord is None:
            coord = get_coord(img_shape)
        self.coord = coord.astype(np.float32)
        self.value = value
        if threshold is None:
            self.threshold = None
        else:
            self.threshold = threshold.reshape(threshold.shape[0], -1).T

    def __len__(self):
        return self.coord.shape[0]

    def __getitem__(self, idx):
        if self.threshold is None:
            batch = self.coord[idx], self.value[idx]
        else:
            batch = (self.coord[idx], self.threshold[idx]), self.value[idx]
        return batch


def fit(
        value,
        hidden_widths, activation, lr, batch_size,
        epochs, prefix,
        coord=None, img_shape=None,
        threshold=None, l2pen=None, device=None):

    value = value.copy()

    if coord is not None:
        coord = normalize_coordinate(coord)

    dataset = ImageDataset(
            coord=coord, value=value, img_shape=img_shape,
            threshold=threshold)
    if coord is None:
        img_ndim = len(img_shape)
    else:
        img_ndim = coord.shape[-1]
    widths = (img_ndim,) + hidden_widths + (1,)
    widths = (widths,) * value.shape[-1]
    model = train_load_model(
            model_class=ParalleleNet,
            model_kwargs=dict(
                shrink=(threshold is not None),
                widths=widths,
                activation=activation,
                l2pen=l2pen,
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs, device=device)
    model.eval()

    model.to(device)
    batch_size = 65536
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    coord_batches = np.array_split(dataset.coord, n_batches)
    coord_batches = [
            torch.tensor(e, device=model.device)
            for e in coord_batches]
    if threshold is None:
        threshold_batches = [None] * n_batches
    else:
        threshold_batches = np.array_split(dataset.threshold, n_batches)
        threshold_batches = [
                torch.tensor(e, device=model.device)
                for e in threshold_batches]
    img_pred = np.concatenate([
            model.forward(coo, thr)
            .cpu().detach().numpy()
            for coo, thr in zip(coord_batches, threshold_batches)])
    return img_pred


def get_coord(shape):
    coord = [np.linspace(0, 1, n) for n in shape]
    coord = np.stack(np.meshgrid(*coord, indexing='ij'), -1)
    coord = coord.reshape(-1, coord.shape[-1])
    return coord


def get_ols_est(x, y):
    xtx = np.swapaxes(x, -1, -2) @ x
    xtx_inv = np.linalg.inv(xtx)
    xty = np.swapaxes(x, -1, -2) @ y
    est = xtx_inv @ xty
    return est, xtx_inv


def get_ols_var(xtx_inv, noise_var):
    a = xtx_inv.diagonal(0, -2, -1)[..., np.newaxis]
    eff_var = noise_var * a
    return eff_var


def compute_maineff(
        indiveff, noiselogvar, x, y, s, img_shape,
        hidden_widths, activation, lr, batch_size, epochs,
        prefix_model, prefix_image,
        shrink=True, threshold=None, visual=True, device=None):
    n_observations, n_features = x.shape
    maineff_obsr, xtx_inv = get_ols_est(x, (y-indiveff))

    if shrink:
        if threshold is None:
            threshold = np.median(np.exp(noiselogvar)**0.5 * 1.96)
            threshold = np.full_like(maineff_obsr, threshold)
    else:
        threshold = None
    maineff_pred = fit(
            value=maineff_obsr.T, coord=s, img_shape=img_shape,
            hidden_widths=hidden_widths,
            activation=activation,
            lr=lr, batch_size=batch_size,
            epochs=epochs,
            prefix=prefix_model,
            threshold=threshold,
            device=device).T
    if visual and s is None:
        i_slice = img_shape[-1] // 2
        for i_feature in range(3):
            n_covariates = maineff_obsr.shape[0]
            maineff_obsr = maineff_obsr.reshape(
                    (n_covariates,) + img_shape)
            maineff_pred = maineff_pred.reshape(
                    (n_covariates,) + img_shape)
            save_image(
                    maineff_obsr[i_feature, ..., i_slice],
                    f'{prefix_image}obsr-{i_feature}.png')
            save_image(
                    maineff_pred[i_feature, ..., i_slice],
                    f'{prefix_image}pred-{i_feature}.png')
    return maineff_pred


def compute_noiselogvar(
        maineff, indiveff, x, y,
        hidden_widths, lr, batch_size, epochs, prefix,
        visual=True, device=None):
    n_observations, n_features = x.shape
    img_shape = y.shape[1:]
    explained = (
            (x @ maineff.reshape(n_features, -1))
            .reshape((n_observations,) + img_shape))
    noiselogvar_obsr = np.log(
            np.square(y - explained - indiveff)
            .mean(0, keepdims=True))
    noiselogvar_pred = fit(
            noiselogvar_obsr,
            hidden_widths=hidden_widths,
            lr=lr, batch_size=batch_size,
            epochs=epochs,
            prefix=prefix,
            threshold=None,
            device=device)
    if visual:
        i_slice = y.shape[-1] // 2
        save_image(noiselogvar_obsr[0, ..., i_slice], f'{prefix}obsr.png')
        save_image(noiselogvar_pred[0, ..., i_slice], f'{prefix}pred.png')
    return noiselogvar_pred


def compute_indiveff(
        maineff, noiselogvar, x, y,
        hidden_widths, lr, batch_size, epochs, prefix, visual=True,
        device=None):
    n_observations, n_features = x.shape
    img_shape = y.shape[1:]
    y_explained = x @ maineff.reshape(maineff.shape[0], -1)
    y_explained = y_explained.reshape(y_explained.shape[:1] + img_shape)
    indiveff_obsr = y - y_explained
    indiveff_pred = fit(
            indiveff_obsr,
            hidden_widths=hidden_widths,
            lr=lr, batch_size=batch_size,
            epochs=epochs,
            prefix=prefix,
            threshold=None,
            l2pen=10.0,
            device=device)
    if visual:
        i_slice = y.shape[-1] // 2
        for i_feature in range(3):
            save_image(
                    indiveff_obsr[i_feature, ..., i_slice],
                    f'{prefix}obsr-{i_feature}.png')
            save_image(
                    indiveff_pred[i_feature, ..., i_slice],
                    f'{prefix}pred-{i_feature}.png')
    return indiveff_pred


def get_gaussian_kernel(std):

    def gaussian_kernel(d):
        wt = np.exp((-0.5) * (d/std)**2)
        return wt

    return gaussian_kernel


def fit_smooth(value, filter_size, coord=None, img_shape=None):
    if coord is None:
        assert img_shape is not None
        value = value.T.reshape((-1,) + img_shape)
        std = 12**(-0.5)  # standard deviation of uniform distribution
        filter_size = np.repeat(std, len(img_shape)) * filter_size
        smoothed = [
                gaussian_filter(e, filter_size).flatten() for e in value]
        smoothed = np.stack(smoothed, -1)
    else:
        shape = coord.shape[:-1]
        assert value.shape[:-1] == shape
        coord = normalize_coordinate(coord)
        coord = coord.reshape(-1, coord.shape[-1])
        value = value.reshape(-1, value.shape[-1])
        truncate = 4
        n_voxels = coord.shape[0]
        n_neighbors = n_voxels * np.prod(filter_size * truncate)
        n_neighbors = np.round(n_neighbors).astype(int)
        n_neighbors = np.clip(n_neighbors, 10, 30)
        filter_size = coord.std(0) * filter_size
        filter_size = (filter_size**2).mean()**0.5
        kernel = get_gaussian_kernel(filter_size)
        model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=kernel)
        model.fit(X=coord, y=value)
        smoothed = model.predict(coord)
    return smoothed


def compute_threshold(
        x, y, alpha,
        hidden_widths, activation, lr, batch_size, epochs, prefix,
        s=None, img_shape=None,
        n_permute=10, fast_approximation=False):

    n_observations = x.shape[0]
    maineff_null_list = []
    for i in range(n_permute):
        order = np.random.choice(
                np.arange(n_observations), n_observations,
                replace=False)
        x_shuffled = x[order]
        if fast_approximation:
            maineff_null, __ = get_ols_est(x_shuffled, y)
            if s is None:
                img_ndim = len(img_shape)
            else:
                img_ndim = s.shape[-1]
            filter_size = (0.01,) * img_ndim
            maineff_null = fit_smooth(
                    value=maineff_null.T,
                    coord=s, img_shape=img_shape,
                    filter_size=filter_size).T
        else:
            maineff_null = compute_maineff(
                    indiveff=np.zeros_like(y),
                    noiselogvar=None,
                    x=x_shuffled, y=y, s=s, img_shape=img_shape,
                    hidden_widths=hidden_widths,
                    lr=lr, batch_size=batch_size,
                    epochs=epochs,
                    prefix=f'{prefix}sample{i:02d}-',
                    shrink=False)
        maineff_null_list.append(maineff_null)

    maineff_null = np.stack(maineff_null_list, 1)
    threshold = np.quantile(np.abs(maineff_null), 1-alpha, (1, 2))
    print('threshold:', threshold)
    threshold = np.full_like(maineff_null_list[0].T, threshold).T

    return threshold


def compute_single_iter(
        maineff, indiveff, noiselogvar, threshold, x, y, s, img_shape,
        hidden_widths, activation, lr, batch_size, epochs,
        prefix_model, prefix_image, device=None):

    maineff = compute_maineff(
            indiveff=indiveff,
            noiselogvar=noiselogvar,
            x=x, y=y, s=s, img_shape=img_shape,
            hidden_widths=hidden_widths,
            activation=activation,
            lr=lr, batch_size=batch_size,
            epochs=epochs,
            prefix_model=f'{prefix_model}maineff-',
            prefix_image=f'{prefix_image}maineff-',
            shrink=True,
            threshold=threshold,
            device=device)

    indiveff = compute_indiveff(
            maineff=maineff,
            noiselogvar=noiselogvar,
            x=x, y=y, s=s,
            hidden_widths=tuple([e//4 for e in hidden_widths[::2]]),
            activation=activation,
            lr=lr, batch_size=batch_size,
            epochs=(epochs+1)//2,
            prefix=f'{prefix_image}indiveff-', device=device)

    noiselogvar = compute_noiselogvar(
            maineff=maineff, indiveff=indiveff,
            x=x, y=y, s=s,
            hidden_widths=hidden_widths,
            activation=activation,
            lr=lr, batch_size=batch_size,
            epochs=(epochs+4)//5,
            prefix=f'{prefix_image}noiselogvar-', device=device)

    pred = dict(
            maineff=maineff, indiveff=indiveff, noiselogvar=noiselogvar,
            s=s, img_shape=img_shape)
    save_pickle(pred, f'{prefix_model}pred.pickle')

    return pred


def compute_multi_iter(
        maineff, indiveff, noiselogvar, threshold,
        x, y, s, img_shape,
        hidden_widths, activation, lr, batch_size, epochs,
        max_iter, prefix_model, prefix_image, device=None):

    for i_iter in range(max_iter):
        prefix_model += 'iter{i_iter:02d}-'
        prefix_model += 'iter{i_iter:02d}-'
        pred = compute_single_iter(
                maineff=maineff, indiveff=indiveff,
                noiselogvar=noiselogvar, threshold=threshold,
                x=x, y=y, s=s, img_shape=img_shape,
                hidden_widths=hidden_widths, activation=activation,
                lr=lr, batch_size=batch_size, epochs=epochs,
                prefix_model=prefix_model, prefix_image=prefix_image,
                device=device)
        maineff = pred['maineff']
        indiveff = pred['indiveff']
        noiselogvar = pred['noiselogvar']
    return pred


def compute_multi_iter_kwargs(kwargs):
    return compute_multi_iter(**kwargs)


def combine_states(pred_list, alpha=0.5):
    out = {}
    for key in ['maineff', 'indiveff', 'noiselogvar']:
        x_list = [e[key] for e in pred_list]
        x = np.mean(x_list, 0)
        prop_pos = np.mean([e > 0 for e in x_list], 0)
        prop_neg = np.mean([e < 0 for e in x_list], 0)
        sign = np.zeros(x.shape, dtype=int)
        sign[prop_pos >= (1-alpha)] = 1
        sign[prop_neg >= (1-alpha)] = -1
        sign[sign != np.sign(x)] = 0
        x = np.abs(x) * sign
        out[key] = x
    out['s'] = pred_list[0]['s']
    out['img_shape'] = pred_list[0]['img_shape']
    return out


def normalize_coordinate(x):
    x = x - x.min()
    x = x / x.max() + 1e-12
    return x


def fit_nnisr(
        x, y, s=None, img_shape=None,
        hidden_widths=(256,)*4, activation='leakyrelu',
        alpha_threshold=0.05,
        n_permute=100, lr=1e-3, epochs=50, batch_size=4096,
        max_iter=2, n_states=11, alpha_states=0.5,
        prefix='nnisr/', device=None, n_jobs=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    threshold = compute_threshold(
            x=x, y=y, s=s, img_shape=img_shape,
            alpha=alpha_threshold,
            hidden_widths=hidden_widths,
            activation=activation,
            lr=lr, batch_size=batch_size,
            epochs=max(1, epochs//64),
            n_permute=n_permute,
            prefix=f'{prefix}threshold/',
            fast_approximation=True)

    # initialize
    n_observations, n_features = x.shape
    indiveff_pred = np.zeros_like(y)
    maineff_pred, __ = get_ols_est(x, y)
    noiselogvar_pred = np.ones_like(y[0])

    kwargs_list = []
    for i_state in range(n_states):
        kwargs = dict(
                maineff=maineff_pred,
                indiveff=indiveff_pred,
                noiselogvar=noiselogvar_pred,
                threshold=threshold,
                x=x, y=y, s=s, img_shape=img_shape,
                hidden_widths=hidden_widths,
                activation=activation,
                lr=lr, batch_size=batch_size, max_iter=max_iter,
                epochs=epochs,
                prefix_model=f'{prefix}states/state{i_state:02d}-',
                prefix_image=f'{prefix}states/state{i_state:02d}-',
                device=device)
        kwargs_list.append(kwargs)

    if n_jobs is None:
        n_jobs = n_states
    else:
        n_jobs = n_jobs
    if n_jobs > 1:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pred_list = pool.map(compute_multi_iter_kwargs, kwargs_list)
    else:
        pred_list = [
                compute_multi_iter_kwargs(kwargs)
                for kwargs in kwargs_list]

    save_pickle(pred_list, f'{prefix}pred-list.pickle')
    pred = combine_states(pred_list, alpha_states)

    return pred
