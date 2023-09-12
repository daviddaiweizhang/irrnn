import numpy as np


def random_cube(v0, v1, v2, lohi=[1.0, 2.0], mima=[0.2, 0.6]):
    size = np.array([v0, v1, v2])
    r = mima[0] + np.random.rand() * (mima[1] - mima[0])
    start = np.random.rand(2) * (1 - r)
    r_len = np.round(r * size[:2]).astype(int)
    start_idx = np.floor(start * size[:2]).astype(int)
    end_idx = start_idx + r_len
    a = np.random.rand() * (lohi[1] - lohi[0]) + lohi[0]
    a *= np.sign(np.random.randn())
    idx = np.stack(np.meshgrid(
        np.linspace(0, 1, end_idx[0] - start_idx[0]),
        np.linspace(0, 1, end_idx[1] - start_idx[1]),
        indexing='ij'), -1)
    t = ((idx - 0.5)**2).sum(-1)**0.5
    t = np.clip(t, None, np.quantile(t, 0.6))
    t -= t.min()
    t /= t.max() + 1e-12
    t = t * a * 0.5 + a * 0.5
    x = np.zeros(size)
    x[
            start_idx[0]:end_idx[0],
            start_idx[1]:end_idx[1]] = t[..., np.newaxis]
    return x


def random_ellipsoid(v0, v1, v2, lohi=[2.0, 4.0], mima=[0.3, 0.7], seed=None):
    if seed is not None:
        np.random.seed(seed)
    size = np.array([v0, v1, v2])
    x = np.zeros(size)
    r = mima[0] + np.random.rand() * (mima[1] - mima[0])
    start = np.random.rand(2) * (1 - r)
    end = start + r
    c = (start + end) / 2
    r = end - c
    start_idx = np.floor(start * size[:2]).astype(int)
    end_idx = np.ceil(end * size[:2]).astype(int)
    a = np.random.rand() * (lohi[1] - lohi[0]) + lohi[0]
    a *= np.sign(np.random.randn())

    idx = np.stack(np.meshgrid(
        range(start_idx[0], end_idx[0]),
        range(start_idx[1], end_idx[1]),
        indexing='ij'), -1)
    s = np.sqrt((((idx / size[:2] - c) / r)**2).sum(-1))
    s = np.clip(s, 0, s[0, (end_idx[1] - start_idx[1]) // 2])
    s -= s.min()
    s /= s.max() + 1e-12
    s = (1 - s) * a
    x[
            start_idx[0]:end_idx[0],
            start_idx[1]:end_idx[1],
            ] = s[..., np.newaxis]
    return x


def random_cell(size):

    assert size[0] % 2 == 0
    assert size[1] % 2 == 0
    size = [size[0] // 2, size[1] // 2, size[2]]
    x_list = np.zeros([2, 2] + size)
    sign_list = np.array([[-1, 1], [1, -1]], dtype=float)
    sign_list *= np.sign(np.random.randn())
    k = 1
    for __ in range(k):
        for __ in range(2):
            t = random_cube(*size, lohi=[1, 2], mima=[0.4, 0.4])
            t = np.abs(t) / np.abs(t).max()
            x_list[0, 0] = np.maximum(x_list[0, 0], t)
        for __ in range(8):
            t = random_cube(*size, lohi=[1, 2], mima=[0.1, 0.1])
            t = np.abs(t) / np.abs(t).max()
            x_list[0, 1] = np.maximum(x_list[0, 1], t)
        for __ in range(2):
            t = random_ellipsoid(*size, lohi=[7, 8], mima=[0.5, 0.5])
            t = np.abs(t) / np.abs(t).max()
            x_list[1, 0] = np.maximum(x_list[1, 0], t)
        for __ in range(8):
            t = random_ellipsoid(*size, lohi=[7, 8], mima=[0.2, 0.2])
            t = np.abs(t) / np.abs(t).max()
            x_list[1, 1] = np.maximum(x_list[1, 1], t)
    scale_list = np.abs(x_list).std((-1, -2, -3), keepdims=True)
    x_list /= scale_list + 1e-12
    x_list *= np.expand_dims(sign_list, (-1, -2, -3))
    x = np.concatenate([
        np.concatenate(x_list[0], axis=1),
        np.concatenate(x_list[1], axis=1)
        ], axis=0)
    return x


def random_paraboloid(size, cut=False, lohi=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    c = np.random.rand(3)
    a = 1.0
    b = 1.0
    if lohi is None:
        u = np.random.randn() * a
    else:
        u = np.random.rand() * (lohi[1] - lohi[0]) + lohi[0]
    idx = np.stack(np.meshgrid(
        range(size[0]), range(size[1]), range(size[2]),
        indexing='ij'), -1)
    t = (idx / size - c)
    x = np.sqrt((t**2)[..., :-1].sum(-1)) * 2 * b
    x = x - x.mean() + u
    if cut:
        lower = np.random.rand(3) * 0.8
        lower = (lower * size).astype(int)
        upper = lower + (0.2 * np.array(size)).astype(int)
        # target = x.min((0, 1)) * 0.5 + x.max((0, 1)) * 0.5
        # x[lower[0]:upper[0]] = target
        # x[:, lower[1]:upper[1]] = target
        x[:lower[0], :lower[1]] = 0.0
        x[:lower[0], upper[1]:] = 0.0
        x[upper[0]:, :lower[1]] = 0.0
        x[upper[0]:, upper[1]:] = 0.0
    isin = ~np.isclose(0, x)
    x[isin] -= x[isin].mean()
    return x


def adj_variance(
        A, beta, omega, sigmasq, beta_stn, omega_stn,
        omega_itv, scale=1.0):
    beta = beta / (A @ beta).std() * scale
    # assuming E[omeag] == 0
    # omega = omega - omega.mean(1, keepdims=True)
    omega = omega / omega.std(1, keepdims=True)
    # omnoi = np.random.randn(omega.shape[0], 1)
    # omnoi = omnoi / omnoi.std()
    # omega = omega + omnoi * np.sqrt(omega_itv)
    omega = (
            omega / omega.std() / np.sqrt(beta_stn)
            * np.sqrt(omega_stn) * scale)
    sigmasq = sigmasq / sigmasq.mean() / beta_stn * scale**2
    return beta, omega, sigmasq


def gen_wave(shape, frq=2, x0=0, y0=0, cut=False):
    ndim = len(shape)
    if np.size(frq) == 1:
        frq = [frq] * ndim
    if np.size(x0) == 1:
        x0 = [x0] * ndim
    if np.size(y0) == 1:
        y0 = [y0] * ndim
    assert len(frq) == len(x0) == len(y0) == ndim
    out = np.zeros(shape)
    for i in range(ndim-1):
        c = np.random.rand()
        x = np.linspace(0, 1, shape[i])
        y = (np.sin(2*np.pi * frq[i] * (x - x0[i] - c)) + y0[i])
        y = np.expand_dims(y, list(range(i)) + list(range(i+1, ndim)))
        out = out + y
    if cut:
        is_pos = out > out.max() * 0.3
        is_neg = out < out.min() * 0.7
        out[(~is_pos) * (~is_neg)] = 0.0
    out -= out.min()
    out /= out.max() + 1e-12
    out *= 2.0
    return out


def gen_data(
        V_out, N, Q, beta_stn=1.0, omega_stn=1.0,
        omega_itv=1.0, noise_dist='gauss', noise_var='cons',
        scale=1.0, Va=128, cut=False, dtype=np.float32):
    assert len(V_out) == 3
    assert np.max(V_out) <= Va
    A = np.random.randn(N, Q)
    V = [Va] * 3
    VV = np.prod(V)
    beta = np.zeros([Q, VV])
    for i in range(Q):
        beta[i] = random_cell(V).flatten()
    omega = np.array([
        random_paraboloid(V, cut=cut).flatten() for _ in range(N)])

    # make sure E[noise] == 0, Var[noise] == 1
    if noise_dist == 'gauss':
        noise = np.random.normal(loc=0.0, scale=1.0, size=(N, VV))
    elif noise_dist == 'chisq':
        df = 3
        noise = np.random.chisquare(df=df, size=(N, VV))
        noise = (noise - df) / np.sqrt(df * 2)
    else:
        raise ValueError('noise distribution not recognized')
    noise = noise - noise.mean(0)
    noise = noise / noise.std(0)

    if noise_var == 'cons':
        sigmasq = np.ones(VV)
    elif noise_var == 'wave':
        sigmasq = np.exp(gen_wave(V, cut=cut).flatten())
    else:
        raise ValueError('noise variance pattern not recognized')

    beta, omega, sigmasq = adj_variance(
            A, beta, omega, sigmasq, beta_stn, omega_stn,
            omega_itv, scale)

    X = A @ beta + omega + np.sqrt(sigmasq) * noise

    beta_stn_emp = (A @ beta).var() / (np.sqrt(sigmasq) * noise).var()
    omega_stn_emp = omega.var() / (np.sqrt(sigmasq) * noise).var()
    # omega_itv_emp = omega.mean(1).var() / omega.var(1).mean()
    scale_emp = (A @ beta).std()
    assert np.isclose(beta_stn_emp, beta_stn)
    assert np.isclose(omega_stn_emp, omega_stn)
    # assert np.isclose(omega_itv_emp, omega_itv)
    assert np.isclose(scale_emp, scale)

    stride = Va // np.array(V_out)
    stop = stride * V_out
    start = np.zeros_like(stop)

    X_out = thin_3d(
            X.reshape(-1, *V),
            start, stop, stride)
    beta_out = thin_3d(
            beta.reshape(-1, *V),
            start, stop, stride)
    omega_out = thin_3d(
            omega.reshape(-1, *V),
            start, stop, stride)
    sigmasq_out = thin_3d(
            sigmasq.reshape(-1, *V),
            start, stop, stride)

    return (
            X_out.astype(dtype), A.astype(dtype),
            beta_out.astype(dtype), omega_out.astype(dtype),
            sigmasq_out.astype(dtype))


def thin_3d(x, start, stop, stride):
    return x[
            ...,
            start[-3]:stop[-3]:stride[-3],
            start[-2]:stop[-2]:stride[-2],
            start[-1]:stop[-1]:stride[-1]]
