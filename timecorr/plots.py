import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import warnings
import numpy as np
import numpy.ma as ma

warnings.simplefilter("ignore")


def positive(mat_corrs, tol=1e-2):
    mask = (mat_corrs[:,:,0] > -tol)
    for i in range(1, mat_corrs.shape[2]):
        mask = mask & (mat_corrs[:,:,i] > -tol)
    return ~mask


def increasing(mat_corrs, tol=1e-1):
    deltas = np.diff(mat_corrs, n=1, axis=2)
    # np.concatenate((np.ones(mat_corrs.shape[:2] + (1,)), deltas), axis=2)
    return positive(deltas, tol)


def zero(mat_corrs, tol=1e-1):
    shape = mat_corrs.shape[:2]
    mask = np.isclose(mat_corrs[:, :, 0], np.zeros(shape), atol=tol)
    for i in range(1, mat_corrs.shape[2]):
        mask = mask & np.isclose(mat_corrs[:, :, i], np.zeros(shape), atol=tol)
    return ~mask


def stationary(mat_corrs, tol=1e-1):
    deltas = np.diff(mat_corrs, n=1, axis=2)
    # deltas = np.concatenate((np.zeros(mat_corrs.shape[:2] + (1,)), deltas), axis=2)
    return zero(deltas, tol)


def corrmap(mat_corrs, mask=None, triu_only=True):
    mean_corrs = np.mean(mat_corrs, axis=2) if mat_corrs.ndim > 2 else mat_corrs
    if triu_only:
        if mask is not None:
            mask = mask | np.tril(np.ones(mean_corrs.shape[:2], dtype=bool), k=-1)
        else:
            mask = np.tril(np.ones(mean_corrs.shape[:2], dtype=bool), k=-1)
    sns.heatmap(mean_corrs, mask=mask)
    plt.show()
    plt.clf()


def animate(corrs, vmin=None, vmax=None, axis=2, delay=200):
    def update(i):
        plt.clf()
        sns.heatmap(corrs[i], vmin=vmin, vmax=vmax)

    if isinstance(corrs, list):
        if len(corrs) == 0:
            raise ValueError(f'Invalid input `corrs`.  Must be a non-empty list of 2D np.ndarrays or a 3D np.ndarray.')
        elif not all((isinstance(obj, np.ndarray) and obj.ndim == 2) for obj in corrs):
            raise ValueError(f'Invalid input `corrs`.  Must be a non-empty list of 2D np.ndarrays or a 3D np.ndarray.')
    elif isinstance(corrs, np.ndarray):
        if corrs.ndim != 3:
            raise ValueError(f'Invalid input `corrs`.  Must be a non-empty list of 2D np.ndarrays or a 3D np.ndarray.')
        else:
            corrs_list = np.split(corrs, corrs.shape[axis], axis=axis)
            corrs = [np.squeeze(mat, axis=axis) for mat in corrs_list]

    fig = plt.figure()
    fr = len(corrs)

    vmin = corrs[0].min() if vmin is None else vmin
    vmax = corrs[0].max() if vmax is None else vmax
    sns.heatmap(corrs[0], vmin=vmin, vmax=vmax)

    return FuncAnimation(fig, update, frames=np.arange(0, fr), fargs=(vmin, vmax,), interval=delay, save_count=fr)
