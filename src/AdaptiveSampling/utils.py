import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def save_arguments(dataset, met, s, iteration, **kwargs):
    filename = f"AdaptiveSampling/results/{dataset}/{met}/Seed{s}_Iteration{iteration}_args.pkl"
    with open(os.path.join(get_project_root(), filename), 'wb') as f:
        pickle.dump(kwargs, f)


def synth_func(x_in):
    randn = np.random.normal(size=len(x_in))
    gauss = (2 + 2 * np.cos(1.2 * x_in))
    noise = gauss * randn
    orig2 = 10 + 5 * np.cos(x_in + 2)
    return orig2 + noise


def create_sin_data_full(n=1000, plot=False):
    """Create a synthetic sinusoidal dataset with varying PI width"""
    np.random.seed(7)
    X = np.linspace(-5, 5, num=n)
    randn = np.random.normal(size=n)
    gauss = (2 + 2 * np.cos(1.2 * X))
    noise = gauss * randn
    orig = 10 + 5 * np.cos(X + 2)
    Y = orig + noise
    P1 = orig + 1.96 * gauss
    P2 = orig - 1.96 * gauss
    if plot:
        plt.figure(figsize=(4, 3))  # (9.97, 7.66)
        plt.fill_between(X, P1, P2, color='gray', alpha=0.5, linewidth=0, label='Ideal 95% PIs')
        plt.scatter(X, Y, label='Data with noise')
        plt.plot(X, orig, 'r', label='True signal')
        plt.legend()
    return X, Y, P1, P2


def create_sin_data(n=1000, plot=False, extra=False):
    """Create a synthetic sinusoidal dataset with varying PI width"""
    np.random.seed(7)
    range_values = np.linspace(-5, 5, 100)
    X = np.array([np.random.choice(range_values) for _ in range(n)])
    Xorig = X.copy()

    # Select a few random samples from certain range
    mask = ((X >= -4) & (X < 1)) | (X > 3)
    mask2 = (X >= 1) & (X <= 3)
    mask3 = (X < -4) & (X >= -5)
    selected_indices = np.random.choice(np.where(mask2)[0], size=1, replace=False)
    selected_indices2 = np.random.choice(np.where(mask3)[0], size=3, replace=False)
    remaining_indices = np.setdiff1d(np.where(mask2)[0], selected_indices + selected_indices2)
    Xextra = X[remaining_indices]
    X = np.concatenate([X[mask], X[selected_indices], X[selected_indices2]])

    randn = np.random.normal(size=len(Xorig))
    gauss = (2 + 2 * np.cos(1.2 * Xorig))
    noise = gauss * randn
    orig = 10 + 5 * np.cos(Xorig + 2)
    Yorig = orig + noise
    Y = np.concatenate([Yorig[mask], Yorig[selected_indices], Yorig[selected_indices2]])

    randn = np.random.normal(size=len(Xextra))
    gauss = (2 + 2 * np.cos(1.2 * Xextra))
    noise = gauss * randn
    orig2 = 10 + 5 * np.cos(Xextra + 2)
    Yextra = orig2 + noise

    Xorig = np.linspace(-5, 5, num=1000)
    gauss = (2 + 2 * np.cos(1.2 * Xorig))
    orig = 10 + 5 * np.cos(Xorig + 2)
    P1 = orig + 1.96 * gauss
    P2 = orig - 1.96 * gauss

    if plot:
        plt.figure(figsize=(9.97, 7.66))
        plt.fill_between(Xorig, P1, P2, color='gray', alpha=0.5, linewidth=0, label='Ideal 95% PIs')
        plt.scatter(X, Y, label='Data with noise')
        # plt.scatter(Xextra, Yextra, label='Data with noise')
        plt.plot(Xorig, orig, 'r', label='True signal')
        plt.legend()
        plt.xlim(-5, 5)

    if not extra:
        return X, Y, Xorig, P1, P2
    else:
        return X, Y, Xorig, P1, P2, Xextra, Yextra


def plot_state(xt, yu, yl, Xtr, Ytr, Xv, Yv, ep_unc, Xorig=None, P1=None, P2=None, sampled=None, total_sampled=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    if Xorig is not None and P1 is not None and P2 is not None:
        axs[0].fill_between(Xorig, P1, P2, color='gray', alpha=0.2, linewidth=0, label='Ideal 95% PIs')
    axs[0].scatter(Xtr, Ytr, s=10)
    if len(xt) == len(yu):
        axs[0].scatter(xt, yu, s=10)
        axs[0].scatter(xt, yl, s=10)
    if total_sampled is not None:
        axs[0].scatter(total_sampled[0], total_sampled[1], s=16, c='k')
    if sampled is not None:
        axs[0].scatter(sampled[0], sampled[1], s=10, c='r')
    axs[0].scatter(Xv, Yv, s=10, c='pink')
    axs[0].set_xlabel('$\mathbf{x}$', fontsize=15)
    axs[0].set_ylabel('$y$', fontsize=15)
    axs[1].scatter(xt, ep_unc)
    axs[1].set_xlabel('$\mathbf{x}$', fontsize=15)
    axs[1].set_ylabel('$Q_t(\mathbf{x})$', fontsize=15)
    plt.ylim(-0.5, 11.5)
    plt.tight_layout()
    # plt.show()
