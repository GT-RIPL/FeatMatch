import numpy as np
import torch
from torch import nn
from termcolor import cprint


def split_data(x, y, rand_seed, ratio):
    """
    split data into two portions.
    :param x: numpy arr. data, the 1st dimension is number of data.
    :param y: numpy arr. labels of data x. Could be None for unlabeled data
    :param rand_seed: random seed for data shuffling. if None then don't shuffle
    :param ratio: the ratio of the first portion.
    :return (x1, y1), (x2, y2): numpy arr. split data
    """
    if rand_seed is not None:
        rng = np.random.RandomState(rand_seed)
        shuffle_idx = rng.permutation(len(x))
        x, y = x[shuffle_idx], y[shuffle_idx]

    # detect and handle exceptions for ratio
    if isinstance(ratio, float):
        if ratio < 0.0:
            cprint(f'[Warning] float ratio = {ratio:.5f} < 0.0, set it to 0.0', color='yellow', attrs=['bold'])
            ratio = 0.0
        elif ratio > 1.0:
            cprint(f'[Warning] float ratio = {ratio:.5f} > 1.0, set it to 1.0', color='yellow', attrs=['bold'])
            ratio = 1.0
    else:
        if ratio < 0:
            cprint(f'[Warning] int ratio = {ratio} < 0, set it to 0', color='yellow', attrs=['bold'])
            ratio = 0
        elif ratio > len(x):
            cprint(f'[Warning] int ratio = {ratio} > Nx = {len(x)}, set it to {len(x)}', color='yellow', attrs=['bold'])
            ratio = len(x)

    if y is not None:
        r = ratio if isinstance(ratio, float) else ratio/len(x)
        x1, x2 = [], []
        y1, y2 = [], []
        for ci in np.unique(y):
            idx_ci = np.where(y == ci)[0]
            i = int(round(len(idx_ci)*r))
            x1.append(x[idx_ci[:i]])
            y1.append(y[idx_ci[:i]])
            x2.append(x[idx_ci[i:]])
            y2.append(y[idx_ci[i:]])
        x1, y1 = np.concatenate(x1), np.concatenate(y1)
        x2, y2 = np.concatenate(x2), np.concatenate(y2)

        if not isinstance(ratio, float):
            if len(x1) > ratio:
                xr1, yr1 = x1[ratio:], y1[ratio:]
                x1, y1 = x1[:ratio], y1[:ratio]
                x2 = np.concatenate([x2, xr1])
                y2 = np.concatenate([y2, yr1])
            elif len(x1) < ratio:
                ratio = len(x) - ratio
                xr2, yr2 = x2[ratio:], y2[ratio:]
                x2, y2 = x2[:ratio], y2[:ratio]
                x1 = np.concatenate([x1, xr2])
                y1 = np.concatenate([y1, yr2])

    else:
        i = round(len(x)*ratio) if isinstance(ratio, float) else ratio
        x1 = x[:i].tolist()
        x2 = x[i:].tolist()
        y1, y2 = None, None

    return (x1, y1), (x2, y2)


def shuffle_data(data, rand_seed=None):
    rng = np.random.RandomState(rand_seed)
    shuffle_idx = rng.permutation(len(data[0]))
    for i, d in enumerate(data):
        data[i] = d[shuffle_idx]
    return data


def compute_zca_components(x, save_dst=None):
    """
    comptute the parameters for ZCA transform
    :param x: NxHxWxC np image array. ranged 0~255.
    :param save_dst: save destination for computed parameters.
    :return m: (H*W*C) dimensional array. mean
    :return U: (H*W*C) x (H*W*C) np array. eigenvectors
    :return S: (H*W*C) dimensional array. singular values
    """
    N = len(x)
    xf = x.transpose((0, 3, 1, 2)).reshape((N, -1)).astype(np.float)/255.0

    # center data
    m = np.mean(xf, axis=0)
    xc = xf - m

    # computer SVD
    C = (xc.T @ xc)/N
    U, S, _ = np.linalg.svd(C)

    # save transform
    if save_dst is not None:
        np.savez_compressed(save_dst, mean=m, U=U, S=S)

    return m, U, S


def load_zca_transform(filepath):
    """
    construct the zca transformation matrix for torch tensor
    Ref:
    - https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
    - http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    Note that PyTorch do matrix multiplication on the left using a row vector.
    :param filepath: filepath to the pre-computed ZCA parameters
    :return T: ZCA transform
    """
    zca = np.load(filepath)
    m = torch.tensor(zca['mean'], dtype=torch.float)

    U = zca['U']
    S = zca['S']
    T = torch.tensor(U @ np.diag(1.0/(np.sqrt(S) + 1e-12)) @ U.transpose(), dtype=torch.float)

    return m, T


class ZCATransformer(nn.Module):
    """
    torch.nn.Module for accelerated ZCA transform on CUDA devices.
    """
    def __init__(self, zca_param_file):
        """
        :param zca_param_file: path to the file of zca parameters computed offline
        """
        super(ZCATransformer, self).__init__()
        zca_m, zca_T = load_zca_transform(zca_param_file)
        self.register_buffer('zca_m', zca_m)
        self.register_buffer('zca_T', zca_T)

    def forward(self, x):
        """
        compute zca transform on torch tensor
        :param x: NxCxHxW torch tensor image
        :return: tensor image after applying zca transform
        """
        shape = x.shape
        return torch.matmul(x.reshape(shape[0], -1) - self.zca_m, self.zca_T.t()).reshape(shape)


class MeanStdTransformer(nn.Module):
    """
    torch.nn.Module for accelerated Mean-Std image normalizer on CUDA devices.
    """
    def __init__(self, mean, std):
        """
        :param mean: list of float. mean of RGB channels respectively.
        :param std: list of float. std of RGB channels respectively.
        """
        super(MeanStdTransformer, self).__init__()
        mean = torch.Tensor(mean).reshape(1, len(mean), 1, 1)
        std = torch.Tensor(std).reshape(1, len(std), 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean)/self.std

