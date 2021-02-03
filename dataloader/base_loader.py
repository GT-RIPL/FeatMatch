from pathlib import Path
import numpy as np
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler


class SSLDataLoader(object):
    def __init__(self, labeled_dset, unlabeled_dset, bsl, bsu, num_workers):
        bs = bsl + bsu
        sampler_lab = InfBatchSampler(len(labeled_dset), bsl)
        sampler_unlab = InfBatchSampler(len(unlabeled_dset), bsu)
        self.labeled_dset = DataLoader(labeled_dset, batch_sampler=sampler_lab, num_workers=int(num_workers*bsl/bs))
        self.unlabeled_dset = DataLoader(unlabeled_dset, batch_sampler=sampler_unlab, num_workers=int(num_workers*bsu/bs))

        self.labeled_iter = iter(self.labeled_dset)
        self.unlabeled_iter = iter(self.unlabeled_dset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            xl, yl = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_dset)
            xl, yl = next(self.labeled_iter)

        try:
            xu = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_dset)
            xu = next(self.unlabeled_iter)

        return xl, yl, xu


class InfBatchSampler(Sampler):
    def __init__(self, N, batch_size):
        self.N = N
        self.batch_size = batch_size if batch_size < N else N
        self.L = N // batch_size

    def __iter__(self):
        while True:
            idx = np.random.permutation(self.N)
            for i in range(self.L):
                yield idx[i*self.batch_size:(i+1)*self.batch_size]

    def __len__(self):
        return sys.maxsize


class SSLDataset(Dataset):
    def __init__(self, x, y, Taggr, Tsimp, K, shape):
        super().__init__()
        self.x = x
        self.y = y
        self.Taggr = Taggr
        self.Tsimp = Tsimp
        self.K = K
        self.shape = shape

    def read_x(self, idx):
        raise NotImplementedError

    def get_x(self):
        x = []
        for idx in range(len(self.x)):
            xi = self.read_x(idx)
            x.append(xi)
        return x

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        """
        static method to split data into train/val/test and lab/unlab sets
        :param root_dir: pathlib.Path object. root dir of the dataset.
        :param tgt_domains: list of str. list of target domains.
        :param src_domains: list of str. list of source domains.
        :param r_val: number. ratio of validation set from the train set.
        :param r_lab: number. ratio of labeled data from the target train set.
        :param r_unlab: number. ratio of unlabeled data between the source train set and the target train set.
        :param w_unlab: list of numbers. sampling weights for unlabeled source sets.
        :param rand_seed: number. random seed.
        :param r_data: number. ratio of data to consider.
        :return xl, yl, xu, xv, yv, xt, yt: different data splits.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xi = self.read_x(idx)
        if self.K is not None:
            x = [self.Tsimp(xi)]
            for _ in range(self.K):
                x.append(self.Taggr(xi))
            x = torch.stack(x)
        else:
            x = self.Tsimp(xi)

        if self.y is not None:
            return x, self.y[idx]
        else:
            return x


class SupDataset(Dataset):
    def __init__(self, x, y, T, shape):
        super().__init__()
        self.x = x
        self.y = y
        self.T = T
        self.shape = shape

    def read_x(self, idx):
        raise NotImplementedError

    @staticmethod
    def split_data(root_dir, domain, r_val, r_data, rand_seed):
        """
        static method to split data into train/val/test and lab/unlab sets
        :param root_dir: pathlib.Path object. root dir of the dataset.
        :param domain: str. target domain.
        :param r_val: number. ratio of validation set from the train set.
        :param r_data: number. ratio of data to consider.
        :param rand_seed: number. random seed.
        :return x, y, xv, yv, xt, yt: different data splits.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.T(self.read_x(idx)), self.y[idx]
