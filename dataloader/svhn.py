import scipy.io as sio
import numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.append('.')
from util import data
from dataloader import SSLDataset


class SVHNSSL(SSLDataset):
    def read_x(self, idx):
        return Image.fromarray(self.x[idx].copy())

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        root_dir = Path(root_dir)

        # test
        d = sio.loadmat(root_dir / 'test_32x32.mat')
        xt = np.transpose(d['X'], (3, 0, 1, 2))
        yt = d['y'].reshape(-1).astype(int) - 1  # SVHN labels are 1-10

        # val, lab, unlab
        d = sio.loadmat(root_dir / 'train_32x32.mat')
        x = np.transpose(d['X'], (3, 0, 1, 2))
        y = d['y'].reshape(-1).astype(int) - 1  # SVHN labels are 1-10
        if r_val is not None:
            (xv, yv), (x, y) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)
        else:
            xv, yv = xt, yt
        (xl, yl), (xu, yu) = data.split_data(x.copy(), y.copy(), rand_seed, r_lab)

        # reduce data
        if r_data is not None:
            xu, yu = data.split_data(xu.copy(), yu.copy(), rand_seed, r_data)[0]

        return xl, yl, xu, xv, yv, xt, yt
