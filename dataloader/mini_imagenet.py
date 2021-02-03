import numpy as np
from pathlib import Path
import csv
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from torchvision import transforms as T
import argparse
from termcolor import cprint
import wget

import sys
sys.path.append('.')
from dataloader import SSLDataset
from util import data


def extract_mini_imagenet(src_dir, dst_dir, output_size=128):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    split_url = "https://github.com/twitter/meta-learning-lstm/raw/master/data/miniImagenet/"
    classes = set()
    for s in ['train', 'val', 'test']:
        url = split_url + f"{s}.csv"
        downloaded_file = wget.download(url, str(dst_dir))
        with open(downloaded_file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i, l in enumerate(csv_reader):
                if i == 0:
                    continue
                classes.add(l[1])

    Tpre = T.Compose([T.Resize(output_size, Image.LANCZOS), T.CenterCrop(output_size)])
    processing = lambda img_file: Tpre(Image.open(img_file).convert("RGB"))

    xtrain, ytrain = list(), list()
    pool = ThreadPool(128)
    cprint(f'Start extracting the TRAIN set', color='blue', attrs=['bold'])
    for ci in tqdm(classes):
        xci = [pool.apply_async(processing, args=(img, )) for img in sorted((src_dir/'train'/ci).iterdir())]
        xci = [np.array(xi.get()) for xi in xci]
        yci = [ci]*len(xci)
        xtrain.extend(xci)
        ytrain.extend(yci)
    xtrain = np.stack(xtrain)
    ytrain = np.array(ytrain)
    np.save(dst_dir/f'xtrain.npy', xtrain)
    np.save(dst_dir/f'ytrain.npy', ytrain)

    xtest, ytest = list(), list()
    cprint(f'Start extracting the TEST set', color='blue', attrs=['bold'])
    for ci in tqdm(classes):
        xci = [pool.apply_async(processing, args=(img, )) for img in sorted((src_dir/'val'/ci).iterdir())]
        xci = [np.array(xi.get()) for xi in xci]
        yci = [ci]*len(xci)
        xtest.extend(xci)
        ytest.extend(yci)
    xtest = np.stack(xtest)
    ytest = np.array(ytest)
    np.save(dst_dir/f'xtest.npy', xtest)
    np.save(dst_dir/f'ytest.npy', ytest)


def dataset_statistics(root_dir):
    root_dir = Path(root_dir)
    xtrain = np.load(root_dir/'xtrain.npy')
    xtest = np.load(root_dir/'xtest.npy')
    x = np.transpose(np.concatenate([xtrain, xtest]), (3, 0, 1, 2))/255.

    mean = np.mean(x.reshape((3, -1)), axis=1)
    mean_str = np.array2string(mean, separator=', ', formatter={'float_kind': lambda x: f'{x:.8f}'})
    print(f'mean = {mean_str}')

    std = np.std(x.reshape((3, -1)), axis=1)
    std_str = np.array2string(std, separator=', ', formatter={'float_kind': lambda x: f'{x:.8f}'})
    print(f'std = {std_str}')

    return mean, std


class MiniImageNetSSL(SSLDataset):
    def read_x(self, idx):
        return Image.fromarray(self.x[idx].copy())

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        root_dir = Path(root_dir)

        # test
        xt = np.load(root_dir / 'xtest.npy')
        yt = np.load(root_dir / 'ytest.npy')
        classes = np.unique(yt)
        name2num = dict(zip(classes, np.arange(len(classes))))
        yt = np.array([name2num[yi] for yi in yt])

        # val, lab, unlab
        x = np.load(root_dir / 'xtrain.npy')
        y = np.load(root_dir / 'ytrain.npy')
        x, y = data.shuffle_data([x, y], rand_seed)
        x, y = x[:50000], y[:50000]
        y = np.array([name2num[yi] for yi in y])
        if r_val is not None:
            (xv, yv), (x, y) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)
        else:
            xv, yv = xt, yt
        (xl, yl), (xu, yu) = data.split_data(x.copy(), y.copy(), rand_seed, r_lab)

        # reduce data
        if r_data is not None:
            xu, yu = data.split_data(xu.copy(), yu.copy(), rand_seed, r_data)[0]

        return xl, yl, xu, xv, yv, xt, yt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract mini-ImageNet')
    parser.add_argument('--src_dir', '-sd', help='source dir where the ILSVRC dataset locates in.')
    parser.add_argument('--dst_dir', '-dd', help='destination dir where the mini-ImageNet dataset should be saved to.')
    parser.add_argument('--size', '-sz', type=int, help='downsampled image size.')
    args = parser.parse_args()

    extract_mini_imagenet(args.src_dir, args.dst_dir, args.size)
