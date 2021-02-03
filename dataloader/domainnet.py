from pathlib import Path
import wget
import zipfile
import argparse
from termcolor import cprint
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

import sys
sys.path.append('.')
from util import data
from dataloader import SSLDataset, SupDataset


def download(root_dir):
    root_dir = Path(root_dir)
    domains = {
        'clipart': ['http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
                    'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt',
                    'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt'],

        # 'infograph': ['http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        #               'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt',
        #               'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt'],

        'painting': ['http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
                     'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt',
                     'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt'],

        'quickdraw': ['http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
                      'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt',
                      'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt'],

        'real': ['http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
                 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt',
                 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt'],

        'sketch': ['http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
                   'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt',
                   'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt']
    }

    for domain, urls in domains.items():
        dst_dir = root_dir/domain
        if dst_dir.exists(): shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            cprint(f'Start downloading [{Path(url).name}]', color='blue', attrs=['bold'])
            wget.download(url, str(dst_dir))
            print()

            if Path(url).suffix == '.zip':
                file = dst_dir / Path(url).name
                with zipfile.ZipFile(file, 'r') as f:
                    f.extractall(root_dir)
                file.unlink()


def dataset_statistics(root_dir, domain):
    root_dir = Path(root_dir)

    xtrain, _ = read(root_dir/domain/f'{domain}_train.txt')
    xtest, _ = read(root_dir/domain/f'{domain}_test.txt')
    x = np.concatenate([xtrain, xtest])

    count = 0
    x_sum = np.zeros(3)
    x_sqsum = np.zeros(3)
    cprint('Start loading images...', color='blue', attrs=['bold'])
    for i, xi in enumerate(tqdm(x)):
        xi = np.asarray(read_img(root_dir/xi))
        xi = np.transpose(xi, (2, 0, 1)) / 255.
        xi = xi.reshape((3, -1))

        count += xi.shape[1]
        x_sum += np.sum(xi, axis=1)
        x_sqsum += np.sum(xi**2, axis=1)

    mean = x_sum/count
    mean_str = np.array2string(mean, separator=', ', formatter={'float_kind': lambda x: f'{x:.8f}'})
    print(f'mean = {mean_str}')

    std = np.sqrt((x_sqsum - count*mean**2)/(count-1))
    std_str = np.array2string(std, separator=', ', formatter={'float_kind': lambda x: f'{x:.8f}'})
    print(f'std = {std_str}')

    return mean, std


def read(file):
    x, y = [], []
    with open(file, 'r') as f:
        for _, line in enumerate(f):
            xi, yi = line.strip().split()
            x.append(xi)
            y.append(int(yi))
    return np.array(x), np.array(y)


def read_img(file, shape):
    file_ = list(file.resolve().parts)
    file_[-4] = file_[-4]+f'-{shape}'
    file_ = Path(*file_)

    if file_.exists():
        tmp = Image.open(file_)
        x = tmp.copy()
        tmp.close()
    else:
        file_.parent.mkdir(parents=True, exist_ok=True)
        x = Image.open(file).convert('RGB')
        resize = T.Compose([T.Resize(shape, Image.LANCZOS), T.CenterCrop(shape)])
        x = resize(x)
        x.save(file_)

    return x


def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data):
    root_dir = Path(root_dir)
    w_unlab = np.array(w_unlab) / np.sum(w_unlab)
    if len(set(tgt_domains) & set(src_domains)) != 0:
        print('tgt_domains should not overlap with src_domains')
        raise AttributeError

    # target test
    xt, yt = [], []
    for i, domain in enumerate(tgt_domains):
        xd, yd = read(root_dir / domain / f'{domain}_test.txt')
        xt.extend(xd.tolist())
        yt.extend(yd.tolist())
    for i, xi in enumerate(xt):
        xt[i] = root_dir / xi
    xt, yt = np.array(xt), np.array(yt)

    # target val, target lab, target unlab
    xv, yv, xl, yl, xu, yu, Nu = [], [], [], [], [], [], 0
    for i, domain in enumerate(tgt_domains):
        xd, yd = read(root_dir / domain / f'{domain}_train.txt')
        # target val
        if r_val is not None:
            (xvd, yvd), (xd, yd) = data.split_data(xd.copy(), yd.copy(), rand_seed, r_val)
            xv.extend(xvd.tolist())
            yv.extend(yvd.tolist())
        # target lab
        (xld, yld), (xud, yud) = data.split_data(xd.copy(), yd.copy(), rand_seed, r_lab)
        xl.extend(xld.tolist())
        yl.extend(yld.tolist())
        # target unlab
        (xdu, ydu), (xres, _) = data.split_data(xud.copy(), yud.copy(), rand_seed, 1.-r_unlab)
        xu.extend(xdu.tolist())
        yu.extend(ydu.tolist())
        Nu += len(xres)
    if r_val is not None:
        for i, xi in enumerate(xv):
            xv[i] = root_dir / xi
        xv, yv = np.array(xv), np.array(yv)
    else:
        xv, yv = xt, yt
    for i, xi in enumerate(xl):
        xl[i] = root_dir / xi
    xl, yl = np.array(xl), np.array(yl)

    # source unlab
    for i, domain in enumerate(src_domains):
        xd, yd = read(root_dir / domain / f'{domain}_train.txt')
        Ndu = int(round(Nu * w_unlab[i]))
        xd, yd = data.split_data(xd.copy(), yd.copy(), rand_seed, Ndu)[0]
        xu.extend(xd.tolist())
        yu.extend(yd.tolist())
    for i, xi in enumerate(xu):
        xu[i] = root_dir / xi
    xu, yu = np.array(xu), np.array(yu)

    # reduce data
    if r_data is not None:
        xl, yl = data.split_data(xl.copy(), yl.copy(), rand_seed, r_data)[0]
        xu, yu = data.split_data(xu.copy(), yu.copy(), rand_seed, r_data)[0]

    return xl, yl, xu, xv, yv, xt, yt


class DomainNetSSL(SSLDataset):
    def read_x(self, idx):
        return read_img(self.x[idx], self.shape)

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        return split_data(root_dir, sorted(tgt_domains), sorted(src_domains), r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data)


class DomainNetSup(SupDataset):
    def read_x(self, idx):
        return read_img(self.x[idx], self.shape)

    @staticmethod
    def split_data(root_dir, domain, r_val, r_data, rand_seed):
        root_dir = Path(root_dir)

        # test
        xt, yt = read(root_dir / domain / f'{domain}_test.txt')
        xt = xt.tolist()
        for i, xi in enumerate(xt):
            xt[i] = root_dir / xi
        xt = np.array(xt)

        xd, yd = read(root_dir / domain / f'{domain}_train.txt')
        # val
        if r_val is not None:
            (xv, yv), (xd, yd) = data.split_data(xd.copy(), yd.copy(), rand_seed, r_val)
            xv = xv.tolist()
            for i, xi in enumerate(xv):
                xv[i] = root_dir / xi
            xv = np.array(xv)
        else:
            xv, yv = xt, yt
        # train
        x, y = data.split_data(xd.copy(), yd.copy(), rand_seed, r_data)[0]
        x = x.tolist()
        for i, xi in enumerate(x):
            x[i] = root_dir / xi
        x = np.array(x)

        return x, y, xv, yv, xt, yt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and extract DomainNet')
    parser.add_argument('--root_dir', '-r', help='root dir where the DomainNet should be downloaded to')
    args = parser.parse_args()
    download(args.root_dir)
