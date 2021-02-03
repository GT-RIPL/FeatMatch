from .base_loader import SSLDataLoader, SSLDataset, SupDataset, InfBatchSampler
from .cifar import Cifar10SSL, Cifar100SSL
from .svhn import SVHNSSL
from .mini_imagenet import MiniImageNetSSL
from .domainnet import DomainNetSSL, DomainNetSup

supported_ssl_dsets = {'cifar10': 'Cifar10SSL',
                       'cifar100': 'Cifar100SSL',
                       'svhn': 'SVHNSSL',
                       'mini-imagenet': 'MiniImageNetSSL',
                       'domainnet': 'DomainNetSSL'}

supported_sup_dsets = {'domainnet': 'DomainNetSup'}
