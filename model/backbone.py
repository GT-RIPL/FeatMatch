import torch
from torch import nn
import torchvision.models as models
from torch.cuda import amp

from .wide_resnet import WideResNet
from .sslnet import SSLNet


def get_model(arch):
    if arch == 'cnn-13':
        model = SSLNet(10)
    elif arch.split('-')[0] == 'resnet':
        arch, depth = arch.split('-')
        model = getattr(models, arch+depth)()
    elif arch.split('-')[0] == 'wresnet':
        arch, depth, width = arch.split('-')
        depth, width = int(depth), int(width)
        model = WideResNet(num_classes=10, depth=depth, widen_factor=width)
    else:
        raise KeyError
    return model


def make_backbone(backbone):
    model = get_model(backbone)
    fext = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    fdim = model.fc.in_features

    return fext, fdim


class AmpModel(nn.Module):
    def __init__(self, model, amp=True):
        super(AmpModel, self).__init__()
        self.amp = amp
        self.model = model

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            return self.model(x)
