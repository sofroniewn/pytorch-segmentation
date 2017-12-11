import click
import segmentation.transforms as extended_transforms
from segmentation.datasets import Dataset
from segmentation.utilities import mIoULoss
from segmentation.model import UNet
from segmentation.main import run
from .common import success, status, error, warn
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os.path import join, isdir, exists
from os import mkdir
from shutil import rmtree
from numpy import random
from pandas import DataFrame, read_csv

@click.argument('output', nargs=1, metavar='<output directory>', required=False, default=None)
@click.argument('model', nargs=1, metavar='<path to model>', required=True)
@click.argument('input', nargs=1, metavar='<input directory>', required=True)
@click.command('evaluate', short_help='evaluate model on input directory', options_metavar='<options>')

def evaluate_command(input, output, model):
    overwrite = True
    output = input if output is None else output

    input_transform = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ])
    target_transform = extended_transforms.MaskToTensor()

    status('setting up dataset from %s' % input)

    dataset = Dataset(input, input_transform=input_transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    status('loading model')
    if torch.cuda.is_available():
        net = UNet(1,1).cuda()
        net.load_state_dict(torch.load(model))
    else:
        net = UNet(1,1)
        net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    status('evaluating model')
    run(loader, net, output)
