#root_dir = '/Users/nicholassofroniew/Documents/BBBC/BBBC020_v1'
#num_epochs = 20

import click
import segmentation.transforms as extended_transforms
from segmentation.datasets import BroadDataset
from segmentation.utilities import mIoULoss
from segmentation.model import UNet
from segmentation.main import train
from .common import success, status, error, warn
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os.path import join


@click.argument('output', nargs=1, metavar='<output directory>', required=False, default=None)
@click.argument('input', nargs=1, metavar='<input directory>', required=True)
@click.option('--epochs', nargs=1, default=2, type=float, help='Number of epochs')
@click.command('train', short_help='train on input directory', options_metavar='<options>')

def train_command(input, output, epochs):
    epochs = int(epochs)
    joint_transform = extended_transforms.Compose([
        extended_transforms.RandomHorizontallyFlip(),
        extended_transforms.RandomVerticallyFlip(),
        extended_transforms.RandomCrop(450)])

    input_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_transform = extended_transforms.MaskToTensor()

    status('setting up dataset from %s' % input)
    train_dataset = BroadDataset(input, 'train', joint_transform=joint_transform, input_transform=input_transform, target_transform=target_transform)

    trainloader = DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)

    status('loading model')
    net = UNet(1)
    net.train()

    criterion = mIoULoss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    status('starting training')
    for epoch in range(epochs):  # loop over the dataset multiple times
        train(trainloader, net, criterion, optimizer, epoch)
    status('finished training')

    status('saving network at %s' % output)
    snapshot_name = 'UNet'
    torch.save(net.state_dict(), join(output, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), join(output, 'opt_' + snapshot_name + '.pth'))
    status('finished training')
