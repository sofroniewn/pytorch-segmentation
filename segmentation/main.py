root_dir = '/Users/nicholassofroniew/Documents/BBBC/BBBC020_v1'
num_epochs = 20

import transforms as extended_transforms
from datasets import BroadDataset
from utilities import mIoULoss
from model import UNet
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os.path import join

joint_transform = extended_transforms.Compose([
    extended_transforms.RandomHorizontallyFlip(),
    extended_transforms.RandomVerticallyFlip(),
    extended_transforms.RandomCrop(450)])

input_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

target_transform = extended_transforms.MaskToTensor()

train_dataset = BroadDataset(root_dir, 'train', joint_transform=joint_transform, input_transform=input_transform, target_transform=target_transform)

net = UNet(1)
net.train()

criterion = mIoULoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainloader = DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)



for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')

snapshot_name = 'UNet'
torch.save(net.state_dict(), join(root_dir, 'models', snapshot_name + '.pth'))
torch.save(optimizer.state_dict(), join(root_dir, 'models', 'opt_' + snapshot_name + '.pth'))
