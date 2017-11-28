from os.path import join
from torch.autograd import Variable
from skimage.io import imsave
import torch

def train(trainloader, net, criterion, optimizer, epoch, display):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
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
        if i % display == display-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / display))
            running_loss = 0.0

def validate(valloader, net, criterion, optimizer, save, output):
    if save:
        torch.save(net.state_dict(), join(output, 'model.pth'))
        torch.save(optimizer.state_dict(), join(output, 'opt.pth'))
    correct = 0
    total = 0
    ind = 0
    for data in valloader:
        images, labels = data
        outputs = net(Variable(images))
        loss = criterion(outputs, Variable(labels)).data.numpy()[0]
        print(loss)
        prediction = F.sigmoid(outputs)
        predict = prediction.squeeze(0).squeeze(0).data.numpy()
        if save:
            imsave(join(output, 'predict_%04d.tif' % ind), (255*predict).astype('uint8'), plugin='tifffile', photometric='minisblack')


    total += labels.size(0)
    correct += loss
    ind += 1

    print('Mean loss: %.2f %%' % (
        correct / total))
