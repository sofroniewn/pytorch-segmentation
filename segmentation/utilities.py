'''
CrossEntropyLoss2d and initialize_weights taken from
https://github.com/ZijunDeng/pytorch-semantic-segmentation
'''

import torch.nn.functional as F
from torch import nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(BCELoss2d, self).__init__()
        self.nll_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(mIoULoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1
        predicted = F.sigmoid(inputs)
        intersection = predicted*targets.float()
        union = predicted + targets.float() - intersection
        return 100*(1-(intersection.sum()+smooth)/(union.sum()+smooth))

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
