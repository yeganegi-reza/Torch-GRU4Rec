import torch
import torch.nn as nn
from .Funcs import softmax_neg
import torch.nn.functional as F
from torch.autograd import Variable

class LossFunction(nn.Module):
    def __init__(self, lossType='top1', useCuda=True, bpreg=1.0):
        
        super(LossFunction, self).__init__()
        self.lossType = lossType
        self.useCuda = useCuda
        lossType = lossType.lower()

        if lossType == 'cross-entropy' or lossType == 'crossentropy':
            self._lossFn = SampledCrossEntropyLoss(self.useCuda)
        elif lossType == 'nll':
            self._lossFn = SampledNLLLoss(self.useCuda)
        elif lossType == 'cross-entropy':
            self._lossFn = nn.CrossEntropyLoss()
        elif lossType == 'top1':
            self._lossFn = TOP1Loss()
        elif lossType == 'bpr':
            self._lossFn = BPRLoss()
        elif (lossType == 'top1-max' or lossType == 'top1max'):
            self._lossFn = TOP1Max()
        elif (lossType == 'bpr-max' or lossType == 'bprmax'):
            self._lossFn = BPRMax(bpreg=bpreg)
        else:
            raise NotImplementedError

    def forward(self, input, target=None):
        return self._lossFn(input, target)

class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = negative Samples"""
    def __init__(self, useCuda):
        super(SampledCrossEntropyLoss, self).__init__()
        self.crossEntropy = nn.CrossEntropyLoss()
        self.useCuda = useCuda

    def forward(self, input, target=None):
        if(terget is None):
            batchSize = input.shape[0]
            target = Variable(torch.arange(batchSize).long())
            if self.useCuda:
                target = target.cuda()

        return self.crossEntropy(input, target)

class SampledNLLLoss(nn.Module):
    """ NLLLoss with n_classes = negative Samples"""
    def __init__(self, useCuda):
        super(SampledNLLLoss, self).__init__()
        self.nllLoss = nn.NLLLoss()
        self.useCuda = useCuda

    def forward(self, input, target=None):
        if (target is None):
            batchSize = input.shape[0]
            target = Variable(torch.arange(batchSize).long())
            if self.useCuda:
                target = target.cuda()

        return self.nllLoss(input, target)

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, input, target=None):
        diff = input.diag().view(-1, 1).expand_as(input) - input
        loss = -torch.mean(F.logsigmoid(diff))
        return loss
    
class BPRMax(nn.Module):
    def __init__(self, bpreg=1):
        super(BPRMax, self).__init__()
        self.bpreg = bpreg
        
    def forward(self, input, target=None):
        softmaxScores = softmax_neg(input)
        diff = input.diag().view(-1, 1) - input
        regTerm = self.bpreg * torch.sum((input ** 2) * softmaxScores, axis=1, keepdim=True)
        loss = torch.mean(-torch.log(torch.sum((torch.sigmoid(diff) * softmaxScores), axis=1, keepdim=True) + 1e-24) + regTerm)
        return loss

class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()
    def forward(self, input, target=None):
        diff = -(input.diag().view(-1, 1).expand_as(input) - input)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(input ** 2).mean()
        return loss

class TOP1Max(nn.Module):
    def __init__(self):
        super(TOP1Max, self).__init__()

    def forward(self, input, target=None):
        softmaxScores = softmax_neg(input)
        diff = -(input.diag().view(-1, 1) - input)
        loss = torch.sum((softmaxScores * (torch.sigmoid(diff) + torch.sigmoid(input ** 2))), axis=1,keepdim=True)
        loss = torch.mean(loss)
        return loss
    
    
