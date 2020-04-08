import torch
import numpy as np
from torch import nn

class GRU4Rec(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize=100, nLayers=1, batchSize=32, negative=True,
                 embeddingDim=-1, dropoutHidden=0.0, dropoutEmbed=0.0, finalAct='elu-1',
                 sigma=0.0, initAsNormal=False, cuda=True):
        '''
    GRU4Rec(inputSize,outputSize,hiddenSize=100,nLayers=1,batchSize=32,negative=True,
                 embeddingDim=-1,dropoutHidden=0.0,dropoutEmbed=0.0,finalAct='tanh',sigma=0.0,
                 initAsNormal=False,cuda=True)
    
    Initializes the network.

    Parameters
    -----------
    inputSize : int 
        number of items in data
    outputSize : int 
        number of items in data 
    hiddenSize : int
        number of GRU units in the layers (default: 100)
    nLayers : int
        number of GRU layers (default: 1)
    batchSize : int
        size of the minibatch, also effect the number of negative samples through minibatch based sampling (default: 32)
    negative : boolean
        if True, only negative and positive sample outputs returns (default: True)
    embeddingDim : int
        size of the embedding used, -1 means not to use embedding (default: -1)
    dropoutHidden : float
        probability of dropout of hidden units (default: 0.0)
    dropoutEmbed : float 
        probability of dropout of the input units, applicable only if embeddings are used (default: 0.0)
    finalAct : 'tanh', 'relu', 'softmax', 'softmax_logit', 'elu-<X>', 'leaky-<X>'
        selects the activation function of the final layer, <X> is the parameter of the activation function (default : 'elu-1')
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0.0)
    initAsNormal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    cuda : boolean
        param to use gpu or not (default: ture)
    '''
        super(GRU4Rec, self).__init__()

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.sigma = sigma
        self.finalAct = finalAct;
        self.negative = negative

        self.embeddingDim = embeddingDim
        
        self.dropoutHidden = dropoutHidden
        self.dropoutEmbed = dropoutEmbed
        self.initAsNormal = initAsNormal
        
        self.useCuda = cuda
        self.device = torch.device('cuda' if self.useCuda else 'cpu')
      
        self.designNetArch();
        self.setFinalAct(self.finalAct)
        self.initParams();
        self = self.to(self.device)

    def setFinalAct(self, finalAct):
        finalAct = finalAct.lower();
        if finalAct == 'tanh':
            self.finalAct = nn.Tanh()
        elif finalAct == 'relu':
            self.finalAct = nn.ReLU()
        elif finalAct == 'softmax':
            self.finalAct = nn.Softmax(dim=1)
        elif finalAct == 'softmaxlogit' or finalAct == 'softmax_logit' :
            self.finalAct = nn.LogSoftmax(dim=1)
        elif finalAct.startswith('elu-'):
            self.finalAct = nn.ELU(alpha=float(finalAct.split('-')[1]))
        elif finalAct.startswith('leaky-'):
            self.finalAct = nn.LeakyReLU(negative_slope=float(finalAct.split('-')[1]))

    def forward(self, input, hidden, target=None):
        
        if(self.embeddingDim > 0):
            embedded = self.embedding(input)
            if(self.training and self.dropoutEmbed > 0.0):
                embedded = self.dropoutEmb(embedded)
        else:
            embedded = self.onehotEncode(input)
        embedded = embedded.unsqueeze(0)
        
        output, hNew = self.gru(embedded, hidden) 
        output = output.view(-1, output.size(-1))  
         
        if(self.dropoutHidden > 0):
            output = self.dropoutH(output)
            hNew[-1] = output            
        output = self.linear(output)
        
        if(target is not None and self.negative):
            output = output[:, target.view(-1)]
        
        output = self.finalAct(output)
        return output, hNew

    def initOnehot(self):
        onehotBuffer = torch.FloatTensor(self.batchSize, self.outputSize)
        onehotBuffer = onehotBuffer.to(self.device)
        return onehotBuffer

    def onehotEncode(self, input):
        self.onehotBuffer.zero_()
        index = input.view(-1, 1)
        onehot = self.onehotBuffer[:len(index)].scatter_(1, index, 1)
        return onehot

    def initHidden(self, batchSize):
        h0 = torch.zeros(self.nLayers, int(batchSize), self.hiddenSize).to(self.device)
        return h0
    
    def resetHidden(self, hidden, finishedMask, validMask):
        """resets hidden state when sessions terminated"""
        if any(finishedMask):
            hidden[:, finishedMask, :] = 0 

        if any((~validMask)):
            hidden = hidden[:, validMask, :]
            
        return hidden.data;
    
    def initParams(self):
        if (self.embeddingDim > 0):
            self.initMatrix(self.embedding.weight)

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                WR, WZ, WN = param.chunk(3, 0)
                self.initMatrix(WR)
                self.initMatrix(WZ)
                self.initMatrix(WN)
    
            elif 'bias' in name :
                param.data.zero_()
        
        self.initMatrix(self.linear.weight)
        self.linear.bias.data.zero_()
        
    def initMatrix(self, param):
        shape = list(param.shape)
        if self.sigma != 0.0:
            sigma = self.sigma;
        else :
            sigma = np.sqrt(6.0 / np.sum(shape))
             
        if self.initAsNormal:
            return param.data.uniform_(0, sigma)
        else:
            return param.data.uniform_(-sigma, sigma)
    
    def designNetArch(self):
        if(self.embeddingDim > 0 ):
            self.embeddingDim = self.embeddingDim
            self.embedding = nn.Embedding(self.inputSize, self.embeddingDim)
            if(self.dropoutEmbed > 0.0):
                self.dropoutEmb = nn.Dropout(self.dropoutEmbed)
            self.gru = nn.GRU(self.embeddingDim, self.hiddenSize, self.nLayers, bias=False,
                              dropout=self.dropoutHidden)
        else:
            self.gru = nn.GRU(self.inputSize, self.hiddenSize, self.nLayers, bias=False,
                              dropout=self.dropoutHidden)
            self.onehotBuffer = self.initOnehot()
            
        if(self.dropoutHidden > 0):
            self.dropoutH = nn.Dropout(self.dropoutHidden)
        
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)
