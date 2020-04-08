import os
import time
import torch
import numpy as np
from tqdm import tqdm
from .Evaluation import Evaluation

class Trainer(object):
    def __init__(self, model, trainGenerator, validGenerator, optim, lossFunc, topN, resultDir):
                
        self.topN = topN
        self.model = model
        self.optim = optim
        self.lossFunc = lossFunc
        self.resultDir = resultDir
        self.device = model.device
        self.evalutor = Evaluation(self.model, self.lossFunc, k=topN)
        
        self.trainGenerator = trainGenerator
        self.validGenerator = validGenerator
        
    def train(self, nEpochs=10):
        for epoch in range(nEpochs):
            st = time.time()
            print('Start Epoch #', epoch)
            
            trainLoss = self.trainEpoch(epoch)
            validLoss, recall, mrr = self.evalutor.evalute(self.validGenerator)
            
            print("Epoch: {}, train loss: {:.4f}, validloss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, trainLoss, validLoss, recall, mrr, time.time() - st))
            self.saveModel(epoch, validLoss, trainLoss, recall, mrr) 

    def trainEpoch(self, epoch):
        losses = []
        self.model.train()
        batchSize = float(self.trainGenerator.batchSize)
        hidden = self.model.initHidden(batchSize)
        negative = self.model.negative
        
        for _ , (input, target, finishedMask, validMask) in tqdm(enumerate(self.trainGenerator),
                                                                total=self.trainGenerator.totalIters,
                                                                miniters=1000, position=0, leave=True):
            input = input.to(self.device)
            target = target.to(self.device)            
            hidden = self.model.resetHidden(hidden, finishedMask, validMask)
            logit, hidden = self.model(input, hidden, target)

            # output sampling
            if(negative):
                loss = self.lossFunc(logit)
            else:
                loss = self.lossFunc(logit, target)
                
            loss = (float(len(input)) / batchSize) * loss
            
            if(~np.isnan(loss.item())):
                losses.append(loss.item())
                loss.backward()
                self.optim.step() 
                self.optim.zero_grad()
            
        meanLoss = np.mean(losses)
        return meanLoss
    
    def saveModel(self, epoch, validLoss, trainLoss, recall, mrr):
        checkPoints = {
              'model': self.model,
              'epoch': epoch,
              'optim': self.optim,
              'validLoss': validLoss,
              'trainLoss': trainLoss,
              'recall': recall,
              'mrr': mrr
        }
        modelName = os.path.join(self.resultDir, "model_{0:05d}.pt".format(epoch))
        torch.save(checkPoints, modelName)
        print("Save model as %s" % modelName)
