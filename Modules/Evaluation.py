import torch
import numpy as np
from tqdm import tqdm
from Modules import Metrics

class Evaluation(object):
    def __init__(self, model, lossFunc=None, k=20):
        self.model = model
        self.lossFunc = lossFunc
        self.topk = k
        self.device = model.device

    def evalute(self, validGenerator):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        with torch.no_grad():
            batchSize = validGenerator.batchSize
            hidden = self.model.initHidden(batchSize)
            for ii , (input, target, finishedMask, validMask) in tqdm(enumerate(validGenerator),
                                                                     total=validGenerator.totalIters,
                                                                     miniters=1000, position=0, leave=True):
                input = input.to(self.device)
                target = target.to(self.device)
                hidden = self.model.resetHidden(hidden, finishedMask, validMask)
                logit, hidden = self.model(input, hidden)
               
                if(self.lossFunc is not None):
                    loss = self.lossFunc(logit, target)
                    if(~np.isnan(loss.item())):
                        losses.append(loss.item())

                recall, mrr = Metrics.calc(logit, target, k=self.topk)
                recalls.append(recall)
                mrrs.append(mrr.cpu().numpy())
                
        if(len(losses)):
            meanLoss = np.mean(losses)
        else :
            meanLoss = 0
                    
        meanRecall = np.mean(recalls)
        meanMrr = np.mean(mrrs)

        return meanLoss, meanRecall, meanMrr
    
