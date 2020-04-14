import pandas as pd
import numpy as np
import torch

class Dataset(object):
    def __init__(self, path, sep=',', sessionKey='SessionId', itemKey='ItemId',
                 timeKey='Time', itemMap=None):
        '''  
        path : string 
            raw data path
        sep : char 
            csv file seperator (default: ',')
        sessionKey : string
            header of the session ID column in the input file (default: 'SessionId')
        itemKey : string
            header of the item ID column in the input file (default: 'ItemId')
        timeKey : string
            header of the timeStamp column in the input file (default: 'Time')
        itemMap : map 
            map ItemId to itemIndex values (default: None)
        '''
        self.sessionKey = sessionKey
        self.itemKey = itemKey
        self.timeKey = timeKey
        
        print("Loading data from {}".format(path))
        # Read data
        self.data = pd.read_csv(path, sep=sep, dtype={self.sessionKey: int, self.itemKey: int,
                                                       self.timeKey: float})
        
        self.itemMap = itemMap;
        if (self.itemMap is None):
            self.itemMap = self.createItemMap();
            
        self.data = self.addItemIndices(self.itemMap);
        self.offsetSessions = self.createOffsetSessions() 
        self.nItems = len(self.itemMap)

    def addItemIndices(self, itemMap):
        data = pd.merge(self.data, itemMap, on=self.itemKey, how='inner')
        sessionLen = data.groupby(self.sessionKey).size()
        data = data[np.in1d(data[self.sessionKey], sessionLen[sessionLen > 1].index)]
        if(len(data) == 0):
            raise ValueError("Data Cannot Be Empty") 
        return data

    def createItemMap(self):
        itemIds = self.data[self.itemKey].unique()
        itemMap = pd.Series(data=np.arange(len(itemIds)), index=itemIds);
        itemDict = {self.itemKey: itemIds, 'ItemIdx': itemMap[itemIds].values}
        itemMap = pd.DataFrame(itemDict)
        return itemMap

    def createOffsetSessions(self):
        self.data.sort_values([self.sessionKey, self.timeKey], inplace=True)
        offsetSessions = np.zeros(self.data[self.sessionKey].nunique() + 1, dtype=np.int32)
        offsetSessions[1:] = self.data.groupby(self.sessionKey).size().cumsum()
        return offsetSessions;

class DataGenerator():
    def __init__(self, dataset, batchSize=32, nSample=2048, sampleAlpha=0.75,
                 timeSort=True, trainRandomOrder=False, sampleStore=10000000):
        """
        creating session-parallel mini-batches.

        Parameters:
        -----------
        dataSet : Dataset
            It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (sessionKey, itemKey, timeKey properties).
        
        batchSize : int
            size of the minibatch, also effect the number of negative samples through minibatch based sampling (default: 32)
            
        nSample : int
            number of additional negative samples to be used (besides the other examples of the minibatch) (default: 2048)
        
        sampleAlpha : float
            the probability of an item used as an additional negative sample is supp^sample_alpha (default: 0.75)
            (e.g.: sampleAlpha=1 --> popularity based sampling; sampleAlpha=0 --> uniform sampling)

        timeSort : boolean
           whether to ensure the the order of sessions is chronological (default: True)
        
        trainRandomOrder : boolean
            whether to randomize the order of sessions in each epoch (default: False)
            
        sampleStore : int
            If additional negative samples are used (nSample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).

        """
        self.data = dataset.data
        self.nItems = dataset.nItems
        self.offsetSessions = dataset.offsetSessions
        
        self.nSample = nSample
        self.timeSort = timeSort       
        self.batchSize = batchSize
        self.sampleAlpha = sampleAlpha
        self.sampleStore = sampleStore
        self.trainRandomOrder = trainRandomOrder

        if(self.nSample > self.nItems):
            self.nSample = self.nItems - 1;
            
        if(self.batchSize > self.nItems):
            self.batchSize = self.nItems

        if (self.batchSize > len(self.offsetSessions) - 1):
            self.batchSize = len(self.offsetSessions) - 1
   
        if self.nSample:
            self.pop = self.createPop(dataset.itemMap, dataset.itemKey);
        
        # Order of data
        self.baseOrder = self.createBaseOrder(dataset.sessionKey, dataset.timeKey);
        # last data in session is only output and do not calculate as data in minibatch data generation
        self.totalIters = ((len(self.data) - len(self.offsetSessions)) // self.batchSize)

    def __iter__(self):
        '''Returns the iterator for producing session-parallel training mini-batches'''
        
        dataItems = self.data.ItemIdx.values;
        offsetSessions = self.offsetSessions
        sessionIdxArr = self.createSessionIdxArr();
        
        iters = np.arange(self.batchSize)
        maxiter = iters.max()
        start = offsetSessions[sessionIdxArr[iters]]
        end = offsetSessions[sessionIdxArr[iters] + 1]      
        nSessions = len(offsetSessions) - 1
        
        finished = False
        finishedMask = (end - start <= 1)
        validMask = (iters < nSessions)
        
        while not finished:
            minlen = (end - start).min()
            outIdx = dataItems[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                inIdx = outIdx
                outIdx = dataItems[start + i + 1]
                if (self.nSample):
                    if(self.sampleStore):
                        if(self.samplePointer == self.generatelength):
                            self.negativSamples = self.generateNegSamples(self.pop, self.generatelength)
                            self.samplePointer = 0
                        sample = self.negativSamples[self.samplePointer]
                        self.samplePointer += 1;
                    else:
                        sample = self.generateNegSamples(self.pop, 1);
                    y = np.hstack([outIdx, sample])
                else:
                    y = outIdx;
                    
                input = torch.LongTensor(inIdx)
                target = torch.LongTensor(y)
                yield input, target, finishedMask, validMask
                                
                finishedMask[:] = False;
                validMask[:] = True
                
            start = start + minlen - 1
            # indicator for the sessions to be terminated
            finishedMask = (end - start <= 1)
            nFinished = finishedMask.sum()
            iters[finishedMask] = maxiter + np.arange(1, nFinished + 1)
            maxiter += nFinished;
            
            # indicator for determining valid batch indices 
            validMask = (iters < nSessions)
            nValid = validMask.sum()
            
            if (nValid == 0):
                finished = True;
                break;
   
            iters[~validMask] = 0;         
            sessions = sessionIdxArr[iters[finishedMask]]
            start[finishedMask] = offsetSessions[sessions]
            end[finishedMask] = offsetSessions[sessions + 1]
            iters = iters[validMask]
            start = start[validMask]
            end = end[validMask]

    def generateNegSamples(self, pop, length):
        if self.sampleAlpha:
            sample = np.searchsorted(pop, np.random.rand(self.nSample * length))
        else:
            sample = np.random.choice(self.nItems, size=self.nSample * length)
        if length > 1:
            sample = sample.reshape((length, self.nSample))
        return sample
    
    def createPop(self, itemMap, itemKey):
        pop = self.data.groupby(itemKey).size()
        itemIds = itemMap.loc[:, itemKey].values
        pop = pop[itemIds].values ** self.sampleAlpha
        pop = pop.cumsum() / pop.sum()
        pop[-1] = 1
        if self.sampleStore:
            self.generatelength = (self.sampleStore // self.nSample)
            if self.generatelength <= 1:
                self.sampleStore = 0
                print('No example store was used')
            else:
                self.negativSamples = self.generateNegSamples(pop, self.generatelength)
                self.samplePointer = 0
        else:
            print('No example store was used')
        return pop;
               
    def createBaseOrder(self, sessionKey, timeKey):
        """
        Creating arrays to arrange data by time or not
        """
        if(self.timeSort):
            baseOrder = np.argsort(self.data.groupby(sessionKey)[timeKey].min().values) 
        else:
            baseOrder = np.arange(len(self.offsetSessions) - 1)
        return baseOrder
    
    def createSessionIdxArr(self):
        """
        Creating an array to fetch data randomly or sequentially
        """
        if (self.trainRandomOrder):
            sessionIdxArr = np.random.permutation(len(self.offsetSessions) - 1) 
        else:
            sessionIdxArr = self.baseOrder;
        
        return sessionIdxArr;
    
            
        
