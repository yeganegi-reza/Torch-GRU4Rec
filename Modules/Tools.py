from Modules import Loss, Dataset, Model, Optimizer, Trainer
import datetime
import os

def saveParams(resultDir, inputSize, outputSize, lossType, finalAct, nLayers, hiddenSize, nEpochs,
               batchSize, negative, dropoutHidden, dropoutEmbed, lr, momentum, weightDecay, embeddingDim,
               trainNSample, validNSample, sampleAlpha, optimizerType, bpreg, sigma, initAsNormal,
               trainRandomOrder, timeSort, topN, sessionKey,
               itemKey, timeKey, cuda):
    now = datetime.datetime.now()
    S = '{:02d}-{};{:02d}.{:02d}'.format(now.day, now.strftime("%B"), now.hour, now.minute)
    
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)        
        
    resultDir = os.path.join(resultDir, S)
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
        
    file = open(os.path.join(resultDir, 'networkParams'), "w");
    file.write("inputSize=" + str(inputSize) + "\n");
    file.write("outputSize=" + str(outputSize) + "\n");
    file.write("lossType=" + str(lossType) + "\n");
    file.write("finalAct=" + str(finalAct) + "\n");
    file.write("nLayers=" + str(nLayers) + "\n");
    file.write("hiddenSize=" + str(hiddenSize) + "\n");
    file.write("nEpochs=" + str(nEpochs) + "\n");
    file.write("batchSize=" + str(batchSize) + "\n");
    file.write("negative=" + str(negative) + "\n");
    file.write("dropoutHidden=" + str(dropoutHidden) + "\n");
    file.write("dropoutEmbed=" + str(dropoutEmbed) + "\n");
    file.write("lr=" + str(lr) + "\n");
    file.write("momentum=" + str(momentum) + "\n");
    file.write("weightDecay=" + str(weightDecay) + "\n");
    file.write("embeddingDim=" + str(embeddingDim) + "\n");
    file.write("trainNSample=" + str(trainNSample) + "\n");
    file.write("validNSample=" + str(validNSample) + "\n");
    file.write("sampleAlpha=" + str(sampleAlpha) + "\n");
    file.write("optimizerType=" + str(optimizerType) + "\n");
    file.write("bpreg=" + str(bpreg) + "\n");
    file.write("sigma=" + str(sigma) + "\n");
    file.write("initAsNormal=" + str(initAsNormal) + "\n");
    file.write("trainRandomOrder=" + str(trainRandomOrder) + "\n");
    file.write("timeSort=" + str(timeSort) + "\n");
    file.write("topN=" + str(topN) + "\n");
    file.write("sessionKey=" + str(sessionKey) + "\n");
    file.write("itemKey=" + str(itemKey) + "\n");
    file.write("timeKey=" + str(timeKey) + "\n");
    file.write("cuda=" + str(cuda) + "\n");
    file.close(); 
    print("Result Folder:{}".format(resultDir))
    return resultDir     

def fitAndEvalute(trainDataSet, validDataSet, resultDir, inputSize, outputSize, lossType, finalAct, nLayers,
                  hiddenSize, nEpochs, batchSize, negative, dropoutHidden, dropoutEmbed, lr, momentum, weightDecay,
                  embeddingDim, trainNSample, validNSample, sampleAlpha, optimizerType, bpreg, sigma, initAsNormal,
                  trainRandomOrder, timeSort, topN, sessionKey, itemKey, timeKey, cuda):
    
    resultDir = saveParams(resultDir, inputSize, outputSize, lossType, finalAct, nLayers,
                             hiddenSize, nEpochs, batchSize, negative, dropoutHidden, dropoutEmbed,
                             lr, momentum, weightDecay, embeddingDim, trainNSample, validNSample,
                             sampleAlpha, optimizerType, bpreg, sigma, initAsNormal, trainRandomOrder,
                             timeSort, topN, sessionKey, itemKey, timeKey, cuda)
    
    lossFunc = Loss.LossFunction(lossType=lossType, useCuda=cuda, bpreg=bpreg)
    
    trainGenerator = Dataset.DataGenerator(trainDataSet, batchSize=batchSize, nSample=trainNSample,
                                           sampleAlpha=sampleAlpha, timeSort=timeSort, trainRandomOrder=trainRandomOrder)
    validGenerator = Dataset.DataGenerator(validDataSet, batchSize=batchSize, nSample=validNSample,
                                           sampleAlpha=sampleAlpha, timeSort=timeSort, trainRandomOrder=trainRandomOrder)
    
    # Initialize the model
    model = Model.GRU4Rec(inputSize=inputSize, outputSize=outputSize, hiddenSize=hiddenSize,
                          nLayers=nLayers, batchSize=batchSize, negative=negative, embeddingDim=embeddingDim,
                          dropoutHidden=dropoutHidden, dropoutEmbed=dropoutEmbed, finalAct=finalAct,
                          sigma=sigma, initAsNormal=initAsNormal, cuda=cuda)
    
    # optimizer
    optimizer = Optimizer.Optimizer(model.parameters(), optimizerType=optimizerType, lr=lr,
                              weightDecay=weightDecay, momentum=momentum)        
    # trainer class
    trainer = Trainer.Trainer(model, trainGenerator=trainGenerator, validGenerator=validGenerator,
                          optim=optimizer, lossFunc=lossFunc, topN=topN, resultDir=resultDir)
    
    print('#### START TRAINING....')
    trainer.train(nEpochs)
