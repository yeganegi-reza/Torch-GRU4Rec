# Torch-GRU4REC
- PyTorch implementation of the algorithm of [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf). With the extensions introduced in  [Recurrent Neural Networks with Top-k Gains for Session-based Recommendations(CIKM 2018)](https://arxiv.org/abs/1706.03847)
- This code is based on [pyGRU4REC](https://github.com/yhs-968/pyGRU4REC) and [original Theano code written by the authors of the GRU4REC paper](https://github.com/hidasib/GRU4Rec) 
- This Version supports TOP1, BPR, TOP1-max, BPR-max, Negative Sampling, Cross-Entropy Losses and other parameters of [original Theano code](https://github.com/hidasib/GRU4Rec).
-  Model achieves the same performance reported in [original Theano code example](https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/run_rsc15.py) for the Recsys Challenge 2015 (RSC15) dataset.

## Requirements
- PyTorch 1.3.1
- Python 3.7
- pandas 0.24.2
- numpy 1.16.2
- jupyter  4.4.0

## Usage

### Dataset
- Download RecSys Challenge 2015 Dataset from [HERE](https://2015.recsyschallenge.com/)

- Extract Data and locate to `PATH_TO_ORIGINAL_DATA` directory. Default `( /Data )`

### Data Preprocessing 
For data preprocessing and data cleaning run `Preprocessing.py` or see ***Data Preprocessing*** section in `Run_GRU4Rec.ipynb.`

- The training set (`yoochoose-clicks.dat`) itself is divided into training, validation and testing sets where the testing is the last day sessions.
- After preprocessing, `rsc15Train.csv`, `rsc15Valid.csv`, `rsc15Test.csv` obtain that stores in `PATH_TO_PROCESSED_DATA` directory. Default `( /Data/Cleaned )`
- for use only 1/N data before preprocessing change N parameter. 

### Training And Testing Using Jupyter Notebook
See ``Run_GRU4Rec.ipnyb` that contains: 

1. **Loads Data** 

   - Load `rsc15Train.csv` , `rsc15Valid.csv` and create Dataset object

2. **Model Parameters**

   - Set model parameters. in this section parameters set to default like [original Theano code](https://github.com/hidasib/GRU4Rec).

   - The following list of parameters 

   - ```inputSize``` GRU Input Size = Number of Items In Dataset <br>
     ```outputSize``` GRU output Size = Number of Items In Dataset <br>
     ```hiddenSize``` Number of Neurons per GRU Layers (Default = 100) <br>
     ```nLayers``` Number of GRU Layers (Default = 1) <br>
     ```batchSize``` Mini Batch Size (Default = 32) <br>
     ```negative``` Use Negative Sampling In Training Process Or Not (Default = True) <br>
     ```embeddingDim``` Size Of The Embedding Used,  `embeddingDim <= 0` Means Not To Use Embedding (Default = -1) <br>
     ```dropoutHidden``` Dropout at each hidden layer (Default = 0.0)<br>
     ```dropoutEmbed``` Dropout Of The Input Units, Applicable Only If Embeddings Are Used (Default: 0.0)
     ```sigma``` "Width" Of Initialization. Either The standard Deviation Or The Min/Max Of The Initializations  Interval (With Normal And Uniform Initializations Respectively). 0 Means Adaptive Normalization (Sigma Depends On The Size Of The Weight Matrix) (Default: 0.0) <br>
     ```initAsNormal``` False: Initializations From Uniform Distribution On [-sigma,sigma]. True: Initializations From Normal Distribution On (0,sigma). (default: False)  <br>
     ```cuda``` Use GPU Or Not  <br>
     ```finalAct``` Activation Function (Default = Elu-1.0) <br>
     ```lossType``` Type of loss function TOP1 / BPR / TOP1-max / BPR-max / Cross-Entropy/ NLL (Default: BPR-max) <br>
     ```optimizerType``` Optimizer (Default = Adagrad)<br>
     ```lr``` Learning rate (Default = 0.1).<br>
     ```weightDecay```  Weight decay (Default = 0.0)<br>
     ```momentum``` Momentum Value (Default = 0.0) <br>
     ```bpreg```  Score Regularization Coefficient For The BPR-max Loss Function (Default: 1.0)<br>
     ```nEpochs``` Number of epochs (Default = 10)<br>
     ```timeSort``` Whether To Ensure The Order Of Sessions Is Chronological (Default: True)<br>
     ```trainRandomOrder``` Whether To Randomize The Order Of Sessions In Each Epoch (Default: False) <br>
     ```sampleAlpha``` The Probability Of An Item Used As An Additional Negative Sample (Default: 0.75)<br>

     ```trainNSample```  Number Of Additional Negative Samples To Be Used In Training Mini Batch Generator (Default: 2048) <br>

     ```validNSample```  Number Of Additional Negative Samples To Be Used In Validation Mini Batch Generator (Default: 2048) <br>

     ```sampleStore```  Number Of by Precomputing  Batch Of Negative Samples (Default: 10000000)<br>

     ```topN```  Value of K used durig Recall@K and MRR@K Evaluation (Default = 20)<br>

3. **BPR-max, no embedding**

   - train and evaluate the model with best parameters for BPR-max,no embedding

4. **BPR-max, constrained embedding**

   - train and evaluate the model with best parameters for BPR-max, constrained embedding

5. **Cross-entropy**

   - train and evaluate the model with best parameters for Log Softmax and Negative Likelihood Loss(Cross Entropy Loss)

6. **Testing**

   - Load `rsc15Test.csv` and create test Dataset object
   - Load trained Model and testing  