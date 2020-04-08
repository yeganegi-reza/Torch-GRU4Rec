import torch.optim as optim

class Optimizer:
    def __init__(self, params, optimizerType='adagrad', lr=0.05,
                 momentum=0, weightDecay=0, eps=1e-6):
        '''
        Parameters
        -----------
            params : 'torch.nn.Parameter'
                The NN parameters to optimize           
            optimizerType : string
                type of the optimizer to use (default : 'adagrad')
            lr : double
                learning rate (default : 0.05)
            momentum : double
                momentum for RMSProp optimizer (default : 0)
            weightDecay : double
                weight decay (default : 0)
            eps : double
                eps parameter 
        '''
        optimizerType = optimizerType.lower()
        if optimizerType == 'rmsprop':
            self.optimizer = optim.RMSprop(params, lr=lr, eps=eps, weight_decay=weightDecay, momentum=momentum)
        elif optimizerType == 'adagrad':
            self.optimizer = optim.Adagrad(params, lr=lr, weight_decay=weightDecay, eps=1e-6)
        elif optimizerType == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr=lr, eps=eps, weight_decay=weightDecay)
        elif optimizerType == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, eps=eps, weight_decay=weightDecay)
        elif optimizerType == 'sparseadam':
            self.optimizer = optim.SparseAdam(params, lr=lr, eps=eps)
        elif optimizerType == 'sgd':
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weightDecay=weightDecay)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
