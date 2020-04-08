import torch
import torch.nn as nn

def softmax_neg(X):
    hm = 1.0 - torch.eye(*X.size(), out=torch.empty_like(X))
    X = X * hm
    Xmax = X.max(axis=1, keepdim=True).values
    e_x = torch.exp(X - Xmax) * hm
    return e_x / e_x.sum(axis=1, keepdim=True)