import torch

def getRecall(indices, targets): 
    """
    Calculates the recall score for the given predictions and targets
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def getMrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    """
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr


def calc(indices, targets, k=20):
    """
    compute Recall@K, MRR@K scores.
    """
    _, indices = torch.topk(indices, k, -1)
    recall = getRecall(indices, targets)
    mrr = getMrr(indices, targets)
    return recall, mrr
