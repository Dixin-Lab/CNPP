"""
This script contains some classes used in our models
"""

import numpy as np
import torch
import torch.nn as nn
from constant import PAD

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float)#.unsqueeze(-1)

def lowtri(A):
    return torch.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + torch.triu(torch.ones_like(A)) * value
    return A

class MaxLogLike(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = sum_{i in batch}[ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """
    def __init__(self):
        super(MaxLogLike, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (1,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t+self.eps).log().sum() + Lambda_t.sum()


class MaxLogLikePerSample(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = [ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """
    def __init__(self):
        super(MaxLogLikePerSample, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (batchsize,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t[:, 0]+self.eps).log() + Lambda_t.sum(1)

class CrossEntropy(nn.Module):
    """
    The cross entropy loss that maximize the conditional probability of current event given its intensity
    ls = -sum_{i in batch} log p(c_i | t_i, c_js, t_js)
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, lambda_t, Lambda_t, c):
        return self.entropy_loss(Lambda_t, c[:, 0])




