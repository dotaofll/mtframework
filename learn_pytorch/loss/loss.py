from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn


class Loss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError(
                "Criterion func must be the subclass of 'nn.module.loss'")

        self.acc_loss = 0  # accumulated loss
        self.norm_term = 0  # normalization term

    def reset(self):

        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


class NLLLoss(Loss):

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask")
            weight[mask] = 0

        super(NLLLoss, self).__init__(self._NAME, nn.NLLLoss(
            weight=weight, size_average=self.size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.data.item()
        if self.size_average:

            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1


class Perplexity(NLLLoss):
    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None, size_average=False):
        super(Perplexity, self).__init__(
            weight=weight, mask=mask, size_average=size_average)

    def eval_batch(self, outputs: torch.Tensor, target: torch.Tensor):
        self.acc_loss = self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)
