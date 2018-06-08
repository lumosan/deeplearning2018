# -*- coding: utf-8 -*-

import numpy as np

from framework.modules import Module

class LossMSE(Module):
    """Implements the MSE loss computation"""
    def forward(self, output, target):
        """
        Carries out the forward pass for backpropagation.
        INPUT
            output: Tensor with output of the network
            target: Tensor with ground truth
        OUTPUT
            loss
        """
        #Compute loss
        self.diff = output.float() - target.float().view(output.size())
        return (self.diff ** 2).sum()

    def backward(self):
        """
        Carries out the backward pass for backpropagation.
        OUTPUT
            Gradient of loss
        """
        #  Gradient
        return self.diff * 2



class CrossEntropyLoss(Module):
    """Implements the Cross-Entropy loss computation"""
    def forward(self, output, target):
        """
        Carries out the forward pass for backpropagation.
        INPUT
            output: Tensor with output of the network
            target: Tensor with ground truth
        OUTPUT
            loss
        """
        self.target = target.float()
        self.output = output.float()
        # Loss with nan_to_num to avoid overflow
        return np.nan_to_num(-self.target * self.output.log() -
            (1 - self.target) * (1 - self.output).log()).sum()


    def backward(self):
        """Carries out the backward pass for backpropagation
        OUTPUT
            Gradient of loss
        """
        # Gradient
        return self.output - self.target
