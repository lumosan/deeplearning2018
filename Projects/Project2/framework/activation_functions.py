# -*- coding: utf-8 -*-

import numpy as np
from torch import Tensor

from framework.modules import Module

class ReLU(Module):
    """Implements the Rectified Linear Unit activation layer"""
    def forward(self, x):
        """Carries out the forward pass for backpropagation.
        INPUT
            x: input
        OUTPUT
            x after applying ReLU
        """
        # Clamp to 0
        self.x = x
        return self.x.clamp(min=0)

    def backward(self, grad):
        """
        Carries out the backward pass for backpropagation.
        INPUT:
            grad: gradient of the previous layer
        OUTPUT:
            grad after derivative of ReLU
        """
        # Derivative
        return grad * Tensor(np.where(self.x <= 0, 0, 1)).view(grad.size())



class LeakyReLU(Module):
    """Implements the Leaky ReLU activation layer"""
    def __init__(self, a=.001):
        self.a = a
    def forward(self, x):
        """Carries out the forward pass for backpropagation.
        INPUT
            x: input
        OUTPUT
            x after applying LeakyReLU
        """
        self.x = x
        # Apply activation
        return Tensor(np.where(x >= 0, x, self.a * x ))

    def backward(self, grad):
        """Carries out the backward pass for backpropagation.
        INPUT:
            grad: gradient of the previous layer
        OUTPUT:
            grad after derivative of LeakyReLU

        """
        # Derivative
        return grad * Tensor(np.where(self.x >= 0,
            1, self.a)).view(grad.size())



class Tanh(Module):
    """Implements the Tanh activation layer"""
    def forward(self, x):
        """Carries out the forward pass for backpropagation.
        INPUT
            x: input
        OUTPUT
            x after applying Tanh

        """
        # Apply activation
        self.x_tanh = x.tanh()
        return self.x_tanh

    def backward(self, grad):
        """Carries out the backward pass for backpropagation.
        INPUT:
            grad: gradient of the previous layer
        OUTPUT:
            grad after derivative of Tanh

        """
        # Derivative
        return grad * (1 - self.x_tanh ** 2).view(grad.size())



class Sigmoid(Module):
    """Implements the Sigmoid activation layer"""
    def forward(self, x):
        """Carries out the forward pass for backpropagation.
        INPUT
            x: input
        OUTPUT
            x after derivative of Sigmoid
        """
        # Apply activation
        self.sigmoid = (1 + (x / 2).tanh()) / 2 #With tanh to avoid overflow
        return self.sigmoid

    def backward(self, grad):
        """Carries out the backward pass for backpropagation.
        INPUT:
            grad: gradient of the previous layer
        OUTPUT:
            grad after applying Sigmoid

        """
        # Derivative
        out = grad * (self.sigmoid * (1 - self.sigmoid)).view(grad.size())
        return out
