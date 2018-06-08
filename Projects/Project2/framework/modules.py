# -*- coding: utf-8 -*-

import numpy as np
from torch import Tensor


################ Generic class ################
class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    def update(self, lr=None, values=None):
        pass



################ Implementations ################
class Linear(Module):
    """Implements the fully connected layer module

    It requires the number of inputs and outputs.

    Weights are initialized assuming that a ReLU module
    will be used afterwards. If a Tanh module will be used
    instead, it is recommended to set
    std_w = 1 / np.sqrt(n_input)

    It is possible to set a default learning rate that will be used
    during backpropagation if no other learning rate is stated.
    """
    def __init__(self, n_input, n_output, lr=1e-5,
        std_w=None, bias=True, std_b=0):
        """
        INPUT:
            n_input: Size of input
            n_output: Number of hidden units
            lr: If adaptive learning rate is not used, learning rate used for update
            std_w: Normal distribution initialization of weights with std = std_w/
                If None, std_w is chosen according to Xavier initialization.
            bias: If true, use bias
            std_b: If 0, initialize bias with 0.
                Otherwise, normal distribution with std = std_b
        """

        if std_w is None:
            # "Xavier" initialization
            std_w = 1 / np.sqrt(.5 * n_input)

        # Set parameters
        self.lr = lr
        self.w = Tensor(n_output, n_input).normal_(0, std_w)
        self.dw = Tensor(self.w.shape).zero_()
        self.bias = bias

        if bias:
            if not std_b:
                self.b = Tensor(n_output, 1).fill_(0)
            else:
                self.b = Tensor(n_output, 1).normal_(0, std_b)
            self.db = Tensor(self.b.shape).zero_()

    def forward(self, x):
        """Carries out the forward pass for backpropagation.
        INPUT:
            x: input with format number_samples x n_input
        OUTPUT
            output of layer with format number_samples x n_output
        """
        self.x = x
        # Obtain output
        self.s = self.w.mm(x.t())
        if self.bias:
            self.s += self.b
        # So its in expected format
        return self.s.t()

    def backward(self, grad):
        """Carries out the backward pass for backpropagation.
        It does not update the parameters.
        INPUT:
            grad: gradient w.r.t. previous layer
        OUTPUT:
            gradient w.r.t. current layer
        """
        # Propagate gradient
        out = grad.mm(self.w)
        self.dw = grad.t().mm(self.x)
        if self.bias:
            self.db = grad.sum(0).view(self.b.shape)
        return out

    def param(self):
        """Returns the list of parameters and gradients.

        OUTPUT:
            parameters
        """
        out = [(self.w, self.dw)]
        if self.bias:
            out.append((self.b, self.db))
        return out

    def update(self, lr=None, values=None):
        """Updates the parameters with the accumulated gradients.
        It must be called explicitly. If no lr is stated, the
        default lr of the module is used.
        INPUT:
            lr: if specified, used for update. Otherwise, use default
            values: if specified, these are the values that are multiplied by lr and
                subtracted from the parameters. Otherwise, the previously calculated
                gradient(s) is (are) used. The format is a list of 1 or 2 elements:
                1st one for updating w, and 2nd one for updating b (if bias is true)
        OUTPUT:
            gradient of current layer
        """
        # not adaptive learning rate
        if lr is None:
            lr = self.lr

        if values is None:
            self.w.add_(-lr * self.dw)
            if self.bias:
                self.b.add_(-lr * self.db)
                self.db = Tensor(self.b.shape).zero_()
        else:
            self.w.add_(-lr * values[0])
            self.dw = Tensor(self.w.shape).zero_()
            if self.bias:
                self.b.add_(-lr * values[1])
                self.db = Tensor(self.b.shape).zero_()


class Sequential(Module):
    """Allows to combine several modules sequentially
    It is possible to either include a loss module in the Sequential
    module or to not include it and use a loss module defined outside
    of the Sequential module instead (the second approach is recommended).
    """
    def __init__(self, layers, loss=None, train=True):
        """
        INPUT:
            layers: list of layers
            loss: loss function
            train:
        """
        self.layers = layers
        self.loss = loss
        self.train = train

    def forward(self, x, target=None):
        """Carries out the forward pass for backpropagation
        To do it it calls the forward functions of each individual
        module.
        INPUT:
            x: input with format number_samples x n_input
            target: target for the loss module (if specified)
        OUTPUT
            final output format number_samples x n_output
        """
        # Check that we have target if loss is specified
        if self.loss is not None:
            assert target is not None, "Target required for loss module"
        # Compute forward pass in every layer
        for l in self.layers:
            if self.train == False and type(l).__name__ == "Dropout":
                x = l.forward(x, self.train)
            else:
                x = l.forward(x)
        # If loss is specified, compute final output
        if self.loss is not None:
            x = self.loss.forward(x, target)
        self.x = x
        return x

    def backward(self, grad=None):
        """Carries out the backward pass for backpropagation
        It calls the backward functions of each individual module

        INPUT:
            grad: gradient of the loss (if loss function not specified)

        """
        # If we have a loss module, obtain gradient
        if self.loss is not None:
            grad = self.loss.backward()
        else:
            assert grad is not None, "Initial gradient required when no loss module defined"
        # Do the backward pass for all layers
        for l in reversed(self.layers):
            if self.train == False and type(l).__name__ == "Dropout":
                grad = l.backward(grad, self.train)
            else:
                grad = l.backward(grad)

    def param(self):
        """Returns the list of parameters and gradients.

        OUTPUT:
            parameters
        """
        # Obtain the parameters of each layer
        return [p for l in self.layers for p in l.param()]

    def update(self, lr=None, values=None):
        """Updates the parameters with the accumulated gradients.
        It must be called explicitly. If no lr is stated, the
        default lr of the module is used.
        INPUT:
            lr: if specified, used for update. Otherwise, use default
            values: if specified, these are the values that are multiplied by lr and
                subtracted from the parameters. Otherwise, the previously calculated
                gradients are used. The format is a list of as many elements as
                required by the architecture.
        OUTPUT:
            gradient of current layer
        """
        if values is None:
            for l in self.layers:
                l.update(lr)
        else:
            assert len(values) == len(self.param()), "Values should be length {}".format(
                len(self.param()))

            init_p = 0
            for l in self.layers:
                len_p = len(l.param())
                if len_p:
                    e_p = values[init_p:init_p + len_p]
                    l.update(lr, e_p)
                    init_p += len_p
                else:
                    l.update(lr)


class Dropout(Module):
    """Dropout module"""
    def __init__(self, drop_prob, train=True):
        """
        INPUT
            drop_prob: probability of dropping a hidden unit
        """
        self.drop_prob=drop_prob

    def forward(self, X, train=True):
        """Carries out the forward pass for backpropagation
        It creates a mask of units to shut down

        INPUT
            X: input
            train: boolean. If false, test mode and dropout should not be computed
        OUTPUT
            X with the units shutdown
        """
        if train == False:
            return X
        # Mask with size of input with random numbers between 0 and 1
        self.mask = Tensor(X.size()).uniform_()
        # Everything with a probability bigger than the drop probability is shutdown
        self.mask = (self.mask > self.drop_prob).float()
        # Shutdown neurons and normalize
        return (X * self.mask)/(1-self.drop_prob)

    def backward(self, grad, train=True):
        """Carries out the backward pass for backpropagation
        INPUT
            grad: gradient of the previous layer
            train: boolean. If false, test mode and dropout should not be computed
        OUTPUT
            grad with the units shutdown
        """
        if train:
            # Shutdown same neurons as in forward pass and normalize
            return (grad * self.mask)/(1-self.drop_prob)
        else:
            return grad
