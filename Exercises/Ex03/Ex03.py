import torch
from torch import Tensor
import dlc_practical_prologue as prologue


# 1 Activation function
def sigma(x):
    return torch.tanh(x)
def dsigma(x):
    return torch.mul(torch.tanh(x), (torch.exp(-x)/(1 + torch.exp(-x))))


# 2 Loss
def loss(v, t):
    return torch.sum(torch.pow(v-t, 2))
def dloss(v, t):
    return torch.mul(t-v, -2)


# 3 Forward and backward passes
def forward_pass(w1, b1, w2, b2, x):
    """
    Inputs:
    w1, b1 weight and bias of first layer
    w2, b2 weight and bias of second layer
    x an input vector to the network
    
    Outputs:
    x, s1, x1, s2, x2
    """
    s1 = torch.mv(w1, x) + b1
    x1 = sigma(s1)
    s2 = torch.mv(w2, x1) + b2
    x2 = sigma(s2)
    return x, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
    t,
    x, s1, x1, s2, x2,
    dl_dw1, dl_db1, dl_dw2, dl_db2):
    """
    Inputs:
    w1, b1 weight and bias of first layer
    w2, b2 weight and bias of second layer
    t target vector
    x, s1, x1, s2, x2 the ones computed in forward_pass
    dl_dw1, dl_db1, dl_dw2, dl_db2 tensors used to store the
        cumulated sums of the gradient on individual samples
    """
    dl_dw1 = x * dsigma(s1)
    dl_db1 = dsigma(s1)
    dl_dw2 = x1 * dsigma(s2)
    dl_db2 = dsigma(s2)

# Define params
input_dim = 784
output_dim = 10
layer_n = 50

# Load data
train_input, train_target, test_input, test_target = load_data(one_hot_labels=True, normalize=True)

train_target = torch.mul(train_target, 0.9)
test_target = torch.mul(test_target, 0.9)

# Create weight and bias tensors
w1 = torch.Tensor(layer_n, input_dim).normal_(mean=0, std=1e-6)
b1 = torch.Tensor(layer_n).normal_(mean=0, std=1e-6)
w2 = torch.Tensor(output_dim, layer_n).normal_(mean=0, std=1e-6)
b2 = torch.Tensor(output_dim).normal_(mean=0, std=1e-6)

# Create tensors to sum up the gradients
# TODO

# Perform gradient steps with step size 0.1 divided by the number of training samples
for i in range(1000):
    
