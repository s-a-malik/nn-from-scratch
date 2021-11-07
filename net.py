"""Neural Network Classes
"""

import numpy as np
import torch
from torch.autograd import Function

class Module:
    """Base class for all neural network modules.
    """

    def zero_grad(self):
        """zero out all the gradients
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def parameters(self):
        return []


class NN(Module):
    """Neural Network Class
    """

    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, initialisation, activation):
        super().__init__()

        self.num_layers = num_layers
        self.initialisation = initialisation
        self.activation = activation

        self.W1 = initialise(input_dim, hidden_dim, initialisation)
        self.b1 = initialise(1, hidden_dim, initialisation)
        self.W2 = initialise(hidden_dim, hidden_dim, initialisation)
        self.b2 = initialise(1, hidden_dim, initialisation)
        self.W3 = initialise(hidden_dim, output_dim, initialisation)
        self.b3 = initialise(1, output_dim, initialisation)

        if activation == 'relu':
            self.act = ReLU
        elif activation == 'sigmoid':
            self.act = Sigmoid
        else:
            raise ValueError('Activation function not supported')
        
    def __call__(self, x):
        """Forward pass of the neural network.
        """
        out = self.act.apply(x @ self.W1 + self.b1)
        if self.num_layers == 2:
            out = self.act.apply(out @ self.W2 + self.b2)
        # return probabilities
        return softmax(out @ self.W3 + self.b3)
    
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


# initialisations
def initialise(nin, nout, init_type):
    """Initialize weights
    """
    if init_type == 'xavier':
        return xavier_init(nin, nout)
    elif init_type == 'normal':
        return normal_init(nin, nout)
    elif init_type == 'uniform':
        return uniform_init(nin, nout)
    else:
        raise ValueError(f"Unknown initialisation type: {init_type}")


def xavier_init(nin, nout):
    """Initialize weights using Xavier initialization.
    """
    a = np.sqrt(6/(nin + nout))
    return torch.tensor(np.random.uniform(-a, a, (nin, nout)), requires_grad=True, dtype=torch.float32)

def normal_init(nin, nout):
    """Initialize weights using normal initialization.
    """
    return torch.tensor(np.random.normal(0, 1, (nin, nout)), requires_grad=True, dtype=torch.float32)

def uniform_init(nin, nout):
    """Initialize weights using uniform initialization between -1 and 1.
    """
    return torch.tensor(np.random.uniform(-1, 1, (nin, nout)), requires_grad=True, dtype=torch.float32)


# activations
class ReLU(Function):
    """ReLU activation function - need to implement the backward pass bc of max.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.max(torch.zeros_like(x), x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0  # zero gradients where x <= 0
        return grad_input

class Sigmoid(Function):
    """Sigmoid activation function
    """

    @staticmethod
    def forward(ctx, x):
        sig = 1/(1 + torch.exp(-x))
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx, grad_output):
        sig = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input *= sig * (1 - sig)
        return grad_input

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)
