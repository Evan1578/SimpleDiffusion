import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.datasets import MNIST
from SyntheticDistributions.base_distributions import *


class ToyConditionalModel(nn.Module):

    def __init__(self, dim, config):
        super(ToyConditionalModel, self).__init__()
        self.dim = dim = dim[0]
        hidden_dim = config.hidden_dim
        self.noise_levels = config.noise_levels
        if config.activation == 'ReLU':
            self.activation = nn.ReLU()
        elif config.activation == 'Softplus':
            self.activation = nn.Softplus()
        else:
            raise Exception("Unknown activation function type")
        self.lin_in = nn.Linear(dim + 1, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.inner_layers1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.num_hidden1)])
        self.middle_layer = nn.Linear(hidden_dim + 1, hidden_dim)
        self.inner_layers2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.num_hidden2)])
        self.lin_out = nn.Linear(hidden_dim + 1, dim)

    def forward(self, x, noise_labels):
        noise_levels = torch.reshape(self.noise_levels[noise_labels], (x.size(dim=0), 1))
        x = torch.cat((x, noise_levels), dim=1)
        x = self.activation(self.lin_in(x))
        x = self.batch_norm(x)
        for layer in self.inner_layers1:
            x = self.activation(layer(x))
        x = torch.cat((x, noise_levels), dim=1)
        x = self.activation(self.middle_layer(x))
        for layer in self.inner_layers2:
            x = self.activation(layer(x))
        x = torch.cat((x, noise_levels), dim=1)
        x = self.lin_out(x)
        return x
