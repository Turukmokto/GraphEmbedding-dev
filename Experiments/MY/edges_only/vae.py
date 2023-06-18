from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from graph import ATTRIBUTES_POS_COUNT, node_to_ops, attribute_parameters


class VAE3(nn.Module):
    def __init__(self, shapes, init_mean, init_std, vocab_size):
        super(VAE3, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, shapes[0])
        # self.vocab_size = vocab_size
        # self.embedding_dim = shapes[0]
        self.encoder = nn.Sequential(nn.Linear(shapes[0], shapes[1]), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(shapes[2], shapes[1]), nn.ReLU(),
            nn.Linear(shapes[1], shapes[0]), nn.Sigmoid()
        )
        self.mu = nn.Linear(shapes[1], shapes[2])
        self.gamma = nn.Linear(shapes[1], shapes[2])
        self.shapes = shapes
        self.init_mean = init_mean
        self.init_std = init_std
        # self.softmax = nn.LogSoftmax(dim=2)

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * 0.01
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        # [1, 2, 3, 4, 5, 6, 7, 8]
        # x = self.embedding(x)
        x = self.encoder(x)
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        mu = self.mu(x)
        gamma = self.gamma(x)
        return mu, gamma

    def decode(self, x):
        x = self.decoder(x)
        # batch_size = x.size(0)
        # x = x.view(batch_size, 8, -1)
        return x

    def latent(self, x, training=True):
        mu, gamma = self.encode(x)
        encoding = self.reparameterize(mu, gamma, training)
        return encoding

    def forward(self, inputs, training=True):
        mu, gamma = self.encode(inputs)
        encoding = self.reparameterize(mu, gamma, training)
        x = self.decode(encoding)
        # x = self.softmax(x)
        BCE, KLD = self.loss_function(inputs, x, mu, gamma)
        return BCE + 0.0005 * KLD, BCE, KLD / (inputs.size(0)), x.view(x.size(0), -1)

    def loss_function(self, input, output, mu, gamma, batch_size=1):
        BCE = F.binary_cross_entropy(output, input, reduction='sum')  # nll_loss
        KLD = -0.5 * torch.sum(1 + gamma - mu.pow(2) - gamma.exp())
        # KLD /= (input.size(0))  # * self.vocab_size
        return BCE, KLD
