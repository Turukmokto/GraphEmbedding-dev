from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from graph import ATTRIBUTES_POS_COUNT, node_to_ops, attribute_parameters


class VAE3(nn.Module):
    def __init__(self, shapes, init_mean, init_std):
        super(VAE3, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(shapes[0], shapes[1]), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(shapes[2], shapes[1]), nn.Tanh(), nn.Linear(shapes[1], shapes[0]), nn.Sigmoid())
        self.mu = nn.Linear(shapes[1], shapes[2])
        self.gamma = nn.Linear(shapes[1], shapes[2])
        self.shapes = shapes
        self.init_mean = init_mean
        self.init_std = init_std

    def reparameterize(self, mu, gamma, training=True):
        if training:
            sigma = torch.exp(0.5 * gamma)
            std_z = Variable(torch.from_numpy(np.random.normal(self.init_mean, self.init_std, size=sigma.size())).float())
            encoding = std_z.mul(sigma).add(mu)
            return encoding
        else:
            return mu

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        gamma = self.gamma(x)
        return mu, gamma

    def decode(self, x):
        return self.decoder(x)

    def latent(self, x, training=True):
        mu, gamma = self.encode(x)
        encoding = self.reparameterize(mu, gamma, training)
        return encoding

    def forward(self, inputs, training=True):
        mu, gamma = self.encode(inputs)
        encoding = self.reparameterize(mu, gamma, training)
        x = self.decoder(encoding)
        loss = self.loss_function(inputs, x, mu, gamma)
        return loss, x.view(-1)

    def loss_function(self, input, output, mu, gamma, batch_size=1):
        BCE = F.mse_loss(output, input)
        KLD = -0.5 * torch.sum(1 + gamma - mu.pow(2) - gamma.exp())
        KLD /= batch_size * self.shapes[0]
        return BCE + KLD