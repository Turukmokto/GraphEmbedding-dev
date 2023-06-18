import random
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from graph import MAX_NODE, ATTRIBUTES_POS_COUNT, attribute_parameters, node_to_ops
from torch.autograd import Variable
import numpy as np

import random


class VAE1(nn.Module):
    def __init__(self, **kwargs):
        # params: input_shape == output_shape
        # input_shape % hidden_shape == 0
        super(VAE1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][0], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2]),
            nn.LeakyReLU(),
        )
        self.hidden2mu = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.hidden2log_var = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.alpha = 1
        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][0]),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(1, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return mu, log_var, self.decoder(hidden)

    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        return self.decoder(x)

    def training_step(self, inputs):
        mu, log_var, x_out = self.forward(inputs)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(inputs.view(-1), x_out.view(-1))
        loss = recon_loss * self.alpha + kl_loss
        return loss, x_out.view(-1)

    def one_encode(self, inputs):
        x = inputs.view(1, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return hidden



class VAE2(nn.Module):
    def __init__(self, shapes, negative_slope=0.01):
        super(VAE2, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(shapes[0], shapes[1])),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(shapes[1], shapes[2])),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))
        self.fc_mu = nn.Linear(shapes[2], shapes[3])
        self.fc_var = nn.Linear(shapes[2], shapes[3])
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(shapes[3], shapes[2])),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(shapes[2], shapes[1])),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(shapes[1], shapes[0])),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self._init_weights()

    def forward(self, x, training=True):
        if training:
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_var(h)
            z = self._reparameterize(mu, logvar)
            y = self.decoder(z)
            return y, mu, logvar
        else:
            z = self.represent(x)
            y = self.decoder(z)
            return y

    def represent(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self._reparameterize(mu, logvar)
        return z

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VAE3(nn.Module):
    def __init__(self, shapes, init_mean, init_std):
        super(VAE3, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(shapes[0], shapes[1]), nn.Tanh(), nn.Linear(shapes[1], shapes[2]), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(shapes[2], shapes[1]), nn.Tanh(), nn.Linear(shapes[1], shapes[0]), nn.Sigmoid())
        self.mu = nn.Linear(shapes[2], shapes[2])
        self.gamma = nn.Linear(shapes[2], shapes[2])
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
        BCE = F.mse_loss(output, input) # binary_cross_entropy
        KLD = -0.5 * torch.sum(1 + gamma - mu.pow(2) - gamma.exp())
        KLD /= batch_size * self.shapes[0]
        return BCE + KLD


class VAE(nn.Module):
    def __init__(self, shapes, init_mean, init_std, vocab_size):
        super(VAE, self).__init__()
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


class Encoder(nn.Module):
    def __init__(self, shapes):
        super(Encoder, self).__init__()
        # Build encoder model
        self.model = nn.Sequential(
            nn.Linear(shapes[0], shapes[1]),
            nn.Tanh(),
            nn.Linear(shapes[1], shapes[2]),
            nn.Tanh(),  # Sigmoid
        )

        # q(z|x)
        self.dense_mu_z = nn.Linear(shapes[2], shapes[2])
        self.dense_logvar_z = nn.Linear(shapes[2], shapes[2])
        self.N = torch.distributions.Normal(0, 1)

    def sampling(self, mu, log_var):
        # Reparameterize
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        z = mu + log_var * self.N.sample(mu.shape)
        # z = mu + eps * std
        return z

    def forward(self, x):
        out = self.model(x)
        # z
        mu_z = self.dense_mu_z(out)
        logvar_z = torch.exp(self.dense_logvar_z(out))
        z = self.sampling(mu_z, logvar_z)

        return mu_z, logvar_z, z


class Decoder(nn.Module):
    def __init__(self, shapes):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(shapes[2], shapes[1]),
            nn.Tanh(),
            nn.Linear(shapes[1], shapes[0])
        )

    def forward(self, x):
        return self.model(x)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="sum"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y)+self.eps)



class AE(nn.Module):
    def __init__(self, shapes):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(shapes[0], shapes[1]),
            nn.Tanh(),
            nn.Linear(shapes[1], shapes[2]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(shapes[2], shapes[1]),
            nn.Tanh(),
            nn.Linear(shapes[1], shapes[0]),
            # nn.Sigmoid(),
        )

    def loss(self, encoded, decoded):
        ae_loss = F.mse_loss(encoded, decoded, reduction='sum')
        return ae_loss

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        loss = self.loss(inputs, decoded)
        return loss, decoded.view(-1)