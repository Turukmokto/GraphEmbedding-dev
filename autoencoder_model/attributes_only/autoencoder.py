import random

import torch
from torch import nn
import torch.nn.functional as F
from graph import MAX_NODE, ATTRIBUTES_POS_COUNT, attribute_parameters, node_to_ops


class VAE(nn.Module):
    def __init__(self, **kwargs):
        # params: input_shape == output_shape
        # input_shape % hidden_shape == 0
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][0], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2]),
            nn.LeakyReLU()
        )
        self.hidden2mu = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.hidden2log_var = nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][2])
        self.alpha = 1
        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][0]),
            nn.Tanh()
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
        recon_loss = recon_loss_criterion(inputs, x_out.view(-1))
        loss = recon_loss * self.alpha + kl_loss
        return loss, x_out.view(-1)

    def one_encode(self, inputs):
        x = inputs.view(1, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return hidden

    @staticmethod
    def create_sequence(inputs):
        operation_id = int(inputs[attribute_parameters['op']['pos']])
        sequence = []
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
        operation = node_to_ops[op_name]
        output_shape = []
        for attribute in operation['attributes']:
            if attribute_parameters[attribute]['len'] == 1:
                ids = [attribute_parameters[attribute]['pos']]
                defaults = [attribute_parameters[attribute]['default']]
            else:
                ids = attribute_parameters[attribute]['pos']
                defaults = attribute_parameters[attribute]['default']
            for i in range(len(ids)):
                if inputs[ids[i]] == -1.:
                    if attribute == 'output_shape':
                        output_shape.append(float(defaults[i]))
                    else:
                        sequence.append(float(defaults[i]))
                else:
                    if attribute == 'output_shape':
                        output_shape.append(float(inputs[ids[i]]))
                    else:
                        sequence.append(float(inputs[ids[i]]))

        return operation_id, output_shape, torch.tensor(sequence)

    @staticmethod
    def get_sequence(operation_id, output_shape, embedding):
        result = [-1.] * ATTRIBUTES_POS_COUNT
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
        operation = node_to_ops[op_name]
        result[attribute_parameters['op']['pos']] = operation_id

        step = 0
        for attribute in operation['attributes']:
            if attribute_parameters[attribute]['len'] == 1:
                ids = [attribute_parameters[attribute]['pos']]
            else:
                ids = attribute_parameters[attribute]['pos']
            for id in ids:
                result[id] = float(embedding[step])
            step += 1

        return torch.tensor(result)