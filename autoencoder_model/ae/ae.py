import random

import torch
from torch import nn
import torch.nn.functional as F
from graph import MAX_NODE

# torch.set_default_tensor_type(torch.DoubleTensor)

class AE(nn.Module):
    def __init__(self, **kwargs):
        # params: input_shape == output_shape
        # input_shape % hidden_shape == 0
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][0], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][2]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["shapes"][2], out_features=kwargs["shapes"][1]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["shapes"][1], out_features=kwargs["shapes"][0]),
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)