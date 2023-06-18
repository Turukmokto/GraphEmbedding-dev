import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNNAttributes(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        # output_size = node_dim
        super(DecoderRNNAttributes, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(output_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input: (batch_size, node_dim)
        # context: (1, batch_size, node_dim)

        input = input.view(1, 1, -1)
        # input: (1, batch_size, node_dim)

        output, hidden = self.gru(input, hidden)
        # output: (1, batch_size, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)

        output = self.fc(output.squeeze(0))
        # output: (batch_size, output_size)

        return output, hidden