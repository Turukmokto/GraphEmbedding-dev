import torch
from torch import nn


class EncoderRNNAttributes(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # input_size = node_dim
        super(EncoderRNNAttributes, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding):
        # embedding: (batch_size, embedding_dim, node_dim)

        embedding = embedding.view(1, -1, 1)
        # embedding: (batch_size, embedding_dim, node_dim)

        output, hidden = self.gru(embedding)
        # output: (batch_size, embedding_dim, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        return hidden