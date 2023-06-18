import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # input_size = node_dim
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding):
        # embedding: (batch_size, embedding_dim, node_dim)

        embedding = self.dropout(embedding)
        # embedding: (batch_size, embedding_dim, node_dim)

        output, (hidden, cell) = self.lstm(embedding)
        # output: (batch_size, embedding_dim, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        return hidden, cell