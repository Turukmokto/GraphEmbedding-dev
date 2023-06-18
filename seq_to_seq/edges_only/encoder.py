import torch
from torch import nn


class EncoderRNNEdges(nn.Module):
    def __init__(self, embedding, hidden_size, rnn_hidden_size, num_layers):
        # input_size = node_dim
        super(EncoderRNNEdges, self).__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, rnn_hidden_size, num_layers, batch_first=True)
        self.hidden = None
        self._init_weights()

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.rnn_hidden_size)

    def forward(self, embedding):
        # embedding: (sequence_len)

        embedding = self.embedding(embedding).unsqueeze(0)
        # embedding: (1, sequence_len, hidden_size)

        _, self.hidden = self.gru(embedding, self.hidden)
        # hidden: (num_layers, batch_size, hidden_size)

        return self.hidden.view(1, -1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def _sample(self, mean, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z
