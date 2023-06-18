import random

import torch
from torch import nn
from graph import MAX_NODE


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_output_size = self.decoder.output_size
        outputs = torch.zeros(target_len, batch_size, target_output_size)

        hidden, cell = self.encoder(source)
        context_hidden, context_cell = hidden, cell

        input = target[:, 0, :]  # SOS_token
        # input: (batch_size, node_dim)

        for i in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell, context_hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = target[:, i, :] if random.random() < teacher_forcing_ratio else output

        return outputs, context_hidden, context_cell

    def encode(self, source):
        # source: (batch_size, embedding_dim, node_dim)
        encoder_hidden, encoder_cell = self.encoder(source)
        hidden = torch.cat([torch.tensor(source.size(1)).view(1, 1, -1), encoder_hidden.view(1, 1, -1), encoder_cell.view(1, 1, -1)], 2)
        return hidden.view(-1)

    def decode(self, embedding, node_embedding_size, SOS_token, batch_size=1):
        embedding_len = int(embedding[0])
        input_len = (embedding.size()[0] - 1) // 2
        context = hidden = embedding[1:(input_len + 1)].view(1, 1, -1)
        cell = embedding[(input_len + 1):].view(1, 1, -1)
        input = SOS_token.view(1, -1)
        outputs = torch.zeros(embedding_len, node_embedding_size)
        for i in range(embedding_len):
            output, hidden, cell = self.decoder(input, hidden, cell, context)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output.squeeze(0)
            input = output

        return outputs