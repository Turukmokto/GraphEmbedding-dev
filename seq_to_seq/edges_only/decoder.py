import random

import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNNEdges(nn.Module):
    def __init__(self, embedding, hidden_size, rnn_hidden_size, num_layers, max_node, SOS_token, EOS_token, criterion):
        # output_size = node_dim
        super(DecoderRNNEdges, self).__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = max_node
        self.num_layers = num_layers
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.criterion = criterion
        self.gru = nn.GRU(hidden_size, rnn_hidden_size, num_layers, batch_first=True)
        self.hidden_to_embed = nn.Linear(rnn_hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, max_node)
        self.out.weight = self.embedding.weight
        self._init_weights()

    def forward(self, inputs, z, teacher_forcing_prob=0.5):
        num_steps = inputs.shape[0]
        X = z
        hidden = X.view(1, self.num_layers, self.rnn_hidden_size)
        input = torch.LongTensor([[self.SOS_token]])
        use_teacher_forcing = random.random() < teacher_forcing_prob
        loss = 0
        outputs = torch.zeros((1, num_steps), dtype=torch.long)
        if use_teacher_forcing:
            for i in range(1, num_steps):
                output, hidden = self._step(input, hidden)
                topv, topi = output.topk(1)
                outputs[:, i] = topi.detach().squeeze()
                loss += self.criterion(output, inputs.view(1, -1)[:, i])
                input = inputs.view(1, -1)[:, i].unsqueeze(dim=1)
        else:
            for i in range(1, num_steps):
                output, hidden = self._step(input, hidden)
                topv, topi = output.topk(1)
                input = topi.detach()
                outputs[:, i] = topi.detach().squeeze()
                loss += self.criterion(output, inputs.view(1, -1)[:, i])
                if input[0].item() == self.EOS_token:
                    break
        return loss, outputs

    def decode(self, embedding, data_len, check_loss=None):
        hidden = embedding.view(1, self.num_layers, self.rnn_hidden_size)
        input = torch.LongTensor([[self.SOS_token]])
        loss = 0
        outputs = torch.zeros((1, data_len + 1), dtype=torch.long)
        for i in range(1, data_len + 1):
            output, hidden = self._step(input, hidden)
            topv, topi = output.topk(1)
            input = topi.detach()
            outputs[:, i] = topi.detach().squeeze()
            if check_loss is not None:
                loss += self.criterion(output, check_loss.view(1, -1)[:, i])
            if input[0].item() == self.EOS_token:
                break
        if check_loss is None:
            return outputs.view(-1)[1:]
        else:
            return loss, outputs.view(-1)[1:]

    def _step(self, input, hidden):
        X = self.embedding(input)
        output, hidden = self.gru(X, hidden)
        output = F.log_softmax(self.out(self.hidden_to_embed(output.squeeze(dim=1))), dim=1)
        return output, hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)