import random

import torch
from torch import nn
from graph import MAX_NODE, ATTRIBUTES_POS_COUNT


class Seq2SeqEdges(nn.Module):
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer):
        super(Seq2SeqEdges, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_batch(self, inputs):
        num_steps = inputs.shape[0]
        self.encoder.hidden = self.encoder.init_hidden()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        z = self.encoder(inputs)
        loss, outputs = self.decoder(inputs, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / num_steps, outputs

    def eval_batch(self, inputs):
        num_steps = inputs.shape[0]
        self.encoder.hidden = self.encoder.init_hidden()
        z = self.encoder(inputs)
        loss, outputs = self.decoder(inputs, z, 0)
        return loss.item() / num_steps, outputs

    def save_models(self, encoder_file_name, decoder_file_name):
        torch.save(self.encoder.state_dict(), encoder_file_name)
        torch.save(self.decoder.state_dict(), decoder_file_name)

    def encode(self, inputs):
        self.encoder.hidden = self.encoder.init_hidden()
        z = self.encoder(inputs)
        return z.view(-1)

    def decode(self, embedding, data_len, check_loss=None):
        loss, outputs = self.decoder.decode(embedding, data_len, check_loss)
        return loss.item() / data_len, outputs