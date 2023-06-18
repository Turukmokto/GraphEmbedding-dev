import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from graph import MAX_NODE, ATTRIBUTES_POS_COUNT


class Seq2SeqAutoencoder(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        batch_size,
        sos_idx,
        eos_idx,
        max_len,
        bidirectional=False,
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
    ):
        """Initialize model."""
        super(Seq2SeqAutoencoder, self).__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.src_vocab_size = src_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = src_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim
        )
        self.trg_embedding = nn.Embedding(
            src_vocab_size,
            trg_emb_dim
        )

        if self.bidirectional and self.nlayers > 1:
            self.encoder = DeepBidirectionalLSTM(
                self.src_emb_dim,
                self.src_hidden_dim,
                self.nlayers,
                self.dropout,
                True
            )

        else:
            hidden_dim = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim
            self.encoder = nn.LSTM(
                src_emb_dim,
                hidden_dim,
                nlayers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=self.dropout
            )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers_trg,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, src_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))

        return h0_encoder, c0_encoder

    def forward(self, input, ctx_mask=None, trg_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input)
        trg_emb = self.trg_embedding(input)

        if self.bidirectional and self.nlayers > 1:
            src_h, (src_h_t, src_c_t) = self.encoder(src_emb)
        else:
            self.h0_encoder, self.c0_encoder = self.get_state(input)

            src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, (self.h0_encoder, self.c0_encoder)
            )

        if self.bidirectional and self.nlayers == 1:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.view(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.view(
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)
                )
            )
        )
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )

        return decoder_logit

    def encode_one(self, input):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input)

        if self.bidirectional and self.nlayers > 1:
            src_h, (src_h_t, src_c_t) = self.encoder(src_emb)
        else:
            self.h0_encoder, self.c0_encoder = self.get_state(input)

            src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, (self.h0_encoder, self.c0_encoder)
            )

        if self.bidirectional and self.nlayers == 1:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))
        return (decoder_init_state.view(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ), c_t.view(self.decoder.num_layers, c_t.size(0), c_t.size(1)))

    def decode_one(self, input):
        (hid, cell) = input
        output = torch.Tensor(self.max_len).long()
        t = 0
        x = torch.Tensor(1).fill_(self.sos_idx).long()
        while t < self.max_len:
            emb = self.trg_embedding(torch.tensor(x.unsqueeze(1)))
            trg_h, (hid, cell) = self.decoder(emb, (hid, cell))
            decoder_logit = self.decoder2vocab(trg_h)
            decoded = self.decode(decoder_logit)
            x = decoded.argmax(dim=2)
            x = x[0]
            if x[0] == self.eos_idx:
                output = output[:t]
                break
            else:
                output[t] = x[0]
            t += 1
        return output

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.src_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=1)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class DeepBidirectionalLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""

    def __init__(
        self, input_size, hidden_size,
        num_layers, dropout, batch_first
    ):
        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.bi_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))

        h0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        return (h0_encoder_bi, c0_encoder_bi), \
            (h0_encoder, c0_encoder)

    def forward(self, input):
        """Propogate input forward through the network."""
        hidden_bi, hidden_deep = self.get_state(input)
        bilstm_output, (_, _) = self.bi_encoder(input, hidden_bi)
        return self.encoder(bilstm_output, hidden_deep)