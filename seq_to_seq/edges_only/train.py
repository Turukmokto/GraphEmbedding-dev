import json
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from seq_to_seq.edges_only.encoder import EncoderRNNEdges
from seq_to_seq.edges_only.decoder import DecoderRNNEdges
from seq_to_seq.edges_only.seq2seq import Seq2SeqEdges

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_N = 3001
teacher_forcing = 0.5
SOS_token = torch.tensor([0])
EOS_token = torch.tensor([3000])
train_losses = []
eval_losses = []
test_losses = []


def fill_matrix(matrix):
    for i in range(MAX_N - matrix.size(1)):
        row = torch.tensor([[[-1.] * ATTRIBUTES_POS_COUNT]])
        matrix = torch.cat([matrix, row], dim=1)
    return matrix


def train(loader, model):
    model.train()
    epoch_loss = 0.
    for i, data in enumerate(loader):
        data = torch.LongTensor(data)
        # data: (batch_size, embedding_dim, node_dim)

        for row in data:
            data_len = int(row[ATTRIBUTES_POS_COUNT])
            if data_len <= 0:
                continue
            source = row[(ATTRIBUTES_POS_COUNT + 1):(ATTRIBUTES_POS_COUNT + data_len + 1)]
            # source: (batch_size, embedding_dim, node_dim)
            # target: (batch_size, embedding_dim, node_dim)
            source = torch.cat([SOS_token, source, EOS_token])
            loss, outputs = model.train_batch(source)
            # output: (embedding_dim, batch_size, node_dim)
            epoch_loss += loss

    return epoch_loss


def evaluate(loader, model):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = torch.LongTensor(data)
            # data: (batch_size, embedding_dim, node_dim)

            for row in data:
                data_len = int(row[ATTRIBUTES_POS_COUNT])
                if data_len <= 0:
                    continue
                source = row[(ATTRIBUTES_POS_COUNT + 1):(ATTRIBUTES_POS_COUNT + data_len + 1)]
                # source: (batch_size, embedding_dim, node_dim)
                # target: (batch_size, embedding_dim, node_dim)
                source = torch.cat([SOS_token, source, EOS_token])
                loss, outputs = model.eval_batch(source)
                # output: (embedding_dim, batch_size, node_dim)
                epoch_loss += loss

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/autoencoder/textrnnae.py
    # https://github.com/chrisvdweth/ml-toolkit/tree/master/pytorch/notebooks
    num_layers = 1
    N_EPOCHS = 10
    dropout = 0
    rnn_hidden_size = 30
    hidden_size = 64
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=True,
        normalize=False  # Do not change
    )
    test_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=False,
        normalize=False  # Do not change
    )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    test_input = []
    test_loss_last = 0
    with open(f'../../data/embeddings/test.json', 'r') as f:
        test_input = json.load(f)
    vals = []
    if os.path.isfile('../../data/embeddings/min_max.json'):
        with open(f'../../data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
        for i in range(len(test_input)):
            for j in range(NODE_EMBEDDING_DIMENSION):
                if j == ATTRIBUTES_POS_COUNT:
                    break
                if max_vals[j] == min_vals[j]:
                    test_input[i][j] = float(max_vals[j])
                elif test_input[i][j] == -1:
                    test_input[i][j] = -1.
                else:
                    test_input[i][j] = (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
    test_len = len(test_input)
    test_input = torch.from_numpy(np.array([test_input])).long()
    # test_input = torch.cat([SOS_token, test_input[0, 0, (ATTRIBUTES_POS_COUNT + 1):]], 1)

    # all_losses = []
    # with open(f'losses2.json', 'r') as f:
    #     all_losses = json.load(f)
    #
    # train_losses = all_losses['train']
    # eval_losses = all_losses['eval']
    # test_losses = all_losses['test']

    criterion = nn.NLLLoss()
    embedding = nn.Embedding(MAX_N, hidden_size)
    encoder = EncoderRNNEdges(embedding, hidden_size, rnn_hidden_size, num_layers).to(device)
    decoder = DecoderRNNEdges(embedding, hidden_size, rnn_hidden_size,
                              num_layers, MAX_N, SOS_token, EOS_token, criterion).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    model = Seq2SeqEdges(encoder, decoder, encoder_optimizer, decoder_optimizer).to(device)
    # model.load_state_dict(torch.load('seq2seq_model.pt'))

    print(f'Iterating: {hidden_size}, {dropout}, {lr}')

    best_valid_loss = float('inf')
    test_loss_change = 0
    test_loss_last = 0
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        test_loss = 0

        train_loss = float(train(train_loader, model))
        train_losses.append(train_loss)
        eval_loss = float(evaluate(test_loader, model))
        eval_losses.append(eval_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        data_len = 5
        test_source = torch.tensor([1, 2, 3, 4, 5])
        test_source = torch.cat([SOS_token, test_source, EOS_token])
        test_embedding = model.encode(test_source)
        test_loss, test_output = model.decode(test_embedding, data_len, test_source)
        test_losses.append(test_loss)
        print(test_output.tolist())

        with open(f'losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            model.save_models('encoder_model.pt', 'decoder_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.5f} | Eval Loss: {eval_loss:.5f} | Test Loss: {test_loss:.5f}, loss change: {test_loss_change:.5f}')
    print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
