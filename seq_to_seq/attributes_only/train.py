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
from seq_to_seq.attributes_only.encoder import EncoderRNNAttributes
from seq_to_seq.attributes_only.decoder import DecoderRNNAttributes
from seq_to_seq.attributes_only.seq2seq import Seq2SeqAttributes

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_N = 3000
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([-2.])
# EOS_token = torch.tensor([[[1.] * (NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT)]])


def fill_matrix(matrix):
    for i in range(MAX_N - matrix.size(1)):
        row = torch.tensor([[[-1.] * ATTRIBUTES_POS_COUNT]])
        matrix = torch.cat([matrix, row], dim=1)
    return matrix


def train(loader, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        data = torch.from_numpy(np.array([data]))
        # data: (batch_size, embedding_dim, node_dim)

        for row in data[0]:
            source = target = model.create_sequence(row[:ATTRIBUTES_POS_COUNT])
            # source: (batch_size, embedding_dim, node_dim)
            # target: (batch_size, embedding_dim, node_dim)
            source = torch.cat([SOS_token, source])
            target = torch.cat([SOS_token, target])
            data_len = len(source)
            output = model(source, target, data_len)
            # output: (embedding_dim, batch_size, node_dim)
            output = output[1:, 0]
            target = target[1:]
            # output = (batch_size * (embedding_dim - 1), node_dim)
            # target: (batch_size * (embedding_dim - 1), node_dim)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader, model, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = torch.from_numpy(np.array([data]))
            # data: (batch_size, embedding_dim, node_dim)

            for row in data[0]:
                source = target = model.create_sequence(row[:ATTRIBUTES_POS_COUNT])
                # source: (batch_size, embedding_dim, node_dim)
                # target: (batch_size, embedding_dim, node_dim)
                source = torch.cat([SOS_token, source])
                target = torch.cat([SOS_token, target])
                data_len = len(source)
                output = model(source, target, data_len)
                # output: (embedding_dim, batch_size, node_dim)
                output = output[1:, 0]
                target = target[1:]
                # output = (batch_size * (embedding_dim - 1), node_dim)
                # target: (batch_size * (embedding_dim - 1), node_dim)
                loss = criterion(output, target)
                epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    num_layers = 1  # 2
    N_EPOCHS = 10
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=True,
        normalize=True
    )
    test_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=False,
        normalize=True
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
                    continue
                if max_vals[j] == min_vals[j]:
                    test_input[i][j] = float(max_vals[j])
                elif test_input[i][j] == -1:
                    test_input[i][j] = -1.
                else:
                    test_input[i][j] = (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
    test_len = len(test_input)
    test_input = torch.from_numpy(np.array([test_input]))[:, :, :(ATTRIBUTES_POS_COUNT + 1)]
    # test_input = torch.cat([SOS_token, test_input[0, 0, (ATTRIBUTES_POS_COUNT + 1):]], 1)

    dropout = 0
    hidden_size = 30  # [128, 256, 512, 1024]
    lr = 1e-3  # [1e-2, 1e-3]
    optim_func = optim.Adam

    # all_losses = []
    # with open(f'attributes_losses.json', 'r') as f:
    #     all_losses = json.load(f)
    #
    # train_losses = all_losses['train']
    # eval_losses = all_losses['eval']
    # test_losses = all_losses['test']

    train_losses = []
    eval_losses = []
    test_losses = []

    print(f'{hidden_size}, {dropout}, {optim_func}, {lr}, {reduction}')
    encoder = EncoderRNNAttributes(1, hidden_size,
                                   num_layers, dropout).to(device)
    decoder = DecoderRNNAttributes(1, hidden_size,
                                   num_layers, dropout).to(device)
    model = Seq2SeqAttributes(encoder, decoder).to(device)
    # model.load_state_dict(torch.load('seq2seq_model_2.pt'))
    optimizer = optim_func(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    best_valid_loss = float('inf')
    test_loss_change = 0
    test_loss_last = 0
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        test_loss = 0

        train_loss += train(train_loader, model, optimizer, criterion, clip=1)
        eval_loss += evaluate(test_loader, model, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        for row in test_input[0]:
            data_len, test_embedding = model.encode(row)
            test_output = model.decode(test_embedding, data_len)
            # test_loss += criterion(test_output[1:, 0], test_seq[1:])
        test_loss_change = test_loss - test_loss_last
        test_loss_last = test_loss

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))
        with open(f'losses2.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), f'seq2seq_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.5f} | Eval Loss: {eval_loss:.5f} | Test Loss: {test_loss:.5f}, loss change: {test_loss_change:.5f}')
    print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
