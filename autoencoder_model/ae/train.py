import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim, float64
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from autoencoder_model.ae.ae import AE
from graph import attribute_parameters

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_NODE = 1_000
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([[[-1.] * NODE_EMBEDDING_DIMENSION]])
EOS_token = torch.tensor([[[1.] * NODE_EMBEDDING_DIMENSION]])


def train(loader, model, optimizer, criterion):
    model.train()
    epoch_loss = 0
    cnt = 0
    for i, data in enumerate(loader):
        for j in range(len(data)):
            row = torch.tensor(data[j][(ATTRIBUTES_POS_COUNT + 1):], dtype=float64)
            for k in range(len(row)):
                row[k] = (row[k] + 1.) / 3001.
            output = model(row)
            loss = criterion(output.view(-1), row)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            cnt += 1

    return epoch_loss / cnt


def evaluate(loader, model, criterion):
    model.eval()
    epoch_loss = 0
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            for j in range(len(data)):
                row = torch.tensor(data[j][(ATTRIBUTES_POS_COUNT + 1):], dtype=float64)
                for k in range(len(row)):
                    row[k] = (row[k] + 1.) / 3001.
                output = model(row)
                loss = criterion(output.view(-1), row)
                epoch_loss += loss.item()
                cnt += 1

    return epoch_loss / cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    num_layers = 1  # 2
    N_EPOCHS = 50
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root=f'../../data/embeddings/',
        train=True,
        normalize=True
    )
    test_dataset = EmbeddingDataset(
        root=f'../../data/embeddings/',
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
                if j >= ATTRIBUTES_POS_COUNT:
                    test_input[i][j] = (test_input[i][j] + 1.) / (3001.)
                    continue
                if j == attribute_parameters['op']['pos']:
                    test_input[i][j] = test_input[i][j] / max_vals[j]
                elif test_input[i][j] == -1 or max_vals[j] == -1:
                    test_input[i][j] = 0.
                elif j not in [attribute_parameters['epsilon']['pos'],
                                   attribute_parameters['momentum']['pos']]:
                    test_input[i][j] = (test_input[i][j] + 1.) / (max_vals[j] + 1.)
    test_len = len(test_input)
    test_input = torch.tensor(test_input, dtype=float64)

    dropouts = [0]
    hidden_sizes = [2048]  # [128, 256, 512, 1024]
    optimizers = [optim.Adam]  # [optim.Adam, optim.AdamW, optim.Adamax]
    learning_rates = [1e-3]  # [1e-2, 1e-3]
    reductions = ['mean']  # ['mean', 'sum']
    iter_num = 0
    train_losses = []
    eval_losses = []
    test_losses = []

    model = AE(shapes=[NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT - 1, 40, 30]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='sum')

    best_valid_loss = float('inf')
    train_loss = 0
    eval_loss = 0
    test_loss_change = 0
    test_loss_last = 0
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        train_loss += train(train_dataset, model, optimizer, criterion)
        eval_loss += evaluate(test_loader, model, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        test_loss = 0
        for j in range(len(test_input)):
            test_embedding = model.encode(test_input[j][(ATTRIBUTES_POS_COUNT + 1):])
            test_output = model.decode(test_embedding)
            test_loss += criterion(test_output.view(-1), test_input[j][(ATTRIBUTES_POS_COUNT + 1):])

        test_loss_change = test_loss - test_loss_last
        test_loss_last = test_loss

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))
        with open(f'edges_losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), f'autoencoder_model_edges.pt')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}')
        with open(f'logs.txt', 'a') as f:
            f.write(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
            f.write(
                f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}\n')
    # print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')