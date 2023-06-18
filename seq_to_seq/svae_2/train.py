import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from seq_to_seq.svae_2.model import SentenceVAE

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_N = 3001
teacher_forcing_ratio = 0.5  # 1
SOS_token = torch.tensor([0])
EOS_token = torch.tensor([3000])
dropout = 0.
hidden_sizes = 10
lr = 1e-3
model = SentenceVAE(MAX_N, 256, 'gru', 128, 0, 0, 30, 0, 3000, 64)
NLL = torch.nn.NLLLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
steps = [0]


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight


def train(loader):
    model.train()
    epoch_loss = 0
    # for i, data in enumerate(loader):
        # data = torch.from_numpy(np.array([data])).long()
        # data: (batch_size, embedding_dim, node_dim)

        # for row in data[0]:
        #     data_len = int(row[ATTRIBUTES_POS_COUNT])
        #     if data_len <= 0:
        #         continue
        #     source = row[(ATTRIBUTES_POS_COUNT + 1):(ATTRIBUTES_POS_COUNT + data_len + 1)]
        #     source = torch.cat([SOS_token, source, EOS_token]).view(1, -1)
        #
        #     # TODO: len(source) --> data_len? SOS, EOS?
        #     logp, mean, logv, z = model(source, torch.tensor(len(source[0])).view(-1))
        #     NLL_loss, KL_loss, KL_weight = loss_fn(logp, source,
        #                                            torch.tensor(len(source[0])).view(-1), mean, logv, 'logistic', steps[0], 0.0025, 2500)
        #     loss = (NLL_loss + KL_weight * KL_loss)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     steps[0] += 1
        #     epoch_loss += loss.item()

    for step in range(500):
        optimizer.zero_grad()
        data_len = random.randint(1, 62)
        row = torch.tensor([random.randint(1, 2999) for i in range(data_len)]).long()
        input = torch.cat([SOS_token, row]).view(1, -1)
        target = torch.cat([row, EOS_token]).view(1, -1)

        logp, mean, logv, z = model(input, torch.tensor(len(input[0])).view(-1))
        NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
                                               torch.tensor(len(target[0])).view(-1), mean, logv, 'logistic', steps[0], 0.0025, 2500)
        loss = (NLL_loss + KL_weight * KL_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps[0] += 1
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = torch.from_numpy(np.array([data])).long()
            # data: (batch_size, embedding_dim, node_dim)

            for row in data[0]:
                data_len = int(row[ATTRIBUTES_POS_COUNT])
                if data_len <= 0:
                    continue
                source = row[(ATTRIBUTES_POS_COUNT + 1):(ATTRIBUTES_POS_COUNT + data_len + 1)]
                input = torch.cat([SOS_token, source]).view(1, -1)
                target = torch.cat([source, EOS_token]).view(1, -1)

                logp, mean, logv, z = model(input, torch.tensor(len(input[0])).view(-1))
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
                                                       torch.tensor(len(target[0])).view(-1), mean, logv, 'logistic', steps[0], 0.0025, 2500)
                loss = (NLL_loss + KL_weight * KL_loss)
                epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github.com/timbmg/Sentence-VAE
    num_layers = 1  # 2
    N_EPOCHS = 1000
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=True,
        normalize=False
    )
    test_dataset = EmbeddingDataset(
        root='../../data/embeddings/',
        train=False,
        normalize=False
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
    test_input = torch.from_numpy(np.array([test_input])).long()
    # test_input = torch.cat([SOS_token, test_input[0, 0, (ATTRIBUTES_POS_COUNT + 1):]], 1)

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

    best_valid_loss = float('inf')
    test_loss_change = 0
    test_loss_last = 0
    tensor = torch.Tensor
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        test_loss = 0

        train_loss += train(train_loader)
        eval_loss += evaluate(test_loader)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        test_row = torch.tensor([1, 2, 300, 1500, 2900]).long()
        test_row = torch.cat([test_row]).view(1, -1)
        # test_row = test_row.view(1, -1)
        model.eval()
        _,_,_,z = model(test_row, torch.tensor(len(test_row[0])).view(-1))
        samples, z = model.inference(z=z)
        test_print = []
        for i in range(len(samples[0])):
            test_print.append(int(samples[0][i]))
        print(f'After epoch {epoch + 1}:')
        print(test_print)

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))
        with open(f'losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), f'seq2seq_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.5f} | Eval Loss: {eval_loss:.5f}')
    print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')