import json
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset import EmbeddingDataset
from seq_to_seq.svae.svae import SentenceVAE

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
steps = [0]
model = SentenceVAE(MAX_N, 0, 3000, training=True)

# EOS_token = torch.tensor([[[1.] * (NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT)]])


def kl_anneal_function(step):
    k = 0.0025
    x0 = 2500
    weight = float(1 / (1 + np.exp(-k * (step - x0))))
    return weight


def loss_function(reconx, x, mu, logvar, step):
    # print(torch.argmax(torch.exp(reconx), dim=-1)[0][:50])
    x = x.view(-1).long()
    x = torch.cat([x, torch.tensor([3000] * (reconx.size(1) - x.size(0))).long()])
    reconx = reconx.view(-1, reconx.size(2))
    while reconx.size(0) < x.size(0):
        new_vec = torch.zeros(1, 3001)
        new_vec[0][3000] = 1
        new_vec = F.log_softmax(new_vec, dim=-1)
        reconx = torch.cat([reconx, new_vec])

    NLL_loss = criterion(reconx, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = kl_anneal_function(step)
    loss = NLL_loss + beta * KLD
    return loss, NLL_loss, KLD, beta


def train(loader, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        data = torch.from_numpy(np.array([data])).long()
        # data: (batch_size, embedding_dim, node_dim)

        for row in data[0]:
            optimizer.zero_grad()
            data_len = int(row[ATTRIBUTES_POS_COUNT])
            if data_len <= 0:
                continue
            source = row[(ATTRIBUTES_POS_COUNT + 1):(ATTRIBUTES_POS_COUNT + data_len + 1)]
            source = torch.cat([SOS_token, source, EOS_token]).view(1, -1)

            mu, logvar, x_emb = model.encode(source)
            z = model.reparameterize(mu, logvar)
            output, logp, z = model.inference(z)
            loss, _, _, _ = loss_function(logp, source, mu, logvar, steps[0])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            steps[0] += 1
            epoch_loss += loss.item()

    for step in range(500):
        optimizer.zero_grad()
        data_len = random.randint(2, 62)
        row = torch.tensor([random.randint(1, 2999) for i in range(data_len)]).long()
        source = torch.cat([SOS_token, row, EOS_token]).view(1, -1)
        mu, logvar, x_emb = model.encode(source)
        z = model.reparameterize(mu, logvar)
        output, logp, z = model.inference(z)
        loss, _, _, _ = loss_function(logp, source, mu, logvar, steps[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader, model, criterion):
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
                source = torch.cat([SOS_token, source, EOS_token]).view(1, -1)
                mu, logvar, x_emb = model.encode(source)
                z = model.reparameterize(mu, logvar)
                output, logp, z = model.inference(z)
                loss, _, _, _ = loss_function(logp, source, mu, logvar, steps[0])
                epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    num_layers = 1  # 2
    N_EPOCHS = 20
    best_valid_loss_total_mean = float('inf')
    best_valid_loss_total_sum = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Local
    paths = ['../../', '']
    # CTlab
    # paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/seq_to_seq/svae/']

    train_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
        train=True,
        normalize=False
    )
    test_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
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
    with open(f'{paths[0]}data/embeddings/test.json', 'r') as f:
        test_input = json.load(f)
    vals = []
    if os.path.isfile(f'{paths[0]}data/embeddings/min_max.json'):
        with open(f'{paths[0]}data/embeddings/min_max.json', 'r') as f:
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

    dropout = 0.
    hidden_sizes = 10
    lr = 1e-3  # 1e-3
    weight_decay = 1e-4

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

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss(reduction='sum')  # NLLLoss

    # test_row = torch.tensor([1, 2, 3]).long()
    # test_row = torch.cat([SOS_token, test_row, EOS_token]).view(1, -1)
    # mu, logvar, x_emb = model.encode(test_row)
    # z = model.reparameterize(mu, logvar)
    # output, logp, z = model.inference(z)
    # test_print = []
    # for i in range(len(output)):
    #     test_print.append(int(torch.exp(output[i]).argmax(0)))
    # print(test_print)
    # loss, _, _, _ = loss_function(logp, test_row, mu, logvar, steps[0])
    # keke = 0

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

        test_row = torch.tensor([1, 2, 3]).long()
        test_row = torch.cat([SOS_token, test_row, EOS_token]).view(1, -1)
        mu, logvar, x_emb = model.encode(test_row)
        z = model.reparameterize(mu, logvar)
        output, logp, z = model.inference(z)
        test_print = []
        for i in range(len(output)):
            test_print.append(int(torch.exp(output[i]).argmax(0)))
        print(f'After epoch {epoch + 1}:')
        print(test_print)
        loss, _, _, _ = loss_function(logp, test_row, mu, logvar, steps[0])
        test_loss_change = loss - test_loss_last
        test_loss_last = loss

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(loss))
        with open(f'{paths[1]}losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            torch.save(model.state_dict(), f'{paths[1]}seq2seq_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.5f} | Eval Loss: {eval_loss:.5f} | Test Loss: {loss:.5f}, loss change: {test_loss_change:.5f}')
    print('\n\n-------------------------------------\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')