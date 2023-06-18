import json
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32, float64
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import EmbeddingDataset
from autoencoder_model.all_attributes.autoencoder import VAE3
from graph import attribute_parameters, node_to_ops

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type(torch.DoubleTensor)
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_NODE = 3_000
train_losses = []
test_losses = []
min_vals = []
max_vals = []
pre_hidden_size = 4  # 4
hidden_size = 2  # 2
max_attrs = 7
N_EPOCHS = 500
lr = 1e-3


def generate(name):
    params = {
        'alpha': ['rand'],
        'axes': ['rand', -1, 3],
        'axis': ['choose', -1, 0, 1, 2, 3],
        'dilations': ['choose', -1, 0, 1, 2],
        'ends': ['rand', -1, 30],
        'epsilon': ['rand'],
        'group': ['rand', -1, 1536],
        'keepdims': ['choose', -1, 0, 1],
        'kernel_shape': ['choose', -1, 0, 1, 3, 5, 7, 9, 11],
        'mode': ['choose', -1, 0, 1, 2, 3, 4],
        'momentum': ['rand'],
        'op': ['rand', 0, 22],
        'output_shape': ['rand', -1, 802815],
        'pads': ['choose', -1, 0, 1, 2],
        'starts': ['rand', -1, 30],
        'steps': ['rand', -1, 30],
        'strides': ['rand', -1, 7],
        'value': ['rand', -1, 10],
        'perm': ['choose', -1, 0, 1, 2, 3, 4]
    }
    sequence = []
    if attribute_parameters[name]['len'] == 1:
        ids = [attribute_parameters[name]['pos']]
    else:
        ids = attribute_parameters[name]['pos']
    for id in ids:
        value = None

        # generate
        if params[name][0] == 'rand' and len(params[name]) == 1:
            value = random.random()
        elif params[name][0] == 'rand' and len(params[name]) > 1:
            value = random.randint(params[name][1], params[name][2])
        else:
            value = params[name][random.randint(1, len(params[name]) - 1)]

        # normalize
        if value == -1 or max_vals[id] == -1:
            value = 0.
        elif name == 'op':
            value = value / max_vals[id]
        elif name not in ['alpha', 'epsilon', 'momentum']:
            value = (value + 1.) / (max_vals[id] + 1.)

        # add
        sequence.append(value)
    return sequence


def train(models, optimizers, name):
    models[name].train()
    sum_loss = 0
    cnt = 0

    for i in range(1000):
        sequence = generate(name)
        optimizers[name].zero_grad()
        sequence = torch.tensor(sequence).view(1, -1)
        loss, out = models[name](sequence, True)
        loss.backward()
        optimizers[name].step()
        sum_loss += loss.item()
        cnt += 1

    return sum_loss / cnt


def evaluate(models, name, tests):
    models[name].eval()
    sum_loss = 0
    results = []

    for test in tests:
        sequence = torch.tensor(test).view(1, -1)
        loss, out = models[name](sequence, False)
        sum_loss += loss.item()
        out = out.view(-1)
        for i in range(len(out)):
            out[i] = float(out[i])
        results.append(out.tolist())

    return sum_loss / len(tests), results


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # Local
    paths = ['../../', 'models/']
    # CTlab
    # paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/autoencoder_model/all_attributes/models/']

    best_valid_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(f'{paths[0]}data/embeddings/min_max.json'):
        with open(f'{paths[0]}data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]

    models = {}
    optimizers = {}
    for name, attrs in attribute_parameters.items():
        if name in ['edge_list_len', 'edge_list']:
            continue
        mean = 0
        if name in ['alpha', 'epsilon', 'momentum']:
            std = 1e-5 / 2.
        elif name == 'op':
            std = (1. / max_vals[attrs['pos']]) / 2.
        else:
            std = (1. / (max_vals[attrs['pos'][0] if isinstance(attrs['pos'], list) else attrs['pos']] + 1.)) / 2.
        models[name] = VAE3(
            shapes=[attrs['len'], pre_hidden_size, hidden_size],
            init_mean=mean,
            init_std=std
        ).to(device)
        optimizers[name] = optim.Adam(models[name].parameters(), lr=attrs['lr'])

    N_EPOCHS = 10_000
    name = 'output_shape'
    tests = [[10. / 802816, 200. / 802816, 500. / 802816, 1000. / 802816]]
    # Eval bests model tests
    # models[name].load_state_dict(torch.load(f'{paths[1]}autoencoder_model_{name}.pt'))
    # test_loss, test_out = evaluate(models, name, tests)
    # print(test_out)

    # Load model and losses
    # models[name].load_state_dict(torch.load(f'{paths[1]}autoencoder_model_{name}.pt'))
    # with open(f'{paths[1]}losses_{name}.json', 'r') as f:
    #     all_losses = json.load(f)
    # train_losses = all_losses['train']
    # test_losses = all_losses['test']
    # N_EPOCHS = 1000

    for epoch in range(N_EPOCHS):
        train_loss = 0
        eval_loss = 0
        start_time = time.time()
        train_loss = train(models, optimizers, name)
        test_loss, test_out = evaluate(models, name, tests)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        models[name].eval()

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        torch.save(models[name].state_dict(), f'{paths[1]}autoencoder_model_{name}.pt')
        with open(f'{paths[1]}losses_{name}.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'test': test_losses}))
        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(models[name].state_dict(), f'{paths[1]}autoencoder_model_{name}_best.pt')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.9f} | Test Loss: {test_loss:.9f}')
        print(f'After epoch {epoch + 1}:\n{test_out}\n')
