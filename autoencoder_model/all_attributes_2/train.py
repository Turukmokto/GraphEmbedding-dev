import json
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import EmbeddingDataset
from autoencoder_model.all_attributes_2.vae import VAE
from graph import attribute_parameters, node_to_ops

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
NODE_EMBEDDING_DIMENSION = 113
ATTRIBUTES_POS_COUNT = 50
MAX_NODE = 3_000
train_losses = []
eval_losses = []
test_losses = []
min_vals = []
max_vals = []


def generate():
    row = [-1.] * ATTRIBUTES_POS_COUNT
    params = {
        'alpha': ['rand'],
        'axes': ['choose', 0, 1, 2, 3],
        'axis': ['choose', 0, 1, 2, 3],
        'dilations': ['choose', 1, 2],
        'ends': ['rand', -1, 30000],
        'epsilon': ['rand'],
        'group': ['choose', 1, 2],
        'keepdims': ['choose', 0, 1],
        'kernel_shape': ['choose', 1, 3, 5, 7, 9, 11],
        'mode': ['choose', 0, 1, 2, 3],
        'momentum': ['rand'],
        'op': ['choose', 0, 2, 6, 10, 12, 13, 15, 16, 18, 21],
        'output_shape': ['rand', 1, 802815],
        'pads': ['choose', 0, 1, 2],
        'starts': ['rand', -1, 30000],
        'steps': ['rand', -1, 30000],
        'strides': ['choose', 1, 2, 3, 4],
        'value': ['rand', 1, 10000],
        'perm': ['choose', 1, 2, 3, 4]
    }
    for param, val in params.items():
        if attribute_parameters[param]['len'] == 1:
            ids = [attribute_parameters[param]['pos']]
        else:
            ids = attribute_parameters[param]['pos']
        for id in ids:
            value = None
            if val[0] == 'rand' and len(val) == 1:
                value = random.random()
            elif val[0] == 'rand' and len(val) > 1:
                value = random.randint(val[1], val[2])
            else:
                value = val[random.randint(1, len(val) - 1)]
            row[id] = value
    return row


def create_sequence(inputs):
    operation_id = int(inputs[attribute_parameters['op']['pos']])
    sequence = []
    op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
    operation = node_to_ops[op_name]
    for attribute in operation['attributes']:
        if attribute_parameters[attribute]['len'] == 1:
            ids = [attribute_parameters[attribute]['pos']]
            defaults = [attribute_parameters[attribute]['default']]
        else:
            ids = attribute_parameters[attribute]['pos']
            defaults = attribute_parameters[attribute]['default']
        for i in range(len(ids)):
            if inputs[ids[i]] == -1.:
                if max_vals[ids[i]] == -1.:
                    sequence.append(0.)
                else:
                    sequence.append(float((defaults[i] + 1.) / (max_vals[ids[i]] + 1.)))
            else:
                sequence.append(float(inputs[ids[i]]))

    return operation_id, torch.tensor(sequence)


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(loader, models, optimizers):
    for model in models:
        if model is not None:
            model.train()
    epoch_loss = 0
    cnt = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)

        for row in data:
            operation_id, sequence = create_sequence(row[:ATTRIBUTES_POS_COUNT])
            if sequence.size(0) == 0:
                continue
            if float(sequence.max()) > 1.:
                print(f'ERROR-train! {sequence} -- {row[:ATTRIBUTES_POS_COUNT]}')
            cnt += 1
            if models[operation_id] is not None:
                sequence = sequence.view(1, -1)
                for optimizer in optimizers:
                    if optimizer is not None:
                        optimizer.zero_grad()
                outputs, mu, logvar = models[operation_id](sequence)
                loss = loss_fn(outputs, sequence, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[operation_id].parameters(), 1)
                optimizers[operation_id].step()
                epoch_loss += loss.item()

    for i in range(500):
        cnt += 1
        row = generate()

        for j in range(ATTRIBUTES_POS_COUNT):
            if j >= ATTRIBUTES_POS_COUNT or j == attribute_parameters['op']['pos']:
                continue
            if max_vals[j] == -1.:
                row[j] = 0.
                continue
            if row[j] > max_vals[j]:
                row[j] = max_vals[j]
            row[j] = (row[j] + 1.) / (max_vals[j] + 1.)

        for j in range(ATTRIBUTES_POS_COUNT):
            if j == attribute_parameters['op']['pos']:
                continue
            if max_vals[j] == -1.:
                row[j] = float(max_vals[j])
                continue
            elif row[j] == -1:
                row[j] = -1.
                continue
            elif row[j] > max_vals[j]:
                row[j] = max_vals[j]
            elif row[j] < min_vals[j]:
                row[j] = min_vals[j]
            row[j] = (row[j] - min_vals[j]) / (max_vals[j] - min_vals[j])

        operation_id, sequence = create_sequence(row)
        if sequence.size(0) == 0:
            continue
        if float(sequence.max()) > 1.:
            print(f'ERROR-train-gen! {sequence} -- {row[:ATTRIBUTES_POS_COUNT]}')
        if models[operation_id] is not None:
            sequence = sequence.view(1, -1)
            for optimizer in optimizers:
                if optimizer is not None:
                    optimizer.zero_grad()
            outputs, mu, logvar = models[operation_id](sequence)
            loss = loss_fn(outputs, sequence, mu, logvar)
            loss.backward()
            if loss.item() > 1000000000.:
                print(f'LOSS! {sequence}, {loss.item()}')
            torch.nn.utils.clip_grad_norm_(models[operation_id].parameters(), 1)
            optimizers[operation_id].step()
            epoch_loss += loss.item()

    return epoch_loss / cnt


def evaluate(loader, models):
    for model in models:
        if model is not None:
            model.eval()
    epoch_loss = 0
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)

            for row in data:
                cnt += 1
                operation_id, sequence = create_sequence(row[:ATTRIBUTES_POS_COUNT])
                if sequence.size(0) == 0:
                    continue
                if models[operation_id] is not None:
                    sequence = sequence.view(1, -1)
                    outputs, mu, logvar = models[operation_id](sequence)
                    loss = loss_fn(outputs, sequence, mu, logvar)
                    epoch_loss += loss.item()

    return epoch_loss / cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github__com.teameo.ca/thuyngch/Variational-Autoencoder-PyTorch/blob/master/models/vae.py
    num_layers = 1
    N_EPOCHS = 30
    dropout = 0
    rnn_hidden_size = 30
    hidden_size = 64
    lr = 1e-3

    # Local
    # paths = ['../../', '']
    # CTlab
    paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/autoencoder_model/all_attributes_2/']

    best_valid_loss_total_mean = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
        train=True,
        normalize=True
    )
    test_dataset = EmbeddingDataset(
        root=f'{paths[0]}data/embeddings/',
        train=False,
        normalize=True
    )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    models = [None] * len(node_to_ops)
    optimizers = [None] * len(node_to_ops)
    for i in range(len(node_to_ops)):
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == i, node_to_ops))[0])
        operation = node_to_ops[op_name]
        total_len = 0
        for attribute in operation['attributes']:
            total_len += attribute_parameters[attribute]['len']
        if total_len > 0:
            models[i] = VAE(shapes=[total_len, 20, 15, 10]).to(device)
            optimizers[i] = optim.Adam(models[i].parameters(), lr=lr, weight_decay=1e-4)

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
                if j >= ATTRIBUTES_POS_COUNT or j == attribute_parameters['op']['pos']:
                    continue
                if test_input[i][j] == -1 or max_vals[j] == -1:
                    test_input[i][j] = -1.
                else:
                    test_input[i][j] = (test_input[i][j] + 1.) / (max_vals[j] + 1.)
    test_len = len(test_input)
    test_row = test_input[0]
    test_op_id, test_sequence = create_sequence(test_row[:ATTRIBUTES_POS_COUNT])

    best_valid_loss = float('inf')
    train_loss = 0
    eval_loss = 0
    test_loss_change = 0
    test_loss_last = 0

    print(f'Testing:\n{test_sequence.tolist()}\n')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0
        eval_loss = 0
        test_loss = 0
        train_loss += train(train_loader, models, optimizers)
        eval_loss += evaluate(test_loader, models)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if models[test_op_id] is not None:
            test_sequence = test_sequence.view(1, -1)
            test_outputs, mu, logvar = models[test_op_id](test_sequence)
            test_loss = loss_fn(test_outputs, test_sequence, mu, logvar)
            test_loss_change = test_loss - test_loss_last
            test_loss_last = test_loss
            print(f'After epoch {epoch + 1}:\n{test_outputs[0].tolist()}\n')
        else:
            print('Model is None\n')

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))

        with open(f'{paths[1]}losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            cnt = 0
            for model in models:
                if model is not None:
                    torch.save(model.state_dict(), f'{paths[1]}autoencoder_model_{cnt}.pt')
                cnt += 1
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}\n')
