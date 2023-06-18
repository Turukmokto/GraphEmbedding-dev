import json
import os
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset
from autoencoder_model.attributes_only.autoencoder import VAE
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


def train(loader, models, optimizers):
    for model in models:
        if model is not None:
            model.train()
    epoch_loss = 0
    for i, data in enumerate(loader):
        # data: (batch_size, embedding_dim, node_dim)

        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad()

        for row in data:
            operation_id, _, sequence = VAE.create_sequence(row[:ATTRIBUTES_POS_COUNT])
            if models[operation_id] is not None:
                loss, _ = models[operation_id].training_step(sequence)
                loss.backward()
                optimizers[operation_id].step()
                epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(loader, models):
    for model in models:
        if model is not None:
            model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)

            for row in data:
                operation_id, _, sequence = VAE.create_sequence(row[:ATTRIBUTES_POS_COUNT])
                if models[operation_id] is not None:
                    loss, _ = models[operation_id].training_step(sequence)
                    epoch_loss += loss.item()

    return epoch_loss / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github.com/reoneo97/vae-playground/blob/main/models/vae.py
    # https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-13dbc559ba4b
    num_layers = 1
    N_EPOCHS = 20
    dropout = 0
    rnn_hidden_size = 30
    hidden_size = 64
    lr = 1e-3

    best_valid_loss_total_mean = float('inf')
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

    models = [None] * len(node_to_ops)
    optimizers = [None] * len(node_to_ops)
    for i in range(len(node_to_ops)):
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == i, node_to_ops))[0])
        operation = node_to_ops[op_name]
        total_len = 0
        for attribute in operation['attributes']:
            if attribute == 'output_shape':
                continue
            total_len += attribute_parameters[attribute]['len']
        if total_len > 0:
            models[i] = VAE(shapes=[total_len, 10, 5]).to(device)
            optimizers[i] = optim.Adam(models[i].parameters(), lr=lr)

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
                if j >= ATTRIBUTES_POS_COUNT or j == attribute_parameters['op']['pos']:
                    continue
                if max_vals[j] == min_vals[j]:
                    test_input[i][j] = float(max_vals[j])
                elif test_input[i][j] == -1:
                    test_input[i][j] = -1.
                else:
                    test_input[i][j] = (test_input[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
    test_len = len(test_input)
    test_row = test_input[0]
    test_op_id, test_output_shape, test_sequence = VAE.create_sequence(test_row[:ATTRIBUTES_POS_COUNT])

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
            test_loss, test_output = models[test_op_id].training_step(test_sequence)
            test_loss_change = test_loss - test_loss_last
            test_loss_last = test_loss
            print(f'After epoch {epoch + 1}:\n{test_output.tolist()}\n')
        else:
            print('Model is None\n')

        train_losses.append(float(train_loss))
        eval_losses.append(float(eval_loss))
        test_losses.append(float(test_loss))

        with open(f'losses.json', 'w') as f:
            f.write(json.dumps({'train': train_losses, 'eval': eval_losses, 'test': test_losses}))

        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            cnt = 0
            for model in models:
                if model is not None:
                    torch.save(model.state_dict(), f'autoencoder_model_{cnt}.pt')
                cnt += 1
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f} | Test Loss: {test_loss:.3f}, loss change: {test_loss_change:.3f}\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
