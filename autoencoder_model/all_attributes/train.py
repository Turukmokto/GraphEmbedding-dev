import json
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32, float64
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn

from dataset import EmbeddingDataset
from autoencoder_model.all_attributes.autoencoder import VAE1, VAE2, VAE3, AE
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
recon_losses = []
kl_losses = []
total_losses = []
min_vals = []
max_vals = []
pre_hidden_size = 4  # 4
hidden_size = 2  # 2
max_attrs = 7
b1 = 0.5
b2 = 0.999
num_layers = 1
N_EPOCHS = 300
dropout = 0
rnn_hidden_size = 30
lr = 1e-3
weight_decay = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphDataset(Dataset):
    def __init__(self):
        self.dataset = []
        with open(f'{paths[0]}data/final_structures6.pkl', 'rb') as f:
            train_data, test_data, graph_args = pickle.load(f)
            for i in range(len(train_data)):
                for j in range(8):
                    self.dataset.append(
                        torch.tensor(to_one_hot(train_data[i][0].vs[j]['type']), dtype=float64).to(device)
                    )
            for i in range(len(test_data)):
                for j in range(8):
                    self.dataset.append(
                        torch.tensor(to_one_hot(test_data[i][0].vs[j]['type']), dtype=float64).to(device)
                    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def to_one_hot(num):
    result = [0.] * 8
    result[num] = 1.
    return result


def generate():
    row = [-1.] * ATTRIBUTES_POS_COUNT
    params = {
        'alpha': ['rand'],
        'axes': ['choose', -1, 0, 1, 2, 3],
        'axis': ['choose', -1, 0, 1, 2, 3],
        'dilations': ['choose', -1, 0, 1, 2],
        'ends': ['rand', -1, 30000],
        'epsilon': ['rand'],
        'group': ['rand', -1, 1536],
        'keepdims': ['choose', -1, 0, 1],
        'kernel_shape': ['choose', -1, 0, 1, 3, 5, 7, 9, 11],
        'mode': ['choose', -1, 0, 1, 2, 3, 4],
        'momentum': ['rand'],
        'op': ['rand', 0, 22],
        'output_shape': ['rand', -1, 802815],
        'pads': ['choose', -1, 0, 1, 2],
        'starts': ['rand', -1, 30000],
        'steps': ['rand', -1, 30000],
        'strides': ['rand', -1, 7],
        'value': ['rand', -1, 10],
        'perm': ['choose', -1, 0, 1, 2, 3, 4]
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
    # if random.randint(1, 5) == 1:
    #     row[19] = 6 # TOOD: remove
    return row


def create_sequence(inputs, models, optimizers, train=True):
    operation_id = round(float(inputs[attribute_parameters['op']['pos']]) * max_vals[attribute_parameters['op']['pos']])
    op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
    operation = node_to_ops[op_name]
    result = []
    cnt = 0
    sum_loss = 0
    for attribute in operation['attributes']:
        sequence = []
        if attribute_parameters[attribute]['len'] == 1:
            ids = [attribute_parameters[attribute]['pos']]
        else:
            ids = attribute_parameters[attribute]['pos']
        for i in range(len(ids)):
            if inputs[ids[i]] == -1. or max_vals[ids[i]] == -1.:
                sequence.append(0.)
            else:
                sequence.append(inputs[ids[i]])

        if train:
            optimizers[attribute].zero_grad()
        sequence = torch.tensor(sequence).view(1, -1)
        loss, out = models[attribute](sequence, train)
        if train:
            loss.backward()
            optimizers[attribute].step()

        cnt += 1
        sum_loss += loss.item()
        result.extend(out.tolist())
    return sum_loss, cnt, result


def train(models, optimizers, loader):
    for attr, model in models.items():
        model.train()
    epoch_loss = 0
    cnt = 0
    BCE_loss = 0
    KLD_loss = 0
    losses = 0

    optimizers['op'].zero_grad()
    for i, data in enumerate(loader):
        # row = data.unsqueeze(0)
        # rows = get_edge_list(data.tolist())
        source = data
        optimizers['op'].zero_grad()
        loss, BCE, KLD, out = models['op'](source, True)
        losses += loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizers['op'].step()
        epoch_loss += loss.item()
        BCE_loss += BCE.item()
        KLD_loss += KLD.item()

    return epoch_loss / len(dataset), BCE_loss / len(dataset), KLD_loss / len(dataset)


def evaluate(loader, models):
    for attr, model in models.items():
        model.eval()
    epoch_loss = 0
    all_cnt = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            for row in data:
                sum_loss, _, result = create_sequence(row[:ATTRIBUTES_POS_COUNT], models, optimizers, False)
                epoch_loss += sum_loss
                all_cnt += 1

    return epoch_loss / all_cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github.com/reoneo97/vae-playground/blob/main/models/vae.py
    # https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-13dbc559ba4b

    # Local
    paths = ['../../', 'models/']
    # CTlab
    # paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/autoencoder_model/all_attributes/models/']

    dataset = GraphDataset()
    loader = DataLoader(dataset=dataset, batch_size=32)

    best_valid_loss_total_mean = float('inf')
    # train_dataset = EmbeddingDataset(
    #     root=f'{paths[0]}data/embeddings/',
    #     train=True,
    #     normalize=True
    # )
    # test_dataset = EmbeddingDataset(
    #     root=f'{paths[0]}data/embeddings/',
    #     train=False,
    #     normalize=True
    # )
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=1,
    #                           shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=1,
    #                          shuffle=False)

    # Read test
    # with open(f'{paths[0]}data/embeddings/test.json', 'r') as f:
    #     test_input = json.load(f)
    #
    #     # TODO:remove
    #     # test_input = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.001, -1, -1, -1, -1, -1, 0.9, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    #
    # vals = []
    # if os.path.isfile(f'{paths[0]}data/embeddings/min_max.json'):
    #     with open(f'{paths[0]}data/embeddings/min_max.json', 'r') as f:
    #         vals = json.load(f)
    #     min_vals = vals[0]
    #     max_vals = vals[1]
    #     for i in range(len(test_input)):
    #         for j in range(NODE_EMBEDDING_DIMENSION):
    #             if j >= ATTRIBUTES_POS_COUNT:
    #                 continue
    #             if j == attribute_parameters['op']['pos']:
    #                 test_input[i][j] = test_input[i][j] / max_vals[j]
    #             elif test_input[i][j] == -1 or max_vals[j] == -1:
    #                 test_input[i][j] = 0.
    #             elif j not in [attribute_parameters['alpha']['pos'], attribute_parameters['epsilon']['pos'],
    #                                attribute_parameters['momentum']['pos']]:
    #                 test_input[i][j] = (test_input[i][j] + 1.) / (max_vals[j] + 1.)
    # test_len = len(test_input)
    # test_row = test_input[0]
    # test_operation_id = round(float(test_row[attribute_parameters['op']['pos']]) * max_vals[attribute_parameters['op']['pos']])
    # op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == test_operation_id, node_to_ops))[0])
    # operation = node_to_ops[op_name]
    # test_seq = []
    # cnt = 0
    # sum_loss = 0
    # for attribute in operation['attributes']:
    #     sequence = []
    #     if attribute_parameters[attribute]['len'] == 1:
    #         ids = [attribute_parameters[attribute]['pos']]
    #     else:
    #         ids = attribute_parameters[attribute]['pos']
    #     for i in range(len(ids)):
    #         if test_row[ids[i]] == -1. or max_vals[ids[i]] == -1.:
    #             test_seq.append(0.)
    #         else:
    #             test_seq.append(test_row[ids[i]])
    # test_seq = torch.tensor(test_seq)


    # Models and optimizers
    models = {}
    optimizers = {}
    schedulers = {}
    for name, attrs in attribute_parameters.items():
        if name in ['edge_list_len', 'edge_list']:
            continue
        mean = 0.
        std = 1.
        # if name in ['alpha', 'epsilon', 'momentum']:
        #     std = 1e-4
        # elif name == 'op':
        #     std = 1. / 7. # / max_vals[attrs['pos']]
        # else:
        #     std = 1.
            # std = 1. / (max_vals[attrs['pos'][0] if isinstance(attrs['pos'], list) else attrs['pos']] + 1.)
        models[name] = VAE3(
            shapes=[8, pre_hidden_size, hidden_size],
            init_mean=mean,
            init_std=std,
            vocab_size=8,
        ).to(device)
        optimizers[name] = optim.Adam(models[name].parameters(), lr=attrs['lr'])  # , betas=(b1, b2)
        schedulers[name] = ReduceLROnPlateau(optimizers[name], 'min', factor=0.1, patience=5, verbose=True)

    # best_valid_loss = float('inf')
    # train_loss = 0
    # eval_loss = 0
    # test_loss_change = 0
    # test_loss_last = 0

    # print(f'Testing:\n{test_seq.tolist()}\n')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        total_loss, recon_loss, kl_loss = train(models, optimizers, loader)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # for attr, model in models.items():
        #     model.eval()
        # test_loss, _, test_sequence = create_sequence(test_row[:ATTRIBUTES_POS_COUNT], models, optimizers, False)
        # test_loss_change = test_loss - test_loss_last
        # test_loss_last = test_loss
        # test_sequence = torch.tensor(test_sequence)
        # print(f'After epoch {epoch + 1}:\n{(test_sequence).tolist()}\n') # - test_seq

        recon_losses.append(float(recon_loss))
        kl_losses.append(float(kl_loss))
        total_losses.append(float(total_loss))

        with open(f'{paths[0]}Experiments/MY/attributes_losses.json', 'w') as f:
            f.write(json.dumps({'total': total_losses, 'recon': recon_losses, 'kl': kl_losses}))

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tRecon Loss: {recon_loss:.3f} | KL Loss: {kl_loss:.3f}\n')

        torch.save(models['op'].state_dict(), f'{paths[0]}Experiments/MY/attributes_model_op.pt')

        schedulers['op'].step(total_loss)  # TODO: all
        test_row = torch.tensor([to_one_hot(3)])
        test_loss, test_bce, test_kld, test_out = models['op'](test_row, False)
        # test_out = test_out.argmax(dim=2)
        test_result = []
        for i in range(8):
            if test_out[0][i] > 0.5:
                test_result.append(i)
        print(f'After epoch {epoch + 1}:\n{test_result}\n')

    # torch.save(model.state_dict(), 'seq2seq_model.pt')
