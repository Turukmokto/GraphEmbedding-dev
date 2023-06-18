import json
import math
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim, float32, float64, int64
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

from dataset import EmbeddingDataset
from autoencoder_model.edges_only.vae import VAE3
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
                        torch.tensor(train_data[i][0].get_adjacency()[j], dtype=float64).to(device)
                    )
            for i in range(len(test_data)):
                for j in range(8):
                    self.dataset.append(
                        torch.tensor(test_data[i][0].get_adjacency()[j], dtype=float64).to(device)
                    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_edge_list(line):
    rows = torch.tensor([], dtype=int64)
    for i in range(len(line)):
        list = []
        row = [0] * 8
        for j in range(8):
            if line[i][j] > 0:
                list.append(j)
        for j in range(len(list)):
            row[j] = list[j]  # / 8.
        row = torch.LongTensor([row])
        rows = torch.cat([rows, row])
    return rows


def train(model, optimizer, loader):
    model.train()
    epoch_loss = 0
    BCE_loss = 0
    KLD_loss = 0
    cnt = 0
    losses = 0

    optimizer.zero_grad()
    for i, data in enumerate(loader):
        # row = data.unsqueeze(0)
        # rows = get_edge_list(data.tolist())
        source = data
        optimizer.zero_grad()
        loss, BCE, KLD, out = model(source, True)
        losses += loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        BCE_loss += BCE.item()
        KLD_loss += KLD.item()

    return epoch_loss / len(loader), BCE_loss / len(loader), KLD_loss / len(loader)


def evaluate(loader, model):
    model.eval()
    epoch_loss = 0
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            # data: (batch_size, embedding_dim, node_dim)

            for row in data:
                data_len = int(row[ATTRIBUTES_POS_COUNT])
                if data_len <= 0:
                    continue
                source = torch.tensor(row[(ATTRIBUTES_POS_COUNT + 1):], dtype=float64).view(1, -1)
                for j in range(len(source[0])):
                    if source[0][j] == -1.:
                        source[0][j] = 0.
                    else:
                        source[0][j] = source[0][j] / 16.
                cnt += 1
                loss, out = model(source, False)
                epoch_loss += loss.item()

    return epoch_loss / cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # https://github__com.teameo.ca/thuyngch/Variational-Autoencoder-PyTorch/blob/master/models/vae.py
    # Local
    paths = ['../../', '']
    # CTlab
    # paths = ['/nfs/home/vshaldin/embeddings/', '/nfs/home/vshaldin/embeddings/autoencoder_model/all_attributes_2/']

    best_valid_loss_total_mean = float('inf')

    dataset = GraphDataset()
    loader = DataLoader(dataset=dataset, batch_size=32)  # , num_workers=2

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

    model = VAE3(
            shapes=[8, 8, 4],
            init_mean=0,
            init_std=1,
            vocab_size=8
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    # scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=N_EPOCHS)

    # model.load_state_dict(torch.load(f'{paths[1]}test_vae_model.pt'))

    # with open(f'{paths[0]}data/embeddings/test.json', 'r') as f:
    #     test_input = json.load(f)
    # vals = []
    # if os.path.isfile(f'{paths[0]}data/embeddings/min_max.json'):
    #     with open(f'{paths[0]}data/embeddings/min_max.json', 'r') as f:
    #         vals = json.load(f)
    #     for i in range(len(test_input)):
    #         for j in range(NODE_EMBEDDING_DIMENSION):
    #             if j > ATTRIBUTES_POS_COUNT:
    #                 if test_input[i][j] == -1:
    #                     test_input[i][j] = 0.
    #                 else:
    #                     test_input[i][j] = max(1., test_input[i][j] / 8.)
    # test_len = len(test_input)
    # test_row = torch.tensor(test_input[0][(ATTRIBUTES_POS_COUNT + 1):])
    # test_row[0] = 7. / 8.
    # test_row[1] = 8. / 8.
    # test_row[2] = 1. / 8.
    # test_row[3] = 2. / 8.
    # test_row[4] = 3. / 8.
    # test_row[5] = 4. / 8.
    # test_row[6] = 5. / 8.
    # test_row[7] = 6. / 8.
    # test_row[8] = 8. / 8.

    # best_valid_loss = float('inf')
    # train_loss = 0
    # eval_loss = 0
    # test_loss_change = 0
    # test_loss_last = 0
    #
    # print(f'Testing:\n{test_row[:8].tolist()}\n')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        total_loss, recon_loss, kl_loss = train(model, optimizer, loader)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        recon_losses.append(float(recon_loss))
        kl_losses.append(float(kl_loss))
        total_losses.append(float(total_loss))

        with open(f'{paths[0]}Experiments/MY/edge_losses.json', 'w') as f:
            f.write(json.dumps({'total': total_losses, 'recon': recon_losses, 'kl': kl_losses}))

        torch.save(model.state_dict(), f'{paths[0]}Experiments/MY/edge_model.pt')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tRecon Loss: {recon_loss:.3f} | KL Loss: {kl_loss:.3f}')

        # Reduce lr step
        scheduler.step(total_loss)
        # scheduler.step()

        test_row = torch.tensor([[0., 1., 1., 0., 1., 1., 1., 0]])
        test_loss, test_bce, test_kld, test_out = model(test_row, False)
        # test_out = test_out.argmax(dim=2)
        for i in range(8):
            test_out[0][i] = 1. if test_out[0][i] > 0.5 else 0.
        print(f'After epoch {epoch + 1}:\n{test_out.tolist()[0]}\n')
