import os
import pickle
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Experiments.MY.all_attributes.autoencoder import VAE3 as attribute_VAE
from Experiments.MY.edges_only.vae import VAE3 as edge_VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type(torch.DoubleTensor)


class GraphDataset(Dataset):
    def __init__(self, train=True):
        self.dataset = []
        with open(f'../../data/final_structures6.pkl', 'rb') as f:
            train_data, test_data, graph_args = pickle.load(f)
            if not train:
                for i in range(len(train_data)):
                    for j in range(8):
                        self.dataset.append(
                            [
                                torch.tensor(to_one_hot(train_data[i][0].vs[j]['type']), dtype=torch.float64).to(device),
                                torch.tensor(train_data[i][0].get_adjacency()[j], dtype=torch.float64).to(device)
                            ]
                        )
            for i in range(len(test_data)):
                for j in range(8):
                    self.dataset.append(
                        [
                            torch.tensor(to_one_hot(test_data[i][0].vs[j]['type']), dtype=torch.float64).to(device),
                            torch.tensor(test_data[i][0].get_adjacency()[j], dtype=torch.float64).to(device)
                        ]
                    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def to_one_hot(num):
    result = [0.] * 8
    result[num] = 1.
    return result

max_num = 0
def is_same(a_in, a_recon, e_in, e_recon):
    global max_num
    max_num += 1
    a_in = a_in.tolist()
    a_recon = a_recon.tolist()
    e_in = e_in.tolist()
    e_recon = e_recon.tolist()
    for i in range(8):
        a_recon[i] = 1. if a_recon[i] > 0.5 else 0.
        e_recon[i] = 1. if e_recon[i] > 0.5 else 0.
    return 1 if a_in == a_recon and e_in == e_recon else 0


def is_valid(attr):
    attr_pos = []
    for i in range(8):
        if attr[i] > 0.5:
            attr_pos.append(i)
    return len(attr_pos) == 1


def extract_latent(model, data):
    model.eval()
    Z = []
    for i, g in enumerate(data):
        mu, _ = model.encode(g[0])
        mu = mu.cpu().detach().numpy()
        Z.append(mu)
    return np.concatenate(Z, 0)

encode_times = 10
decode_times = 10
n_perfect = 0


if __name__ == '__main__':
    train_dataset = GraphDataset()
    test_dataset = GraphDataset(train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32)

    model_attribute = attribute_VAE(
        shapes=[8, 4, 2],
        init_mean=0,
        init_std=1,
        vocab_size=8,
    ).to(device)
    model_edge = edge_VAE(
        shapes=[8, 8, 4],
        init_mean=0,
        init_std=1,
        vocab_size=8
    ).to(device)

    model_attribute.load_state_dict(torch.load('attributes_model_op.pt'))
    model_edge.load_state_dict(torch.load('edge_model.pt'))

    # Accuracy
    # for i, data in enumerate(test_loader):
    #     attribute = data[0]
    #     edge = data[1]
    #     attribute_mu, attribute_gamma = model_attribute.encode(attribute)
    #     edge_mu, edge_gamma = model_edge.encode(edge)
    #     for _ in range(encode_times):
    #         attribute_z = model_attribute.reparameterize(attribute_mu, attribute_gamma, False)
    #         edge_z = model_edge.reparameterize(edge_mu, edge_gamma, False)
    #         for _ in range(decode_times):
    #             attribute_recon = model_attribute.decode(attribute_z)
    #             edge_recon = model_edge.decode(edge_z)
    #             n_perfect += sum(is_same(a0, a1, e0, e1) for a0, a1, e0, e1 in zip(attribute, attribute_recon, edge, edge_recon))
    #
    # acc = n_perfect / max_num
    # print(f'Accuracy: {acc}')


    # Validity
    # n_latent_points = 1000
    # batch_size = 32
    # cnt = 0
    # n_valid = 0
    # max_valid = 0
    # Z_train = extract_latent(model_attribute, train_loader)
    # z_mean, z_std = Z_train.mean(0), Z_train.std(0)
    # z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    # for i in range(n_latent_points):
    #     cnt += 1
    #     if cnt == batch_size or i == n_latent_points - 1:
    #         attribute_z = torch.randn(1, 2)
    #         attribute_z = attribute_z * z_std + z_mean
    #         for j in range(decode_times):
    #             g_attribute = model_attribute.decode(attribute_z)
    #             for g0 in g_attribute:
    #                 max_valid += 1
    #                 if is_valid(g0):
    #                     n_valid += 1
    #
    # valid = n_valid / max_valid
    # print(f'Validity: {valid}')
