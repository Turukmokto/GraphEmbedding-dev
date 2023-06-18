import json
from torch.utils.data import Dataset
import torch
import os
import utils
import zipfile
import random

SEED = 1234
random.seed(SEED)

class EmbeddingDataset(Dataset):
    def __init__(self, root, train=True, transform=None, normalize=True):
        self.train_data = []
        self.test_data = []
        self.train = train
        self.transform = transform
        archive_path = os.path.join(root, 'embeddings-zip.zip')
        archive = zipfile.ZipFile(archive_path, 'r')
        for name in archive.namelist():
            with archive.open(name, 'r') as file:
                embedding = json.load(file)
                rnd = random.random()
                # if self.train and rnd < 0.7:
                self.train_data.append(embedding)
                if not self.train and rnd > 0.7:
                    self.test_data.append(embedding)
        if normalize:
            if self.train:
                self.train_data = utils.normalize_dataset(self.train_data)
            else:
                self.test_data = utils.normalize_dataset(self.test_data)
        archive.close()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            embedding = self.train_data[idx]
        else:
            embedding = self.test_data[idx]
        if self.transform:
            embedding = self.transform(embedding)
        return embedding
