import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
from Experiments.alexnets.origin.original_alexnet import AlexNet as OriginalAlexNet
from Experiments.alexnets.naive.converted_alexnet import ConvertedAlexNet as NaiveAlexNet
from Experiments.alexnets.compressed.compressed_alexnet import AlexNet as CompressedAlexNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(dataset):
    training_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        optimizer.step()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


if __name__ == '__main__':
    # train_dataloader, test_dataloader = load_data(datasets.MNIST)
    train_dataloader, test_dataloader = load_data(datasets.CIFAR10)
    models = {
        # 'origin': OriginalAlexNet(num_classes=10).to(device),
        # 'naive': NaiveAlexNet().to(device),
        'compressed': CompressedAlexNet().to(device)
    }
    # model = ConvertedAlexNet()

    # model.load_state_dict(torch.load("models/model.pth"))
    # model.eval()
    # summary(model, input_size=(1, 28, 28))
    # print(model)

    for model_name, model in models.items():
        train_losses = []
        test_losses = []

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        # scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100)

        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t + 1} \n-------------------------------") #  Learning_Rate {scheduler.get_last_lr()}
            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            test_loss = test(test_dataloader, model, loss_fn)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            with open(f'Experiments/alexnets/{model_name}/{model_name}2_losses.json', 'w') as f:
                f.write(json.dumps({'train': train_losses, 'test': test_losses}))

            torch.save(model.state_dict(), f'Experiments/alexnets/{model_name}/{model_name}2_model.pt')
            # scheduler.step()
        print("Done!")

        print("Saved PyTorch Model State to model.pth")

