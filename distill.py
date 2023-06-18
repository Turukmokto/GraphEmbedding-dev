import KD_Lib
import torch
import torch.optim as optim
import torchvision
from KD_Lib.models import ResNet50, ResNet18, LeNet, ModLeNet
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, SelfTraining
from network import NeuralNetwork

# This part is where you define your datasets, dataloaders, models and optimizers

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

is_self_train = False

if is_self_train:
    teacher_model = NeuralNetwork()
    student_model = NeuralNetwork()

    teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    student_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
                          teacher_optimizer, student_optimizer, log=True)
    distiller.train_teacher(epochs=5, plot_losses=True, save_model=False)
    distiller.evaluate(teacher=True)
    distiller.get_parameters()
else:
    teacher_model = ModLeNet(in_channels=1, img_size=28)
    teacher_model.load_state_dict(torch.load("models/teacher.pt"))
    teacher_model.eval()

    # print(ResNet18([4, 4, 4, 4, 4]))
    # exit(0)

    student_model = NeuralNetwork()

    teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    student_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
                          teacher_optimizer, student_optimizer, log=True)
    # distiller.train_teacher(epochs=30, plot_losses=True, save_model=False)
    distiller.train_student(epochs=5, plot_losses=True, save_model=False)
    distiller.evaluate(teacher=False)
    distiller.get_parameters()
