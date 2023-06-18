import torch
import onnx
from KD_Lib.models import ResNet18, LeNet, LSTMNet
from torchvision.models import resnet101, alexnet, densenet201

from graph import NeuralNetworkGraph
from network import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets, models
import hiddenlayer as hl


# model = NeuralNetwork()
model = alexnet()
# model = densenet201()
# model = mnasnet1_3()
# model = squeezenet1_1()
# model = vgg19_bn()
# model = LeNet(in_channels=1, img_size=28)
# model = resnet101()
# xs = torch.zeros([1, 1, 28, 28])
xs = torch.zeros([1, 3, 224, 224])

hl_graph = hl.build_graph(model, xs, transforms=None)
hl_graph.theme = hl.graph.THEMES["blue"].copy()
hl_graph.save('LeNet', format='png')

# onnx.export: model -> graph
# torch.onnx.export(model, xs, "test.onnx", verbose=True)
# model = onnx.load("test.onnx")
# print(onnx.helper.printable_graph(model.graph))

