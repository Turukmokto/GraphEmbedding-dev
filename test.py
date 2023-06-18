# Print all attributes for models
import importlib
import json
import os
from autoencoder_model.edges_only.vae import VAE3 as edges_VAE
from autoencoder_model.all_attributes.autoencoder import VAE3 as attributes_VAE

from convert import Converter
from graph import NeuralNetworkGraph
import torch
import torchvision.models as models
from graph import attribute_parameters

# models = {
# 'resnet18': models.resnet18(),
# 'resnet34': models.resnet34(),
# 'resnet50': models.resnet50(),
# 'resnet101': models.resnet101(),
# 'resnet152': models.resnet152(),
# 'resnext50_32x4d': models.resnext50_32x4d(),
# 'resnext101_32x8d': models.resnext101_32x8d(),
# 'wide_resnet50_2': models.wide_resnet50_2(),
# 'wide_resnet101_2': models.wide_resnet101_2(),
# 'unet': UNet(3, 10),
# 'alexnet':models.alexnet(),
# 'vgg16': models.vgg16(),
# 'vgg11': models.vgg11(),
# 'vgg11_bn': models.vgg11_bn(),
# 'vgg13': models.vgg13(),
# 'vgg13_bn': models.vgg13_bn(),
# 'vgg16_bn': models.vgg16_bn(),
# 'vgg19_bn': models.vgg19_bn(),
# 'vgg19': models.vgg19(),
# 'squeezenet1_0': models.squeezenet1_0(),
# 'squeezenet1_1': models.squeezenet1_1(),
# 'densenet161' : models.densenet161(),
# 'densenet121': models.densenet121(),
# 'densenet169': models.densenet169(),
# 'densenet201': models.densenet201(),
# 'inception': models.inception_v3(aux_logits=False),
# 'googlenet': models.googlenet(aux_logits=False),
# 'mnasnet1_0': models.mnasnet1_0(),
# 'mnasnet0_5': models.mnasnet0_5(),
# 'mnasnet0_75': models.mnasnet0_75(),
# 'mnasnet1_3': models.mnasnet1_3(),
# 'gen1':    GeneratedModel1(),
# 'GeneratedDensenet': GeneratedDensenet(),
# 'classification': NaturalSceneClassification(),
# }

torch.set_default_tensor_type(torch.DoubleTensor)
with open(f'./data/embeddings/test.json', 'r') as f:
    test_input = json.load(f)
vals = []
min_vals = []
max_vals = []
NODE_EMBEDDING_DIM = 113
ATTRIBUTES_POS_COUNT = 50
MAX_N = 3000
if os.path.isfile('./data/embeddings/min_max.json'):
    with open(f'./data/embeddings/min_max.json', 'r') as f:
        vals = json.load(f)
    min_vals = vals[0]
    max_vals = vals[1]
    for i in range(len(test_input)):
        for j in range(NODE_EMBEDDING_DIM):
            if j >= ATTRIBUTES_POS_COUNT:
                continue
            if j == attribute_parameters['op']['pos']:
                test_input[i][j] = test_input[i][j] / max_vals[j]
            elif test_input[i][j] == -1 or max_vals[j] == -1:
                test_input[i][j] = 0.
            elif j not in [attribute_parameters['alpha']['pos'], attribute_parameters['epsilon']['pos'],
                           attribute_parameters['momentum']['pos']]:
                test_input[i][j] = (test_input[i][j] + 1.) / (max_vals[j] + 1.)
test_len = len(test_input)
test_input = torch.tensor(test_input)


# Initializing models
model_edges = edges_VAE(
            shapes=[62, 45, 30],
            init_mean=0,
            init_std=1. / 3000
        )
model_edges.load_state_dict(torch.load(f'autoencoder_model/edges_only/vae_model.pt'))
models_attributes = {}
pre_hidden_size = 4  # 4
hidden_size = 2  # 2
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
    models_attributes[name] = attributes_VAE(
        shapes=[attrs['len'], pre_hidden_size, hidden_size],
        init_mean=mean,
        init_std=std
    )
    models_attributes[name].load_state_dict(torch.load(f'autoencoder_model/all_attributes/models/autoencoder_model_{name}.pt'))


# Test: network -> graph -> embedding -> graph -> network
# embed = model_edges.encode(torch.LongTensor([1, 2, 3, 4, 5, 6, 7]), 7)
# out_embed = model_edges.decode(embed, 5)

# data_len, embed = model_attributes.encode([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 11, 11, -1, -1, 0, 4, 64, 55, 55, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
# out_embed = model_attributes.decode(embed, data_len)


xs = torch.zeros([1, 3, 224, 224])
from Experiments.alexnets.origin.original_alexnet import AlexNet as OriginalAlexNet
model = OriginalAlexNet(num_classes=10)
g = NeuralNetworkGraph(model=model, test_batch=xs)

embedding = g.get_embedding(models_attributes, model_edges)
g1 = NeuralNetworkGraph.get_graph(embedding, models_attributes, model_edges)

# print(g.get_embedding(model))

Converter(g1, filepath='Experiments/alexnets/compressed/compressed_alexnet.py', model_name='AlexNet')
from Experiments.alexnets.compressed.compressed_alexnet import AlexNet as CompressedAlexNet
CompressedAlexNet()(xs)
kek = 0
os.remove('tmp_model.py')


# import tmp_model
# # # TODO: fix pad 0
# model = Net11()
# xs = torch.zeros([64, 3, 28, 28])
# g = NeuralNetworkGraph(model=model, test_batch=xs)
# # g1 = NeuralNetworkGraph.get_graph(g.get_naive_embedding())
# Converter(g, filepath='./tmp_model.py', model_name='Tmp')
# importlib.reload(tmp_model)
# tmp_model.Tmp()(xs)
# kek = 0


# # TIMM models
# def create_model_list(filters):
#     model_names = []
#     for filter in filters:
#         model_names.extend(timm.list_models(filter))
#     return model_names
#
# all_models = {}
# with open('./models/original/timm_models.json', 'r') as f:
#     all_models = json.load(f)
# available_models = all_models['available']
#
# model_names = timm.list_models()
#
# with open('./tmp_model.py', 'w') as f:
#     f.write('')
# import tmp_model
# flag = True
# for name in model_names:
#     if (name != 'xcit_large_24_p8_384_dist' and flag) or name.startswith('cait') or name.startswith('beit') or name.startswith('coat') or name.startswith('convit') or name.startswith('deit') or name.startswith('levit') or name.startswith('vit') or name.startswith('tf') or name.startswith('nfnet') or name.startswith('xcit') or name.startswith('twins') or 'bit' in name:
#         if name == 'xcit_large_24_p8_384_dist':
#             flag = False
#         continue
#     print(f"{name} model is processing:")
#     try:
#         if name in available_models:
#             model = timm.create_model(name, output_stride=32)
#         else:
#             model = timm.create_model(name)
#         xs = torch.zeros([1, *model.default_cfg['input_size']])
#         g = NeuralNetworkGraph(model=model, test_batch=xs)
#         embedding = g.get_naive_embedding()
#         Converter(g, filepath='./tmp_model.py', model_name='Tmp')
#         importlib.reload(tmp_model)
#         tmp_model.Tmp()(xs)
#         with zipfile.ZipFile('./data/embeddings/embeddings-zip-timm.zip', 'a') as archive:
#             for e in embedding:
#                 for i in range(len(e)):
#                     if e[i] == None:
#                         e[i] = -1
#             file_name = f'timm--{name}.json'
#             if name in available_models:
#                 file_name = f'timm-generated--{name}.json'
#             archive.writestr(file_name, json.dumps(embedding))
#         print(f"    - ok")
#     except Exception as e:
#         print(f"    - an error occurred: {e}")
#         # failed_models.append(name)
#         # with open('./models/original/foz_models.json', 'w') as f:
#         #     f.write(json.dumps({'available': available_models, 'failed': failed_models}))
#     print('--------------------\n')
# os.remove('./tmp_model.py')






# pool_list = ['avgmax', 'max']
# blocks = [BasicBlock, Bottleneck]
# with open('./tmp_model.py', 'w') as f:
#     f.write('')
# import tmp_model
# cnt = 0
#
# for l_1 in [4, 8, 16, 32, 64]:
#     for l_2 in [4, 8, 16, 32, 64]:
#         for l_3 in [4, 8, 16, 32, 64]:
#             for l_4 in [4, 8, 16, 32, 64]:
#                 for pool in pool_list:
#                     for block in blocks:
#                         if pool == 'avgmax' and isinstance(block, BasicBlock):
#                             continue
#                         try:
#                             model = ResNet(block, layers=[l_1, l_2, l_3, l_4], in_chans=3, global_pool=pool)
#                             xs = torch.zeros([1, 3, 224, 224])
#                             g = NeuralNetworkGraph(model=model, test_batch=xs)
#                             embedding = g.get_naive_embedding()
#                             Converter(g, filepath='./tmp_model.py', model_name='Tmp')
#                             importlib.reload(tmp_model)
#                             tmp_model.Tmp()(xs)
#                             cnt += 1
#                             if cnt == 3:
#                                 cnt = 400
#                             with zipfile.ZipFile('./data/embeddings/embeddings-zip-gen-timm.zip', 'a') as archive:
#                                 for e in embedding:
#                                     for i in range(len(e)):
#                                         if e[i] == None:
#                                             e[i] = -1
#                                 archive.writestr(f'generated-timm--resnet_{cnt}.json', json.dumps(embedding))
#                         except Exception as e:
#                             pass
# os.remove('./tmp_model.py')




# models = {}
# for name in model_names:
#     models[name] = timm.create_model(name)

# # Test: network -> graph -> network
# with open('./tmp_model.py', 'w') as f:
#     f.write('')
# import tmp_model
# for name, model in models.items():
#     print(f'{name} model is processing')
#     xs = torch.zeros([4, 3, 224, 224])
#     if name == 'inception':
#         xs = torch.zeros([4, 3, 299, 299])
#     if name == 'classification':
#         xs = torch.zeros([128, 3, 150, 150])
#     g = NeuralNetworkGraph(model=model, test_batch=xs)
#     embedding = g.get_naive_embedding()
#     Converter(g, filepath='./tmp_model.py', model_name='Tmp')
#     importlib.reload(tmp_model)
#     tmp_model.Tmp()(xs)
# os.remove('./tmp_model.py')


# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=False,
#     transform=ToTensor(),
# )
# xs = torch.zeros([64, 3, 224, 224])
# models = {
#     'My Network': NeuralNetwork(),
#     # 'googlenet': googlenet(),
#     # 'inception_v3': inception_v3(),
#     # 'mobilenet_v3_large': mobilenet_v3_large(),
#
#     'ResNet101': resnet101(),
#     'Alexnet': alexnet(),
#     'densenet201': densenet201(),
#     'mnasnet1_3': mnasnet1_3(),
#     'squeezenet1_1': squeezenet1_1(),
#     'vgg19_bn': vgg19_bn()
# }
# dic_attrs = dict()
#
# with open('layers.txt', 'w') as f:
#     for model in models:
#         print(model)
#         if model == 'My Network':
#             xs = torch.zeros([64, 1, 28, 28])
#         else:
#             xs = torch.zeros([64, 3, 224, 224])
#         g = NeuralNetworkGraph(model=models[model], test_batch=xs)
#         f.write(model + '\n')
#         for node in g.nodes:
#             attrs = g.nodes[node]
#             op = attrs['op']
#             del attrs['op']
#             f.write(op + ': ' + str(attrs) + '\n')
#             if not op in dic_attrs:
#                 dic_attrs[op] = dict()
#             for attr in attrs:
#                 if not attr in dic_attrs[op]:
#                     dic_attrs[op][attr] = attrs[attr]
#         f.write('\n')
#     f.write('\n----------------------------------------------------------------------\n')
#     for dic_attr in dic_attrs:
#         f.write(dic_attr + ': ' + str(dic_attrs[dic_attr]) + '\n')
#     f.write('\n----------------------------------------------------------------------\n')
#     map_counter = 0
#     map_attrs = dict()
#     for dic_attr in dic_attrs:
#         for attr in dic_attrs[dic_attr]:
#             new_attr = attr
#             if attr == 'output_shape' and dic_attrs[dic_attr][attr]:
#                 new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
#             if attr == 'pads' and dic_attrs[dic_attr][attr]:
#                 new_attr += "_" + str(len(dic_attrs[dic_attr][attr]))
#             if not new_attr in map_attrs:
#                 if isinstance(dic_attrs[dic_attr][attr], list):
#                     map_pos = []
#                     for i in range(0, len(dic_attrs[dic_attr][attr])):
#                         map_pos.append(map_counter)
#                         map_counter += 1
#                     f.write(new_attr + ': ' + str(map_pos) + '\n')
#                     map_attrs[new_attr] = map_pos
#                 else:
#                     f.write(attr + ': ' + str(map_counter) + '\n')
#                     map_attrs[new_attr] = map_counter
#                     map_counter += 1
#             elif isinstance(dic_attrs[dic_attr][attr], list) and len(dic_attrs[dic_attr][attr]) != len(map_attrs[new_attr]):
#                 print('Error -- ' + new_attr + ': ' + str(dic_attrs[dic_attr][attr]) + ', ' + str(map_attrs[new_attr]))
