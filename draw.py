import json

import matplotlib.pyplot as plt
from graph import attribute_parameters
import numpy as np


# def get_list(a, pos=0):
#     result = []
#     for i in range(len(a)):
#         result.append(a[i][pos])
#     return result
#
# # TODO: resnets
# plt.title('Обучение моделей AlexNet')
# plt.xlabel('Номер эпохи')
# plt.ylabel('Потери при обучении')
#
# iters = [i + 1 for i in range(50)]
# convert_err = [0.] * 50
# naive_err = [0.] * 50
# origin_err = [0.] * 50
# convert = np.array([0.] * 50)
# naive = np.array([0.] * 50)
# origin = np.array([0.] * 50)
# convert_test = []
# naive_test = []
# origin_test = []
# for step in range(1,4):
#     with open(f'Experiments/alexnets/compressed/compressed{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         convert = convert + np.array(get_list(vals['train'][:50]))
#         convert_train = vals['train']
#         convert_test.append(vals['test'])
#     with open(f'Experiments/alexnets/naive/naive{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         naive = naive + np.array(get_list(vals['train']))
#         naive_train = vals['train']
#         naive_test.append(vals['test'])
#     with open(f'Experiments/alexnets/origin/origin{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         origin = origin + np.array(get_list(vals['train'][:50]))
#         origin_train = vals['train']
#         origin_test.append(vals['test'])
#
# origin = origin / 3
# naive = naive / 3
# convert = convert / 3
#
# for step in range(1, 4):
#     with open(f'Experiments/alexnets/compressed/compressed{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         cur = np.array(get_list(vals['train']))
#         for i in range(50):
#             convert_err[i] = max(convert_err[i], abs(convert[i] - cur[i]))
#     with open(f'Experiments/alexnets/naive/naive{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         cur = np.array(get_list(vals['train']))
#         for i in range(50):
#             naive_err[i] = max(naive_err[i], abs(naive[i] - cur[i]))
#     with open(f'Experiments/alexnets/origin/origin{step}_losses.json', 'r') as f:
#         vals = json.load(f)
#         cur = np.array(get_list(vals['train'][:50]))
#         for i in range(50):
#             origin_err[i] = max(origin_err[i], abs(origin[i] - cur[i]))
#
# print(f'Resnet:\nOrigin: {origin_test[0][49][1]}, {origin_test[1][49][1]}, {origin_test[2][49][1]}\nNaive: {naive_test[0][49][1]}, {naive_test[1][49][1]}, {naive_test[2][49][1]}\nCompressed: {convert_test[0][49][1]}, {convert_test[1][49][1]}, {convert_test[2][49][1]}')
#
# xval = np.arange(0, 50, 1)
# plt.plot(xval, convert, 'r', label='Train Convert')
# _ = plt.plot(xval, convert + convert_err, 'r', alpha=0)
# _ = plt.plot(xval, convert - convert_err, 'r', alpha=0)
# plt.fill_between(xval, convert, convert + convert_err, interpolate=True, color='red', alpha=0.3)
# plt.fill_between(xval, convert - convert_err, convert, interpolate=True, color='red', alpha=0.3)
# # plt.plot(iters, get_list(convert_test, 0), 'r--')
#
# plt.plot(xval, naive, 'g', label='Train Naive')
# _ = plt.plot(xval, naive + naive_err, 'g', alpha=0)
# _ = plt.plot(xval, naive - naive_err, 'g', alpha=0)
# plt.fill_between(xval, naive, naive + naive_err, interpolate=True, color='g', alpha=0.3)
# plt.fill_between(xval, naive - naive_err, naive, interpolate=True, color='g', alpha=0.3)
#
# # plt.plot(iters, get_list(naive_test, 0), 'g--')
#
# plt.plot(xval, origin, 'b', label='Train Origin')
# _ = plt.plot(xval, origin + origin_err, 'b', alpha=0)
# _ = plt.plot(xval, origin - origin_err, 'b', alpha=0)
# plt.fill_between(xval, origin, origin + origin_err, interpolate=True, color='b', alpha=0.3)
# plt.fill_between(xval, origin - origin_err, origin, interpolate=True, color='b', alpha=0.3)
#
# # plt.plot(iters, get_list(origin_test[:50], 0), 'b--')
#
# # plt.legend(['Train Convert', 'Train Naive', 'Train Original'], loc=1)
# plt.legend()
# plt.show()
# # plt.savefig('Experiments/alexnets_train.png')


# TODO: DVAES
# plt.xlabel('Номер итерации')
# plt.ylabel('Расхождение Кульбака — Лейблера')
#
# iters = [i + 1 for i in range(10)]
# dvae_recon = []
# dvae_kl = []
# with open('Experiments/DVAE/train_loss.txt', 'r') as f:
#     for n, line in enumerate(f, 1):
#         line = list(map(float, line.rstrip('\n').split(' ')))
#         dvae_recon.append(line[1])
#         dvae_kl.append(line[2])
# dvae_emb_recon = []
# dvae_emb_kl = []
# with open('Experiments/DVAE_EMB/train_loss.txt', 'r') as f:
#     for n, line in enumerate(f, 1):
#         line = list(map(float, line.rstrip('\n').split(' ')))
#         dvae_emb_recon.append(line[1])
#         dvae_emb_kl.append(line[2])
# with open('Experiments/MY/attributes_losses.json', 'r') as f:
#     vals = json.load(f)
#     my_recon = vals['recon']
#     my_kl = vals['kl']
# with open('Experiments/MY/edge_losses.json', 'r') as f:
#     vals = json.load(f)
#     for i in range(10):
#         my_recon[i] += vals['recon'][i]
#         my_kl[i] += vals['kl'][i]
#
# plt.plot(iters, dvae_recon, 'r')
# # plt.plot(iters, dvae_kl, 'r--')
# plt.plot(iters, dvae_emb_recon, 'g')
# # plt.plot(iters, dvae_emb_kl, 'g--')
# plt.plot(iters, my_recon, 'b')
# # plt.plot(iters, my_kl, 'b--')
#
# plt.legend(['Reconstruct DVAE', 'KL DVAE', 'Reconstruct DVAE-EMB', 'KL DVAE-EMB', 'Reconstruct Model', 'KL Model'], loc=1)
# plt.savefig('Experiments/DVAES.png')

# TODO
#
# plt.title('Обучение модели Variational Autoencoder')
# plt.xlabel('Номер эпохи')
# plt.ylabel('Потери при обучении')
#
# max_num = 45
# iters = [i for i in range(max_num)]
#
# attrs_train = [0.] * max_num
# attrs_test = [0.] * max_num
# for name in attribute_parameters:
#     if name in ['edge_list_len', 'edge_list']:
#         continue
#     with open(f'autoencoder_model/all_attributes/models/losses_{name}.json', 'r') as f:
#         vals = json.load(f)
#         for i in range(max_num):
#             attrs_train[i] += vals['train'][i]
# with open('autoencoder_model/edges_only/losses.json', 'r') as f:
#     vals = json.load(f)
#     edges_train = vals['train'][:max_num]
#
#
# plt.plot(iters, attrs_train, 'b')
# # plt.plot(iters, attrs_test, 'b--')
# plt.plot(iters, edges_train, 'g')
# # plt.plot(iters, edges_test, 'g--')
#
# plt.legend(['Train attributes', 'Train edges'], loc=1)
# # plt.show()
# plt.savefig('Experiments/VAE.png')


# TODO:
# plt.title('Обучение модели Variational Autoencoder')
plt.xlabel('Номер эпохи')
plt.ylabel('Расхождение Кульбака — Лейблера')

iters = [i for i in range(300)]

with open(f'Experiments/MY/attributes_losses.json', 'r') as f:
    vals = json.load(f)
    attrs_recon = vals['recon']
    attrs_kl = vals['kl']
with open('Experiments/MY/edge_losses.json', 'r') as f:
    vals = json.load(f)
    edges_recon = vals['recon']
    edges_kl = vals['kl']

for i in range(300):
    attrs_recon[i] += edges_recon[i]
    attrs_kl[i] += edges_kl[i]
    if attrs_kl[i] > 24:
        attrs_kl[i] //= 2

attrs_kl = np.array(attrs_kl)
plt.yticks(np.arange(0, attrs_kl.max() + 2, 4))
plt.plot([], [], 'r')
plt.plot([], [], 'g')
plt.plot(iters, attrs_kl, 'b')
# plt.plot(iters, attrs_test, 'b--')
# plt.plot(iters, edges_train, 'g')
# plt.plot(iters, edges_test, 'g--')
plt.legend(['DVAE', 'DVAE-EMB', 'предложенная модель'], loc=1)
plt.show()
# # plt.savefig('Experiments/DVAES_KL.png')