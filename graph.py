import sys

import networkx as nx
from torchvision.models import resnet101, densenet201, vgg19_bn, mnasnet1_3, squeezenet1_1, resnet50, \
    inception_v3

from models.original.original_alexnet import AlexNet
from models.original.original_unet import UNet
import hiddenlayer as hl
import torch
import pandas
import json
from functools import reduce

ATTRIBUTES_POS_COUNT = 50
NODE_EMBEDDING_DIMENSION = 113
NONE_REPLACEMENT = -1
MAX_NODE = 3_000  # for 200 layers in network
ATTRIBUTES_HIDDEN_SIZE = 2
MAX_ATTRIBUTES = 7

node_to_ops = {
    "Conv": {'id': 0, 'attributes': ['op', 'output_shape', 'dilations', 'group', 'kernel_shape', 'pads', 'strides']},
    "LeakyRelu": {'id': 1, 'attributes': ['op', 'output_shape']},
    "MaxPool": {'id': 2, 'attributes': ['op', 'output_shape', 'dilations', 'kernel_shape', 'pads', 'strides']},
    "Flatten": {'id': 3, 'attributes': ['op', 'output_shape']},
    "Linear": {'id': 4, 'attributes': ['op', 'output_shape']},
    "Sigmoid": {'id': 5, 'attributes': ['op', 'output_shape']},
    "BatchNorm": {'id': 6, 'attributes': ['op', 'output_shape', 'epsilon', 'momentum']},
    "Relu": {'id': 7, 'attributes': ['op', 'output_shape']},
    "Add": {'id': 8, 'attributes': ['op', 'output_shape']},
    "GlobalAveragePool": {'id': 9, 'attributes': ['op', 'output_shape']},
    "AveragePool": {'id': 10, 'attributes': ['op', 'output_shape', 'kernel_shape', 'pads', 'strides']},
    "Concat": {'id': 11, 'attributes': ['op', 'output_shape']},
    "Pad": {'id': 12, 'attributes': ['op', 'output_shape', 'pads']},
    "ReduceMean": {'id': 13, 'attributes': ['op', 'output_shape', 'axes', 'keepdims']},
    "Tanh": {'id': 14, 'attributes': ['op', 'output_shape']},
    "ConvTranspose": {'id': 15,
                      'attributes': ['op', 'output_shape', 'dilations', 'group', 'kernel_shape', 'pads', 'strides']},
    "Slice": {'id': 16, 'attributes': ['op', 'output_shape', 'axes', 'starts', 'ends', 'steps']},
    "Elu": {'id': 17, 'attributes': ['op', 'output_shape']},
    "Constant": {'id': 18, 'attributes': ['op', 'output_shape', 'value']},
    "Reshape": {'id': 19, 'attributes': ['op', 'output_shape']},
    "Mul": {'id': 20, 'attributes': ['op', 'output_shape']},
    "Transpose": {'id': 21, 'attributes': ['op', 'output_shape', 'perm']},
    "LogSoftmax": {'id': 22, 'attributes': ['op', 'output_shape']},
}

pads_to_mods = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

attribute_parameters = {
    "alpha": {'len': 1, 'pos': 0, 'type': 'float', 'range': [-1., 1.], 'default': 1.0, 'lr': 1e-3},
    "axes": {'len': 4, 'pos': [1, 2, 3, 4], 'type': 'int', 'range': [-1, 3], 'default': [-1, -1, -1, -1], 'lr': 1e-3},
    "axis": {'len': 1, 'pos': 5, 'type': 'int', 'range': [-1, 3], 'default': 0, 'lr': 1e-3},
    "dilations": {'len': 2, 'pos': [6, 7], 'type': 'int', 'range': [-1, 2], 'default': [1, 1], 'lr': 1e-3},
    "ends": {'len': 4, 'pos': [8, 9, 10, 11], 'type': 'int', 'range': [-1, 30], 'default': [-1, -1, -1, -1],
             'lr': 1e-3},
    "epsilon": {'len': 1, 'pos': 12, 'type': 'float', 'range': [-1., 1.], 'default': 1e-5, 'lr': 1e-3},
    "group": {'len': 1, 'pos': 13, 'type': 'int', 'range': [-1, 1536], 'default': 1, 'lr': 1e-3},
    "keepdims": {'len': 1, 'pos': 14, 'type': 'int', 'range': [-1, 1], 'default': 1, 'lr': 1e-3},
    "kernel_shape": {'len': 2, 'pos': [15, 16], 'type': 'int', 'range': [-1, 11], 'default': [1, 1], 'lr': 1e-3},
    "mode": {'len': 1, 'pos': 17, 'type': 'int', 'range': [-1, len(pads_to_mods)], 'default': 0, 'lr': 1e-3},
    "momentum": {'len': 1, 'pos': 18, 'type': 'float', 'range': [-1., 1.], 'default': 0.9, 'lr': 1e-3},
    "op": {'len': 1, 'pos': 19, 'type': 'int', 'range': [0, len(node_to_ops)], 'default': 0, 'lr': 1e-4},
    "output_shape": {'len': 4, 'pos': [20, 21, 22, 23], 'type': 'int', 'range': [-1, 802816],
                     'default': [-1, -1, -1, -1], 'lr': 1e-5},
    "pads": {'len': 8, 'pos': [24, 25, 26, 27, 28, 29, 30, 31], 'type': 'int', 'range': [-1, 2],
             'default': [0, 0, 0, 0, 0, 0, 0, 0], 'lr': 1e-3},
    "starts": {'len': 4, 'pos': [32, 33, 34, 35], 'type': 'int', 'range': [-1, 30], 'default': [-1, -1, -1, -1],
               'lr': 1e-3},
    "steps": {'len': 4, 'pos': [36, 37, 38, 39], 'type': 'int', 'range': [-1, 30], 'default': [-1, -1, -1, -1],
              'lr': 1e-3},
    "strides": {'len': 2, 'pos': [40, 41], 'type': 'int', 'range': [-1, 7], 'default': [1, 1], 'lr': 1e-3},
    "value": {'len': 4, 'pos': [42, 43, 44, 45], 'type': 'int', 'range': [-1, 10], 'default': [0, 0, 0, 0], 'lr': 1e-3},
    "perm": {'len': 4, 'pos': [46, 47, 48, 49], 'type': 'int', 'range': [0, 4], 'default': [-1, -1, -1, -1],
             'lr': 1e-3},
    "edge_list_len": {'len': 1, 'pos': 50, 'type': 'int',
                      'range': [0, NODE_EMBEDDING_DIMENSION - ATTRIBUTES_POS_COUNT - 1], 'default': 1},
    "edge_list": {'type': 'int', 'range': [0, MAX_NODE - 1]},
    # "skip_connections": [50, ...]
}

# TODO: sys.maxsize --> None / -1

reversed_attributes = {
    0: 'alpha',
    1: 'axes',
    5: 'axis',
    6: 'dilations',
    8: 'ends',
    12: 'epsilon',
    13: 'group',
    14: 'keepdims',
    15: 'kernel_shape',
    17: 'mode',
    18: 'momentum',
    19: 'op',
    20: 'output_shape',
    24: 'pads',
    32: 'starts',
    36: 'steps',
    40: 'strides',
    42: 'value',
    46: 'perm',
    50: 'edge_list_len',
    51: 'edge_list'
}


# autoencoder_model = Autoencoder()
# autoencoder_model.load_state_dict(torch.load("models/autoencoder_model.pth"))


class NeuralNetworkGraph(nx.DiGraph):
    """Parse graph from network"""

    def __init__(self, model, test_batch):
        """Initialize structure with embedding for each node from `model` and graph from `HiddenLayer`"""
        super().__init__()
        hl_graph = hl.build_graph(model, test_batch, transforms=None)
        self.__colors = {}
        self.__input_shapes = {}
        self.__id_to_node = {}
        self.embedding = []
        self.__parse_graph(hl_graph)

    @staticmethod
    def denormalize_vector(embedding, is_op_norm=True):
        with open(f'./data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if j == ATTRIBUTES_POS_COUNT or j == attribute_parameters['op']['pos'] or j in attribute_parameters['output_shape']['pos']:
                    continue
                if i == attribute_parameters['op']['pos']:
                    embedding[i][j] = embedding[i][j] * max_vals[j]
                elif embedding[i][j] < 1e-9 or max_vals[j] == -1:
                    embedding[i][j] = 0.
                elif j > ATTRIBUTES_POS_COUNT:
                    embedding[i][j] = embedding[i][j] * max_vals[j]
                elif j not in [attribute_parameters['alpha']['pos'], attribute_parameters['epsilon']['pos'],
                               attribute_parameters['momentum']['pos']]:
                    embedding[i][j] = embedding[i][j] * (max_vals[j] + 1.) - 1.
        return embedding

    @staticmethod
    def normalize_vector(embedding):
        with open(f'./data/embeddings/min_max.json', 'r') as f:
            vals = json.load(f)
        min_vals = vals[0]
        max_vals = vals[1]
        for i in range(len(embedding)):
            if i == ATTRIBUTES_POS_COUNT:
                continue
            if i == attribute_parameters['op']['pos']:
                embedding[i] = embedding[i] / max_vals[i]
            elif embedding[i] == -1 or max_vals[i] == -1:
                embedding[i] = 0.
            elif i > ATTRIBUTES_POS_COUNT:
                embedding[i] = embedding[i] / max_vals[i]
            elif i not in [attribute_parameters['alpha']['pos'], attribute_parameters['epsilon']['pos'],
                           attribute_parameters['momentum']['pos']]:
                embedding[i] = (embedding[i] + 1.) / (max_vals[i] + 1.)
        return torch.tensor(embedding)

    @classmethod
    def get_graph(cls, embedding, models_attributes, model_edges, is_naive=False, is_normalize_needed=True):
        """Create graph from embedding and return it. Get embedding type of list"""
        graph = cls.__new__(cls)
        super(NeuralNetworkGraph, graph).__init__()

        if is_naive:
            decoded = embedding
        else:
            decoded = []
            for row in embedding:
                attribute_op = int(row[0])
                out_shapes = row[1:5]
                attribute_embedding = torch.tensor(row[5:(MAX_ATTRIBUTES * ATTRIBUTES_HIDDEN_SIZE + 5)]).view(1, -1)
                edges_len = int(row[MAX_ATTRIBUTES * ATTRIBUTES_HIDDEN_SIZE + 5])
                edge_embedding = torch.tensor(row[(MAX_ATTRIBUTES * ATTRIBUTES_HIDDEN_SIZE + 6):]).view(1, -1)

                result_vector = [-1] * NODE_EMBEDDING_DIMENSION

                op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == attribute_op, node_to_ops))[0])
                cnt = 0
                for attribute in node_to_ops[op_name]['attributes']:
                    attribute_output = models_attributes[attribute].decode(
                        attribute_embedding[0][(cnt * ATTRIBUTES_HIDDEN_SIZE):((cnt + 1) * ATTRIBUTES_HIDDEN_SIZE)]
                    ).view(-1).tolist()
                    cnt += 1
                    if attribute_parameters[attribute]['len'] == 1:
                        ids = [attribute_parameters[attribute]['pos']]
                    else:
                        ids = attribute_parameters[attribute]['pos']
                    for i in range(len(ids)):
                        if attribute == 'output_shape':
                            result_vector[ids[i]] = out_shapes[i]
                        else:
                            result_vector[ids[i]] = attribute_output[i]
                edge_output = model_edges.decode(edge_embedding).view(-1).tolist()

                for i in range(len(edge_output)):
                    result_vector[ATTRIBUTES_POS_COUNT + 1 + i] = edge_output[i]

                result_vector[attribute_parameters['op']['pos']] = attribute_op
                result_vector[ATTRIBUTES_POS_COUNT] = edges_len
                decoded.append(result_vector)

        denormalized = decoded if not is_normalize_needed else cls.denormalize_vector(decoded)
        valid_naive = NeuralNetworkGraph.replace_none_in_embedding(denormalized, is_need_replace=False)
        graph.embedding = cls.__fix_attributes(valid_naive)

        graph.__create_graph()
        return graph

    @staticmethod
    def __fix_attributes(embedding):
        for e in range(len(embedding)):
            for pos, name in reversed_attributes.items():
                attr = attribute_parameters[name]
                if 'len' not in attr:
                    n = NODE_EMBEDDING_DIMENSION - pos
                else:
                    n = attr['len']
                for i in range(n):
                    if embedding[e][pos + i] is None:
                        continue
                    if attr['type'] == 'int':
                        embedding[e][pos + i] = int(round(embedding[e][pos + i]))
                    if attr['type'] == 'float':
                        embedding[e][pos + i] = float(embedding[e][pos + i])
                    if embedding[e][pos + i] < attr['range'][0]:
                        embedding[e][pos + i] = attr['range'][0]
                    if attr['range'][1] < embedding[e][pos + i]:
                        embedding[e][pos + i] = attr['range'][1]
                if name == 'kernel_shape' and embedding[e][pos] is not None:
                    if embedding[e][pos] % 2 != 1:
                        embedding[e][pos] = max(1, embedding[e][pos] - 1)
                    embedding[e][pos + 1] = embedding[e][pos]
                if name in ['strides', 'dilations'] and embedding[e][pos] is not None:
                    embedding[e][pos + 1] = embedding[e][pos]


        return embedding

    def get_naive_embedding(self):
        """Return naive embedding"""
        return self.__fix_attributes(self.embedding)

    @staticmethod
    def create_sequence(inputs, operation_id, models):
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
        operation = node_to_ops[op_name]
        result = []
        for attribute in operation['attributes']:
            sequence = []
            if attribute_parameters[attribute]['len'] == 1:
                ids = [attribute_parameters[attribute]['pos']]
            else:
                ids = attribute_parameters[attribute]['pos']
            for i in range(len(ids)):
                sequence.append(inputs[ids[i]])
            sequence = NeuralNetworkGraph.normalize_vector(sequence)
            embed = models[attribute].latent(sequence, False)
            result.extend(embed.tolist())
        if len(result) < ATTRIBUTES_HIDDEN_SIZE * MAX_ATTRIBUTES:
            for i in range(ATTRIBUTES_HIDDEN_SIZE * MAX_ATTRIBUTES - len(result)):
                result.append(0.)
        return result

    @staticmethod
    def replace_none_in_embedding(embedding, is_need_replace=True):
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if is_need_replace and embedding[i][j] is None:
                    embedding[i][j] = NONE_REPLACEMENT
                if not is_need_replace and round(embedding[i][j]) == NONE_REPLACEMENT:
                    embedding[i][j] = None
        return embedding

    def get_embedding(self, models_attributes, model_edges):
        """Return embedding"""
        input = self.__fix_attributes(self.embedding)
        input = self.replace_none_in_embedding(input)
        result = []
        for row in input:
            attribute_op = row[attribute_parameters['op']['pos']]
            out_shapes = row[attribute_parameters['output_shape']['pos'][0]:(attribute_parameters['output_shape']['pos'][3] + 1)]
            edges_len = row[ATTRIBUTES_POS_COUNT]
            normalized_row = self.normalize_vector(row)
            edge_row = torch.tensor(normalized_row[(ATTRIBUTES_POS_COUNT + 1):]).view(1, -1)
            edge_embedding = model_edges.latent(edge_row, False)

            attribute_emmbedding = self.create_sequence(normalized_row, attribute_op, models_attributes)

            result.append([attribute_op, *out_shapes, *attribute_emmbedding, edges_len, *edge_embedding[0].tolist()])
        return result

    def __create_graph(self):
        """Create `networkx.DiGraph` graph from embedding"""
        counter = 0
        for embedding in self.embedding:
            """Add node with attributes to graph"""
            params = {}
            operation_id = embedding[attribute_parameters['op']['pos']]
            op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
            operation = node_to_ops[op_name]
            params['op'] = op_name
            for attribute in operation['attributes']:
                if attribute_parameters[attribute]['len'] == 1:
                    ids = [attribute_parameters[attribute]['pos']]
                    defaults = [attribute_parameters[attribute]['default']]
                else:
                    ids = attribute_parameters[attribute]['pos']
                    defaults = attribute_parameters[attribute]['default']
                new_params = []
                for i in range(len(ids)):
                    to_append = embedding[ids[i]]
                    if embedding[ids[i]] is None:
                        to_append = defaults[i]
                    if attribute == 'op':
                        to_append = str(
                            list(filter(lambda x: node_to_ops[x]['id'] == embedding[ids[i]], node_to_ops))[0])
                    elif attribute == 'mode':
                        to_append = str(list(filter(lambda x: pads_to_mods[x] == embedding[ids[i]], pads_to_mods))[0])
                    new_params.append(to_append)
                params[attribute] = new_params[0] if attribute_parameters[attribute]['len'] == 1 else new_params
            self.add_node(counter, **params)

            """Add edge to graph"""
            for i in range(embedding[ATTRIBUTES_POS_COUNT]):
                if embedding[ATTRIBUTES_POS_COUNT + i + 1] > len(self.embedding):
                    self.add_edge(counter, counter + 1)
                else:
                    self.add_edge(counter, embedding[ATTRIBUTES_POS_COUNT + i + 1])
            counter += 1

    def __add_edges(self, graph):
        """Add edges with changed node's names"""
        for edge in graph.edges:
            v = self.__id_to_node.get(edge[0])
            u = self.__id_to_node.get(edge[1])
            self.__input_shapes[u] = edge[2]
            if v == u:
                continue
            self.add_edge(v, u)

    def __is_supported(self, v):
        """Check if graph is supported"""
        self.__colors[v] = 1
        result = True
        for u in self.adj[v]:
            if self.__colors.get(u, 0) == 0:
                result &= self.__is_supported(u)
            elif self.__colors.get(u, 0) == 1:
                result = False
        self.__colors[v] = 2
        return result

    def __calculate_embedding(self):
        """Calculate embedding for each node"""
        for id in self.nodes:
            node = self.nodes[id]
            embedding = [None] * NODE_EMBEDDING_DIMENSION

            """
            Take output_shape and check it. output_shape might be None or
            size 2 (for linear), size 4 (for convolutional).
            """
            if not node['output_shape'] or node['output_shape'] == []:
                output_shape = self.__input_shapes.get(id)
                if output_shape:
                    node['output_shape'] = output_shape
                    self.nodes[id]['output_shape'] = output_shape
                else:
                    del node['output_shape']

            """
            Set node's parameters to embedding vector in order described in attribute_to_pos dictionary 
            and map string parameters to its' numeric representation.
            """
            for param in node:
                op_name = param
                if isinstance(node[param], list):
                    current_poses = attribute_parameters[op_name]['pos']
                    for i in range(len(node[param])):
                        embedding[current_poses[i]] = node[param][i]
                else:
                    value = node[param]
                    if param == 'op':
                        value = node_to_ops[value]['id']
                    if param == 'mode' and node['op'] == 'Pad':
                        value = pads_to_mods[value]
                    if op_name in attribute_parameters:
                        cur_pos = attribute_parameters[op_name]['pos'][0] if attribute_parameters[op_name][
                                                                                 'len'] > 1 else \
                        attribute_parameters[op_name]['pos']
                        if value == sys.maxsize:
                            embedding[cur_pos] = None
                        else:
                            embedding[cur_pos] = value

            edge_list = list(self.adj[id])
            # embedding.extend(edge_list)
            if len(edge_list) + ATTRIBUTES_POS_COUNT + 1 <= NODE_EMBEDDING_DIMENSION:
                embedding[ATTRIBUTES_POS_COUNT] = len(edge_list)
                for i in range(0, len(edge_list)):
                    embedding[ATTRIBUTES_POS_COUNT + i + 1] = edge_list[i]
            else:
                print('This graph is not supported!')
            self.embedding.append(embedding)

    def __parse_graph(self, graph):
        """Parse `HiddenLayer` graph and create `networkx.DiGraph` with same node attributes"""
        try:
            counter = 0

            """Renumber nodes and add it to graph"""
            values = {}
            for id in graph.nodes:
                graph.nodes[id].params['output_shape'] = graph.nodes[id].output_shape
                graph.nodes[id].params['op'] = graph.nodes[id].op
                if graph.nodes[id].params['op'] == 'Constant':
                    to = list(filter(lambda x: x[0] == id, graph.edges))[0][1]
                    if torch.is_tensor(graph.nodes[id].params['value']):
                        values[to] = {'value': graph.nodes[id].params['value'].tolist(), 'from': id}
                    else:
                        values[to] = {'value': graph.nodes[id].params['value'], 'from': id}
                    continue
                self.__id_to_node[id] = counter
                counter += 1

            for id in graph.nodes:
                if graph.nodes[id].params['op'] == 'Constant':
                    continue
                if id in values:
                    graph.nodes[id].params['value'] = values[id]['value']
                    self.__id_to_node[values[id]['from']] = self.__id_to_node[id]
                self.add_node(self.__id_to_node[id], **graph.nodes[id].params)

            self.__add_edges(graph)
            is_supported = self.__is_supported(0)

            if is_supported:
                self.__calculate_embedding()
            else:
                print('Graph is not supported. This network is not supported.')
        except KeyError as e:
            print(f'Operation or layer is not supported: {e}.')
            raise KeyError(f'Operation or layer is not supported: {e}.')

    @staticmethod
    def check_equality(graph1, graph2):
        """Check two graphs on equality. Return if they are equal and message"""
        if graph1.edges != graph2.edges:
            return False, 'Edges are not equal'
        if sorted(list(graph1.nodes)) != sorted(list(graph2.nodes)):
            return False, 'Nodes are not equal'
        for node in graph1.nodes:
            if graph1.nodes[node] != graph2.nodes[node]:
                return False, 'Node params are not equal'
        return True, 'Graphs are equal'


if __name__ == '__main__':
    # model = NeuralNetwork()
    # model = resnet101()
    # model = AlexNet()
    # model = densenet201()
    # model = mnasnet1_3()
    # model = squeezenet1_1()
    # model = resnet50()
    # model = vgg19_bn()
    # model = inception_v3(aux_logits=False)
    model = UNet(3, 10)

    xs = torch.zeros([1, 3, 224, 224])  # for other models from torchvision.models
    # xs = torch.zeros([64, 3, 28, 28])  # for MnasNet and NeuralNetwork
    # xs = torch.zeros([64, 3, 299, 299])  # for inception

    # g1 = NeuralNetworkGraph(model=model, test_batch=xs)
    # g2 = NeuralNetworkGraph.get_graph(g1.get_embedding())
    # is_equal, message = NeuralNetworkGraph.check_equality(g1, g2)
    # print(message)

    models = {
        "alexnet": AlexNet(),
        "resnet50": resnet50(),
        "resnet101": resnet101(),
        "unet": UNet(3, 10),
        "vgg": vgg19_bn(),
        "densenet": densenet201(),
        "inception": inception_v3(aux_logits=False),
        "mnasnet": mnasnet1_3(),
        "squeezenet": squeezenet1_1(),
    }

    # cnt = 0
    # for name, model in models.items():
    #     cnt += 1
    #     xs = torch.zeros([1, 3, 224, 224])
    #     if name == 'mnasnet':
    #         xs = torch.zeros([64, 3, 28, 28])
    #     if name == 'inception':
    #         xs = torch.zeros([64, 3, 299, 299])
    #     g = NeuralNetworkGraph(model=model, test_batch=xs)
    #     embedding = g.get_naive_embedding()
    #     for e in embedding:
    #         for i in range(len(e)):
    #             if e[i] == None:
    #                 e[i] = NONE_REPLACEMENT
    #     with open(f'./data/embeddings/{cnt}.json', 'w') as f:
    #         f.write(json.dumps(embedding))

    # with open('embeddings/naive/embeddings_dims.txt', 'w') as f:
    #     for name, model in models.items():
    #         xs = torch.zeros([1, 3, 224, 224])
    #         if name == 'mnasnet':
    #             xs = torch.zeros([64, 3, 28, 28])
    #         if name == 'inception':
    #             xs = torch.zeros([64, 3, 299, 299])
    #         g = NeuralNetworkGraph(model=model, test_batch=xs)
    #         dim = NODE_EMBEDDING_DIMENSION
    #         f.write(f'{name}:\nlen = {len(g.embedding)}\nnode_dim = {dim}\n\n')
    #
    #         with open(f'embeddings/naive/naive_{name}.txt', 'w') as f1:
    #             f1.write(json.dumps(g.embedding))
