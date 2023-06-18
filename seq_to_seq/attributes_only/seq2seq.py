import json
import random

import torch
from torch import nn
from graph import MAX_NODE, node_to_ops, attribute_parameters, ATTRIBUTES_POS_COUNT


class Seq2SeqAttributes(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttributes, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, data_len, teacher_forcing_ratio=0.5):
        # source: (batch_size, embedding_dim, node_dim)
        # target: (batch_size, embedding_dim, node_dim)

        target_output_size = self.decoder.output_size
        outputs = torch.zeros(data_len, target_output_size)

        hidden = self.encoder(source)

        input = target[0]  # SOS_token
        # input: (batch_size, node_dim)

        for i in range(1, data_len):
            output, hidden = self.decoder(input, hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = target[i] if random.random() < teacher_forcing_ratio else output

        return outputs

    def create_sequence(self, input):
        operation_id = round(float(input[attribute_parameters['op']['pos']]) * len(node_to_ops))
        sequence = [float(operation_id)]
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
        operation = node_to_ops[op_name]
        for attribute in operation['attributes']:
            if attribute_parameters[attribute]['len'] == 1:
                ids = [attribute_parameters[attribute]['pos']]
                defaults = [attribute_parameters[attribute]['default']]
            else:
                ids = attribute_parameters[attribute]['pos']
                defaults = attribute_parameters[attribute]['default']
            for i in range(len(ids)):
                if input[ids[i]] == -1.:
                    sequence.append(float(defaults[i]))
                else:
                    sequence.append(float(input[ids[i]]))
        return torch.tensor(sequence)

    def get_sequence(self, embedding, operation_id):
        result = [-1.] * ATTRIBUTES_POS_COUNT
        op_name = str(list(filter(lambda x: node_to_ops[x]['id'] == operation_id, node_to_ops))[0])
        operation = node_to_ops[op_name]
        result[attribute_parameters['op']['pos']] = operation_id

        step = 0
        for attribute in operation['attributes']:
            if attribute_parameters[attribute]['len'] == 1:
                ids = [attribute_parameters[attribute]['pos']]
            else:
                ids = attribute_parameters[attribute]['pos']
            for id in ids:
                result[id] = float(embedding[step])
            step += 1

        return torch.tensor(result)

    def encode(self, source):
        # source: (node_embedding_dim)
        SOS_token = torch.tensor([-2.])
        sequence = self.create_sequence(source[:ATTRIBUTES_POS_COUNT])
        operation_id = sequence[0]
        sequence = torch.cat([SOS_token, sequence])
        data_len = len(sequence)
        encoded = self.encoder(sequence)
        return data_len, torch.cat([torch.tensor([operation_id]), encoded.view(-1)])

    def decode(self, embedding, data_len, batch_size=1):
        operation_id = int(embedding[0])
        embedding = embedding[1:]
        SOS_token = torch.tensor([-2.])
        hidden = embedding.view(1, 1, -1)
        input = SOS_token
        target_output_size = self.decoder.output_size
        outputs = torch.zeros(data_len, target_output_size)
        for i in range(1, data_len):
            output, hidden = self.decoder(input, hidden)
            # output: (1, output_size)
            # hidden: (num_layers, batch_size, hidden_size)

            outputs[i] = output
            input = output
        output_vector = self.get_sequence(outputs[1:, 0], operation_id)
        return output_vector