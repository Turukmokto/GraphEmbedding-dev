class OperationMapping:
    @staticmethod
    def relu_map(node=None, in_shape=None, out_shape=None):
        return 'nn.ReLU()'

    @staticmethod
    def leakyrelu_map(node=None, in_shape=None, out_shape=None):
        return 'nn.LeakyReLU()'

    @staticmethod
    def sigmoid_map(node, in_shape=None, out_shape=None):
        return 'nn.Sigmoid()'

    @staticmethod
    def get_new_pads(node):
        new_pads = None
        isCeil = False
        if node.get('pads'):
            if (node['pads'][0] + node['pads'][2]) % 2 == 1 or (node['pads'][1] + node['pads'][3]) % 2 == 1:
                isCeil = True
            new_pads = [(node['pads'][0] + node['pads'][2]) // 2, (node['pads'][1] + node['pads'][3]) // 2]
        return new_pads, isCeil

    @staticmethod
    def maxpool_map(node, in_shape=None, out_shape=None):
        result = f'nn.MaxPool2d('
        new_pads, isCeil = OperationMapping.get_new_pads(node)
        parameters = {
            'kernel_size': node.get('kernel_shape'),
            'stride': node.get('strides'),
            'padding': new_pads,
            'dilation': node.get('dilations'),
            'ceil_mode': isCeil,
        }
        is_first = True
        for param, value in parameters.items():
            if value:
                if not is_first:
                    result += ', '
                result += f'{param}={value}'
                is_first = False
        return result + ')'

    @staticmethod
    def flatten_map(node=None, in_shape=None, out_shape=None):
        return 'nn.Flatten()'

    @staticmethod
    def adaptive_avgpool_map(node, in_shape=None, out_shape=None):
        return f"nn.AdaptiveAvgPool2d({(node['output_shape'][2], node['output_shape'][3])})"

    @staticmethod
    def avgpool_map(node, in_shape=None, out_shape=None):
        # TODO: Global avgpool https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
        if not node.get('kernel_shape'):
            return OperationMapping.adaptive_avgpool_map(node, in_shape, out_shape)
        new_pads, isCeil = OperationMapping.get_new_pads(node)
        result = f'nn.AvgPool2d('
        parameters = {
            'kernel_size': node.get('kernel_shape'),
            'stride': node.get('strides'),
            'padding': new_pads,
            'ceil_mode': isCeil,
        }
        is_first = True
        for param, value in parameters.items():
            if value:
                if not is_first:
                    result += ', '
                result += f'{param}={value}'
                is_first = False
        return result + ')'

    @staticmethod
    def batchnorm_map(node, in_shape=None, out_shape=None):
        result = f'nn.BatchNorm2d(num_features={in_shape}'
        parameters = {
            'eps': node.get('epsilon'),
            # 'momentum': node.get('momentum'),
        }
        for param, value in parameters.items():
            if value:
                result += f', {param}={value}'
        return result + ')'

    @staticmethod
    def tanh_map(node=None, in_shape=None, out_shape=None):
        return 'nn.Tanh()'

    @staticmethod
    def elu_map(node=None, in_shape=None, out_shape=None):
        return 'nn.ELU()'

    @staticmethod
    def log_softmax_map(node, in_shape=None, out_shape=None):
        return f"nn.LogSoftmax()"

    @staticmethod
    def pad_map(node, in_shape=None, out_shape=None):
        # TODO: F.pad https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def reducemean_map(node, in_shape=None, out_shape=None):
        # TODO: numpy.mean https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def add_map(node, in_shape=None, out_shape=None):
        return f"#  Unsupportable layer type: {node['op']}"

    @staticmethod
    def concat_map(node, in_shape=None, out_shape=None):
        return f"#  Unsupportable layer type: {node['op']}"


class LayerMapping:
    @staticmethod
    def linear_map(node, in_feature, out_feature):
        return f'nn.Linear(in_features={in_feature}, out_features={out_feature})'

    @staticmethod
    def conv_map(node, in_feature, out_feature):
        result = f'nn.Conv2d(in_channels={in_feature}, out_channels={out_feature}'
        new_pads, isCeil = OperationMapping.get_new_pads(node)
        parameters = {
            'kernel_size': tuple(node['kernel_shape']) if node.get('kernel_shape') else None,
            'stride': tuple(node['strides']) if node.get('strides') else None,
            'padding': new_pads,
            'dilation': tuple(node['dilations']) if node.get('dilations') else None,
        }
        for param, value in parameters.items():
            if value:
                result += f', {param}={value}'
        return result + ')'

    @staticmethod
    def transpose_conv_map(node, in_feature, out_feature):
        result = f'nn.ConvTranspose2d(in_channels={in_feature}, out_channels={out_feature}'
        new_pads, _ = OperationMapping.get_new_pads(node)
        parameters = {
            'kernel_size': tuple(node['kernel_shape']) if node.get('kernel_shape') else None,
            'stride': tuple(node['strides']) if node.get('strides') else None,
            'padding': new_pads,
            'dilation': tuple(node['dilations']) if node.get('dilations') else None,
        }
        for param, value in parameters.items():
            if value:
                result += f', {param}={value}'
        return result + ')'


class NetworkMapping:
    # Layers:
    __name_to_layer = {
        "Linear": LayerMapping.linear_map,
        "Conv": LayerMapping.conv_map,
        "ConvTranspose": LayerMapping.transpose_conv_map
    }
    # Operations:
    __name_to_operation = {
        "Relu": OperationMapping.relu_map,
        "MaxPool": OperationMapping.maxpool_map,
        "AdaptiveAveragePool": OperationMapping.adaptive_avgpool_map,
        "AveragePool": OperationMapping.avgpool_map,
        "GlobalAveragePool": OperationMapping.avgpool_map,
        "Flatten": OperationMapping.flatten_map,
        "LeakyRelu": OperationMapping.leakyrelu_map,
        "Sigmoid": OperationMapping.sigmoid_map,
        "BatchNorm": OperationMapping.batchnorm_map,
        "Pad": OperationMapping.pad_map,
        "ReduceMean": OperationMapping.reducemean_map,
        "Tanh": OperationMapping.tanh_map,
        "Elu": OperationMapping.elu_map,
        "LogSoftmax": OperationMapping.log_softmax_map,
    }

    @staticmethod
    def map_node(node, in_shape=None, out_shape=None):
        if node['op'] in NetworkMapping.__name_to_layer:
            return NetworkMapping.__name_to_layer[node['op']](node, in_shape, out_shape)
        if node['op'] in NetworkMapping.__name_to_operation:
            return NetworkMapping.__name_to_operation[node['op']](node, in_shape, out_shape)
        return f"#  Unsupportable layer type: {node['op']}"
