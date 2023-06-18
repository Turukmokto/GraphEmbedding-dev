Fix problem with shapes:
1) Open pytorch_builder.py from HiddenLayer lib
2) Add in method get_shape in else condition ```shape = next(torch_node.outputs()).type().sizes()```