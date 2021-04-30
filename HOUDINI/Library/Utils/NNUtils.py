import math
import json
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Callable, Optional, Dict, Tuple, Union, List

from HOUDINI.Library.NN import NetMLP, NetDO, SaveableNNModule


def get_nn_from_params_dict(uf: Dict) -> Tuple[nn.Module, List]:
    """Instantiate the unkown function (uf) required by the high-order functions with
    a neural network

    Args:
        uf: the dict of unknown function storing the parameters for the nn candidate

    Returns:
        the NN class
        trainable parameters
    """

    new_nn = None
    if uf['type'] == 'MLP':
        new_nn = NetMLP(uf['name'],
                        uf['input_dim'],
                        uf['output_dim'],
                        uf['bias'])
    elif uf['type'] == 'DO':
        new_nn = NetDO(uf['name'],
                       uf['input_dim'],
                       uf['dt_name'])
    else:
        raise NotImplementedError()

    if 'initialize_from' in uf and uf['initialize_from'] is not None:
        new_nn.load(uf['initialize_from'])

    if torch.cuda.is_available():
        new_nn.cuda()

    new_nn.params_dict = uf
    c_trainable_parameters = list(new_nn.parameters())

    return new_nn, c_trainable_parameters


def create_and_load(directory: str,
                    name: str,
                    new_name: str = None) -> nn.Module:
    """Instantiate an unkown function (uf) required by the high-order functions with
    a trained neural network

    Args:
        directory: directory to the saved weights of an NN
        name: name of the unknown function
        new_name: the new name of the unknown function

    Returns:
        the NN class with trained weights
    """

    if new_name is None:
        new_name = name

    with open('{}/{}.json'.format(directory, name)) as json_data:
        params_dict = json.load(json_data)
        params_dict['name'] = new_name

        if params_dict['output_activation'] == 'None':
            params_dict['output_activation'] = None
        elif params_dict['output_activation'] == 'sigmoid':
            params_dict['output_activation'] = torch.sigmoid
        elif params_dict['output_activation'] == 'softmax':
            params_dict['output_activation'] = nn.Softmax(dim=1)
        else:
            raise NotImplementedError()

    new_fn, _ = get_nn_from_params_dict(params_dict)
    new_fn.load('{}/{}.pth'.format(directory, name))
    new_fn.eval()
    return new_fn
