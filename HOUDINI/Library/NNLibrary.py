import numpy as np
import pickle
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional, Tuple, Union, List

from HOUDINI.Synthesizer import AST
from HOUDINI.Library.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Library.NN import NetCNN, NetGRAPHNew, NetMLP, NetRNN


class NNLibrary(FnLibrary):
    def __init__(self, nn_dir, nns=None):
        self.items = dict()
        self.dir = nn_dir
        if nns:
            self.addNNItems(nns)

    def addNNItem(self, nn: Tuple[str, AST.PP]):
        obj = self.create_and_load(nn[0])
        self.addItem(PPLibItem(nn[0], nn[1], obj))

    def addNNItems(self, nns: List[Tuple[str, AST.PP]]):
        for nn in nns:
            self.addNNItem(nn)

    def create_and_load(self, name, new_name=None):
        if new_name is None:
            new_name = name

        with open('{}/{}.json'.format(self.dir, name)) as json_data:
            params_dict = json.load(json_data)
            params_dict["name"] = new_name

            if params_dict["output_activation"] == "None":
                params_dict["output_activation"] = None
            elif params_dict["output_activation"] == "sigmoid":
                params_dict["output_activation"] = torch.sigmoid
            elif params_dict["output_activation"] == "softmax":
                params_dict["output_activation"] = nn.Softmax(dim=1)
            else:
                raise NotImplementedError

        new_fn, _ = self.get_nn_from_params_dict(params_dict)
        # print(new_name)
        # new_fn = new_fn_dict[new_name]
        new_fn.load("{}/{}.pth".format(self.dir, name))
        new_fn.eval()
        return new_fn

    def get_nn_from_params_dict(self, uf):
        new_nn = None
        if uf["type"] == "MLP":
            new_nn = NetMLP(uf["name"],
                            uf["input_dim"],
                            uf["output_dim"],
                            uf["output_activation"])
        elif uf["type"] == "CNN":
            new_nn = NetCNN(uf["name"],
                            uf["input_dim"],
                            uf["input_ch"])
        elif uf["type"] == "RNN":
            output_dim = uf["output_dim"] if "output_dim" in uf else None
            output_activation = uf["output_activation"] if "output_activation" in uf else None
            output_sequence = uf["output_sequence"] if "output_sequence" in uf else False
            new_nn = NetRNN(uf["name"],
                            uf["input_dim"],
                            uf["hidden_dim"],
                            output_dim=output_dim,
                            output_activation=output_activation,
                            output_sequence=output_sequence)
        elif uf["type"] == "GCONVNew":
            new_nn = NetGRAPHNew(uf["name"],
                                 None,
                                 uf["input_dim"],
                                 num_output_channels=100)
        else:
            raise NotImplementedError()

        if "initialize_from" in uf and uf["initialize_from"] is not None:
            new_nn.load(uf["initialize_from"])

        if torch.cuda.is_available():
            new_nn.cuda()

        new_nn.params_dict = uf
        c_trainable_parameters = list(new_nn.parameters())

        return new_nn, c_trainable_parameters
