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
from HOUDINI.Library.NN import NetMLP, NetDO
from HOUDINI.Library.Utils import NNUtils


class NNLibrary(FnLibrary):
    def __init__(self, nn_dir, nns=None):
        self.items = dict()
        self.dir = nn_dir
        if nns:
            self.addNNItems(nns)

    def addNNItem(self, nn: Tuple[str, AST.PP]):
        obj = NNUtils.create_and_load(self.dir, nn[0])
        self.addItem(PPLibItem(nn[0], nn[1], obj))

    def addNNItems(self, nns: List[Tuple[str, AST.PP]]):
        for nn in nns:
            self.addNNItem(nn)
