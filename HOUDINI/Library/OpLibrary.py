import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional, Tuple, Union, List

from HOUDINI.Synthesizer import AST
from HOUDINI.Library.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Library import Op
from HOUDINI.Library.NN import NetCNN, SaveableNNModule
from HOUDINI.Synthesizer.AST import mkTensorSort, mkFuncSort, mkListSort, mkRealTensorSort, mkGraphSort


class OpLibrary(FnLibrary):
    def __init__(self, ops):
        self.items = dict()
        self.A = AST.PPSortVar('A')
        self.B = AST.PPSortVar('B')
        self.C = AST.PPSortVar('C')

        self.addOpItems(ops)

    def func(self, *lst):
        return mkFuncSort(*lst)

    def lst(self, t):
        return mkListSort(t)

    def graph(self, t):
        return mkGraphSort(t)

    def addOpItem(self, op: str):
        if op == 'compose':
            self.addItem(PPLibItem('compose', self.func(self.func(self.B,
                                                                  self.C),
                                                        self.func(self.A,
                                                                  self.B),
                                                        self.func(self.A,
                                                                  self.C)), Op.pp_compose))
        elif op == 'repeat':
            self.addItem(PPLibItem('repeat', self.func(AST.PPEnumSort(1, 4),
                                                       self.func(self.A,
                                                                 self.A),
                                                       self.func(self.A,
                                                                 self.A)), Op.pp_repeat))
        elif op == 'cat':
            self.addItem(PPLibItem('cat', self.func(self.func(self.A,
                                                              self.lst(self.B)),
                                                    self.func(self.A,
                                                              self.B)), Op.pp_cat))
        elif op == 'map':
            self.addItem(PPLibItem('map', self.func(self.func(self.A,
                                                              self.B),
                                                    self.func(self.lst(self.A),
                                                              self.lst(self.B))), Op.pp_map))
        elif op == 'fold':
            self.addItem(PPLibItem('fold', self.func(self.func(self.B, self.A, self.B),
                                                     self.B,
                                                     self.func(self.lst(self.A), self.B)), Op.pp_reduce))
        elif op == 'conv':
            self.addItem(PPLibItem('conv', self.func(self.func(self.lst(self.A), self.B),
                                                     self.func(self.lst(self.A), self.lst(self.B))), Op.pp_conv))
        elif op == 'zeros':
            self.addItem(PPLibItem('zeros', self.func(AST.PPDimVar('a'),
                                                      mkRealTensorSort([1, 'a'])), Op.pp_get_zeros))
        elif op == 'conv_g':
            self.addItem((PPLibItem('conv_g', self.func(self.func(self.lst(self.A), self.B),
                                                        self.func(self.graph(self.A), self.graph(self.B))), Op.pp_conv_graph)))
        elif op == 'map_g':
            self.addItem((PPLibItem('map_g', self.func(self.func(self.A, self.B),
                                                       self.func(self.graph(self.A), self.graph(self.B))), Op.pp_map_g)))
        elif op == 'fold_g':
            self.addItem((PPLibItem('fold_g', self.func(self.func(self.B, self.A, self.B),
                                                        self.B,
                                                        self.func(self.graph(self.A), self.B)), Op.pp_reduce_graph)))
        elif op == 'cat_l':
            self.addItem(PPLibItem('cat_l', self.func(self.lst(self.A),
                                                      self.B), Op.pp_cat_list))

        elif op == 'flatten_2d_list':
            self.addItem(PPLibItem('flatten_2d_list', self.func(self.func(self.B,
                                                                          self.C),
                                                                self.func(self.A,
                                                                          self.B),
                                                                self.func(self.A,
                                                                          self.C)), Op.pp_flatten_2d_list))
        else:
            raise NameError(
                'Op name {} does not have corresponding function'.format(op))

    def addOpItems(self, ops: List[str]):
        for op in ops:
            self.addOpItem(op)


# lib_items_repo = dict()


# def add_lib_item(li):
#     lib_items_repo[li.name] = li


# def add_libitems_to_repo():
#     def func(*lst):
#         return mkFuncSort(*lst)

#     def lst(t):
#         return mkListSort(t)

#     def graph(t):
#         return mkGraphSort(t)


#     add_lib_item(PPLibItem('compose', func(
#         func(self.B, self.C), func(self.A, self.B), func(self.A, self.C)), Op.pp_compose))
#     add_lib_item(PPLibItem('repeat', func(
#         AST.PPEnumSort(9, 10), func(self.A, self.A), func(self.A, self.A)), Op.pp_repeat))
#     add_lib_item(PPLibItem('map_l', func(
#         func(self.A, self.B), func(lst(self.A), lst(self.B))), Op.pp_map_list))
#     add_lib_item(PPLibItem('fold_l', func(
#         func(self.B, self.A, self.B), self.B, func(lst(self.A), self.B)), Op.pp_reduce_list))
#     add_lib_item(PPLibItem('conv_l', func(
#         func(lst(self.A), self.B), func(lst(self.A), lst(self.B))), Op.pp_conv_list))
#     add_lib_item(PPLibItem('zeros', func(AST.PPDimVar('a'),
#                                          mkRealTensorSort([1, 'a'])), Op.pp_get_zeros))

#     add_lib_item(PPLibItem('conv_g', func(func(lst(self.A), self.B),
#                                           func(graph(self.A), graph(self.B))), Op.pp_conv_graph))
#     add_lib_item(PPLibItem('map_g', func(
#         func(self.A, self.B), func(graph(self.A), graph(self.B))), Op.pp_map_g))
#     add_lib_item(PPLibItem('fold_g', func(
#         func(self.B, self.A, self.B), self.B, func(graph(self.A), self.B)), Op.pp_reduce_graph))

#     add_lib_item(PPLibItem('flatten_2d_list', func(
#         func(self.B, self.C), func(self.A, self.B), func(self.A, self.C)), Op.pp_flatten_2d_list))


# def get_items_from_repo(item_names: List[str]):
#     lib_items = [lib_items_repo[item_name] for item_name in item_names]
#     return lib_items


# def get_item_from_repo(item_name: str):
#     lib_item = lib_items_repo[item_name]
#     return lib_item


# def get_all_items_from_repo():
#     return lib_items_repo.values
