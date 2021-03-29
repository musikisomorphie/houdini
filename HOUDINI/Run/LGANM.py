import argparse
import sys
import random
import sempler
import numpy as np
import networkx as nx
from collections import namedtuple
from enum import Enum
from pathlib import Path

from HOUDINI.Config import config
from HOUDINI.Run.Task import Task
from HOUDINI.Run.Task import TaskSettings
from HOUDINI.Run.TaskSeq import TaskSeqSettings, TaskSeq
from HOUDINI.Run.Utils import get_portec_io_examples
from HOUDINI.Library.OpLibrary import OpLibrary
from HOUDINI.Library.FnLibrary import FnLibrary
from HOUDINI.Synthesizer.AST import mkFuncSort, mkRealTensorSort, mkListSort, mkBoolTensorSort


class IdentifyTask(Task):
    def __init__(self,
                 exp_dict,
                 settings,
                 seq,
                 dbg_learn_parameters):
        self.exp_dict = exp_dict()
        self.feat = portec['clinical_meta']['causal']
        self.label = portec['clinical_meta']['outcome'][0]
        self.file = portec['file']
        input_type = mkRealTensorSort([1, len(self.feat)])
        output_type = mkBoolTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def _graph_info(self,
                    i,
                    W,
                    interventions=None):
        """Returns the parents, children, parents of children and markov
        blanket of variable i in DAG W, using the graph structure
        """
        G = nx.from_numpy_matrix(W, create_using=nx.DiGraph)
        parents = set(G.predecessors(i))
        children = set(G.successors(i))
        parents_of_children = set()
        for child in children:
            parents_of_children.update(G.predecessors(child))
        if len(children) > 0:
            parents_of_children.remove(i)
        mb = parents.union(children, parents_of_children)
        return (parents, children, parents_of_children, mb)

    def _gen_cases(self):
        if 'random_state' in self.exp_dict:
            np.random.seed(self.exp_dict['random_state'])
        cases = []
        # i = 0
        # while i < self.exp_dict['n']:
        for i in range(self.exp_dict['n']):
            if self.exp_dict['p_min'] != self.exp_dict['p_max']:
                p = np.random.randint(self.exp_dict['p_min'],
                                      self.exp_dict['p_max']+1)
            else:
                p = self.exp_dict['p_min']
            W = sempler.dag_avg_deg(p,
                                    self.exp_dict['k'],
                                    self.exp_dict['w_min'],
                                    self.exp_dict['w_max'])
            target = np.random.choice(range(p))
            parents = self._graph_info(target, W)[0]
            if bool(parents):  # and len(parents) != len(mb):
                sem = sempler.LGANM(W,
                                    (self.exp_dict['var_min'],
                                     self.exp_dict['var_max']),
                                    (self.exp_dict['int_min'],
                                     self.exp_dict['int_max']))
                # truth = self._graph_info(target, W)[0]
                cases.append((i, sem, target, parents))
                # i += 1
        return cases

    def get_io_examples(self):
        if 'load_dataset' in self.exp_dict:
            dt_dir = Path(self.exp_dict['load_dataset'])
            print('\nLoading test cases from {}'.format(dt_dir))
            Ws, means, variances, targets = [], [], [], []
            dag_dir = dt_dir / 'dags'
            for d in dag_dir.glob('*'):
                if d.is_dir():
                    Ws.append(np.loadtxt(str(d / 'adjacency.txt')))
                    means.append(np.loadtxt(str(d / 'means.txt')))
                    variances.append(np.loadtxt(str(d / 'variances.txt')))
                    targets.append(int(np.loadtxt(str(d / 'target.txt'))))

            cases = []
            for i, W in enumerate(Ws):
                sem = sempler.LGANM(W, variances[i], means[i])
                parents = self._graph_info(targets[i], W)[0]
                cases.append((i, sem, targets[i], parents))
        else:
            cases = self._gen_cases()
        return get_portec_io_examples(self.file,
                                      self.feat,
                                      self.label)

    def name(self):
        return 'classify_RFS'

    def sname(self):
        return 'cr'
