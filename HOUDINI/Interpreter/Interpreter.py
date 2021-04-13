import sys
import math
import random
import pathlib
import numpy as np
from enum import Enum
from functools import reduce
from collections import OrderedDict
from typing import NamedTuple, Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Data.DataGenerator import NumpyDataSetIterator, ListNumpyDataSetIterator
from HOUDINI.Config import config
from HOUDINI.Library import Loss, Metric
from HOUDINI.Library.Utils import MetricUtils
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ReprUtils
from HOUDINI.Library import NN
from HOUDINI.Library.FnLibrary import FnLibrary


class ProgramOutputType(Enum):
    INTEGER = 1,  # using l2 regression, evaluating using round up/down
    SIGMOID = 2,
    SOFTMAX = 3,
    HAZARD = 4
    # REAL = 2 # using l2 regression, no accuracy evaluated. Perhaps can use some epsilon, not implemented
# ProgramRep = TypeVar('ProgramRep')


class Interpreter:
    """
    TODO: remember, once the program has been found, we might need to continue training it until convergence.
    Also, even if there's no new module introduced in the program, we might want to fine-tune existing modules.
    """

    def __init__(self,
                 settings,
                 library: FnLibrary,
                 evaluate_every_n_percent=1.):
        #  batch_size=150,
        #  epochs=1,
        #  lr=0.02,
        #  evaluate_every_n_percent=1.):
        print(settings)
        # self.settings = settings
        self.library = library
        self.data_dict = settings.data_dict
        self.batch_size = settings.batch_size
        self.epochs = settings.epochs
        self.lr = settings.learning_rate
        # epochs is changed according to data size, but this stays the same
        self.original_num_epochs = self.epochs
        self.evaluate_every_n_percent = evaluate_every_n_percent

    @classmethod
    def create_nns(cls, unknown_fns):
        """
        Creates the NN functions.
        :return: a tuple: ({fn_name: NN_Object}, trainable_parameters)
        """
        trainable_parameters = []
        new_fns_dict = {}
        for uf in unknown_fns:
            new_fns_dict[uf["name"]
                         ], c_trainable_params = NN.get_nn_from_params_dict(uf)
            if "freeze" in uf and uf["freeze"]:
                print("freezing the weight of {}".format(uf["name"]))
                continue
            trainable_parameters += list(c_trainable_params)

        return new_fns_dict, trainable_parameters

    def _get_data_loader(self, io_examples):
        if io_examples is None:
            return None
        if issubclass(type(io_examples), NumpyDataSetIterator):
            return io_examples
        elif type(io_examples) == tuple:
            return NumpyDataSetIterator(io_examples[0], io_examples[1], self.batch_size)
        elif type(io_examples) == list:
            loaders_list = []
            for io_eg in io_examples:
                if issubclass(type(io_eg), NumpyDataSetIterator):
                    loaders_list.append(io_eg)
                elif type(io_eg) == tuple:
                    loaders_list.append(NumpyDataSetIterator(
                        io_eg[0], io_eg[1], self.batch_size))
                else:
                    raise NotImplementedError
            return loaders_list
        else:
            raise NotImplementedError

    def _predict_data(self, program, data_loader, new_fns_dict):
        """
        An iterator, which executes the given program on mini-batches the data and returns the results.
        """
        # list of data_loader_iterators
        if issubclass(type(data_loader), NumpyDataSetIterator):
            data_loader = [data_loader]

        # creating a shallow copy of the list of iterators
        dl_iters_list = list(data_loader)
        while dl_iters_list.__len__() > 0:
            data_sample = None
            while data_sample is None and dl_iters_list.__len__() > 0:
                # choose an iterator at random
                c_rndm_iterator = random.choice(dl_iters_list)
                # try to get a data sample from it
                try:
                    data_sample = c_rndm_iterator.next()
                except StopIteration:
                    data_sample = None
                    # if there are no items left in the iterator, remove it from the list
                    dl_iters_list.remove(c_rndm_iterator)
            if data_sample is not None:
                x, y = data_sample
                # batch, env = y.shape[0], y.shape[1]
                # y = np.reshape(y, (batch*env, -1))
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                # print(y.shape)
                batch, env = y.shape[0], y.shape[1]
                y = torch.reshape(y, (batch * env, -1))
                x = Variable(
                    x).cuda() if torch.cuda.is_available() else Variable(x)
                y = Variable(
                    y).cuda() if torch.cuda.is_available() else Variable(y)
                # global_vars = {"lib": self.library, "inputs": x}
                global_vars = {"lib": self.library}
                global_vars = {**global_vars, **new_fns_dict}
                y_pred = eval(program, global_vars)(x.float())
                # print(len(y_pred), y_pred[0].shape, y_pred[1].shape)
                yield (y_pred, y.float(), x.float())

    def _get_accuracy(self,
                      program,
                      data_loader,
                      output_type,
                      new_fns_dict,
                      compute_grad=False):
        c_num_matching_datapoints = 0
        mse = 0  # atm, only accumulated, if the output is a real number
        global_vars = {"lib": self.library}
        global_vars = {**global_vars, **new_fns_dict}

        # data_loader_tr is either a dataloader or a list of dataloaders
        if issubclass(type(data_loader), NumpyDataSetIterator):
            num_datapoints = data_loader.num_datapoints
        else:
            num_datapoints = reduce(
                lambda a, b: a + b.num_datapoints, data_loader, 0.)
        y_pred_all, y_all, grad_all = list(), list(), list()
        # num_datapoints = data_loader.dataset.__len__() if type(data_loader) == DataLoader else data_loader[0].dataset.__len__()
        for y_pred, y, x_in in self._predict_data(program, data_loader, new_fns_dict):

            # if x is a 2d list, convert it to a variable
            graph = y_pred
            if type(graph) == list and type(graph[0]) == list:
                # check if it's a list of tuples
                if graph.__len__() > 0 and type(graph[0]) == list \
                        and type(graph[0][0]) == tuple:
                    # graph = [i[1] for i in graph]
                    graph = [[j[1] for j in i] for i in graph]

                if graph.__len__() > 0 and type(graph[0]) == list:
                    # concatenate all along cols
                    graph = [[torch.unsqueeze(j, dim=2)
                              for j in i] for i in graph]
                    graph = [torch.cat(i, dim=2) for i in graph]

                    # concatenate along rows
                    graph = [torch.unsqueeze(a, dim=2) for a in graph]
                    graph = torch.cat(graph, dim=2)
                y_pred = graph
            y_pred_output = y_pred if type(y_pred) != tuple else y_pred[1]
            if y_pred_output.shape.__len__() == 4 and y_pred_output.shape[1] == 2:
                y_pred_output = torch.squeeze(y_pred_output[:, 1, :, :])

            # in case of an int, also calcualte the sqrt(RMSE)
            if output_type == ProgramOutputType.INTEGER:
                num_outputs_per_data_point = 1
                for d in range(1, y_pred_output.shape.__len__()):
                    num_outputs_per_data_point *= y_pred_output.shape[d]
                torch_mse = F.mse_loss(y_pred_output, y, reduction='sum')
                mse += (torch_mse.cpu().data.numpy() /
                        float(num_outputs_per_data_point))

            if output_type == ProgramOutputType.INTEGER or output_type == ProgramOutputType.SIGMOID:
                y_pred_int = y_pred_output.data.cpu().numpy().round().astype(np.int)
                y_int = y.data.cpu().numpy().round().astype(np.int)
                c_num_matching_datapoints += (y_pred_int ==
                                              y_int).astype(np.float32).sum()
            elif output_type == ProgramOutputType.HAZARD:
                torch_mse = Loss.cox_ph_loss(y_pred_output, y)
                mse += torch_mse
                y_pred_all.append(y_pred_output)
                y_all.append(y)
            else:
                # y_pred_output_np = y_pred_output.data.cpu().numpy()
                y_pred_int = y_pred_output.data.cpu().numpy().argmax(
                    axis=1).reshape(-1, 1)  # get the index of the max log-probability
                y_int = y.data.cpu().numpy().reshape(-1, 1)
                c_num_matching_datapoints += (y_pred_int ==
                                              y_int).astype(np.float32).sum()

            # print(x_in.shape)
            if compute_grad:
                x_in = torch.autograd.Variable(x_in.clone(),
                                               requires_grad=True)
                x_pred = eval(program, global_vars)(x_in)
                # if type(y_pred) == tuple:
                #     x_pred = x_pred[1]
                grad_outputs = torch.ones(x_pred.shape)
                if torch.cuda.is_available():
                    grad_outputs = grad_outputs.cuda()
                x_grads = torch.autograd.grad(outputs=x_pred,
                                              inputs=x_in,
                                              grad_outputs=grad_outputs,
                                              create_graph=True,
                                              retain_graph=True,
                                              only_inputs=True)[0]
                x_grads = x_grads.detach().clone()
                x_grads = torch.reshape(x_grads, (-1, x_grads.shape[-1]))
                x_grads_norm = x_grads.norm(dim=0)
                grad_all.append(x_grads_norm.detach().clone().cpu().numpy())
                # print(x_grads.shape)
        if compute_grad:
            grad_all = np.asarray(grad_all)
            grad_mean = np.mean(grad_all, axis=0)
            outcome = self.data_dict['target']
            parents = set(self.data_dict['truth'])
            # number of all variables: causal var candidates + outcomet
            par_cand = list(range(grad_mean.shape[0] + 1))
            del par_cand[outcome]

            grad_idx = np.argsort(-grad_mean)
            grad_idx = grad_idx[:len(parents)]
            cand_set = set(np.array(par_cand)[grad_idx])
            jacob = len(parents.intersection(cand_set)) / \
                len(parents.union(cand_set))
            print(grad_mean, grad_idx, cand_set, outcome, parents, jacob)

            # print(grad_mean)

        accuracy = c_num_matching_datapoints / float(num_datapoints)
        mse = mse / float(num_datapoints)
        rmse = math.sqrt(mse)
        # if type(mse) == Variable:

        # returning -rmse, so that we select the best performance.
        if output_type in (ProgramOutputType.INTEGER, ProgramOutputType.HAZARD):
            perform_metric = -rmse
            if output_type == ProgramOutputType.HAZARD:
                y_all_np = torch.cat(y_all, dim=0).cpu().detach().numpy()
                y_pred_all_np = torch.cat(
                    y_pred_all, dim=0).cpu().detach().numpy()
                # print(y_all_np.shape, y_pred_all_np.shape)
                cox_metric = Metric.coxph(y_pred_all_np, y_all_np)
                cox_scores = cox_metric.eval_surv(y_pred_all_np, y_all_np)

                g_in = torch.tensor(
                    list(self.data_dict['clinical_meta']['causal'].values()))
                g_in = g_in.unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
                g_in = torch.autograd.Variable(g_in,
                                               requires_grad=True)
                g_pred = eval(program, global_vars)(g_in)
                # g_out = torch.ones(g_pred.shape)
                cox_grads = torch.autograd.grad(outputs=g_pred,
                                                inputs=g_in,
                                                # grad_outputs=g_out,
                                                create_graph=True,
                                                retain_graph=True,
                                                only_inputs=True)[0]
                cox_grads = cox_grads.cpu().detach().numpy()
                cox_grads = np.squeeze(cox_grads)
                return perform_metric, cox_scores, cox_grads
        else:
            perform_metric = accuracy
        return perform_metric, np.inf, np.inf

    def _clone_hidden_state(self, state):
        result = OrderedDict()
        for key, val in state.items():
            result[key] = val.clone()
        return result

    def learn_neural_network_(self,
                              program,
                              output_type,
                              new_fns_dict,
                              trainable_parameters,
                              data_loader_tr: NumpyDataSetIterator,
                              data_loader_val: NumpyDataSetIterator,
                              data_loader_test: NumpyDataSetIterator):
        if trainable_parameters is not None and trainable_parameters.__len__() == 0:
            print(
                "Warning! learn_neural_network_ called, with no learnable parameters! returning -inf accuracy.")
            return new_fns_dict, -sys.float_info.max, 0.0001
        if output_type == ProgramOutputType.INTEGER:
            criterion = torch.nn.MSELoss()
        elif output_type == ProgramOutputType.SIGMOID:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif output_type == ProgramOutputType.HAZARD:
            criterion = Loss.cox_ph_loss
        else:
            # combines log_softmax and cross-entropy
            criterion = F.cross_entropy

        optimizer = torch.optim.Adam(
            trainable_parameters, lr=self.lr, weight_decay=0.001)

        # data_loader_tr is either a dataloader or a list of dataloaders
        if issubclass(type(data_loader_tr), NumpyDataSetIterator):
            num_datapoints_tr = data_loader_tr.num_datapoints
        else:
            num_datapoints_tr = reduce(
                lambda a, b: a + b.num_datapoints, data_loader_tr, 0.)

        num_iterations_in_1_epoch = num_datapoints_tr // self.batch_size + \
            (0 if num_datapoints_tr % self.batch_size == 0 else 1)
        num_iterations = num_iterations_in_1_epoch * self.epochs
        evaluate_every_n_iters = int(
            self.evaluate_every_n_percent*num_iterations/100.)
        evaluate_every_n_iters = 1 if evaluate_every_n_iters == 0 else evaluate_every_n_iters
        # print("evaluate_every_n_iters=", evaluate_every_n_iters)

        # ***************** Train *****************
        max_accuracy = -sys.float_info.max
        # a dictionary of state_dicts for each new neural module
        max_accuracy_new_fns_states = {}
        accuracies_val = []
        accuracies_test = []
        iterations = []

        # set all new functions to train mode
        for key, value in new_fns_dict.items():
            value.train()

        prev_accuracy = 0
        current_iteration = 0
        for epoch in range(self.epochs):
            print("Starting epoch ", epoch, " / ", self.epochs)

            for y_pred, y, x_in in self._predict_data(program, data_loader_tr, new_fns_dict):
                # print(x_in.shape)
                # if x is a 2d list, convert it to a variable
                if type(y_pred) == list:
                    # check if it's a list of tuples
                    if y_pred.__len__() > 0 and type(y_pred[0]) == list \
                            and type(y_pred[0][0]) == tuple:
                        # y_pred = [i[1] for i in y_pred]
                        y_pred = [[j[1] for j in i] for i in y_pred]

                    if y_pred.__len__() > 0 and type(y_pred[0]) == list:
                        # concatenate all along cols
                        y_pred = [[torch.unsqueeze(j, dim=2)
                                   for j in i] for i in y_pred]
                        y_pred = [torch.cat(i, dim=2) for i in y_pred]

                        # concatenate along rows
                        y_pred = [torch.unsqueeze(a, dim=2) for a in y_pred]
                        y_pred = torch.cat(y_pred, dim=2)
                    y_pred = y_pred[:, 1, :, :]
                if type(y_pred) == tuple:
                    # if it's a tuple, then its (output_logits, output)
                    loss = criterion(y_pred[0], y)
                else:
                    # print(y.max())
                    # print(y_pred.data.shape)
                    loss = criterion(y_pred, y)
                #print("loss:", loss.data[0])
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if current_iteration % evaluate_every_n_iters == 0 or (epoch == self.epochs-1 and current_iteration == num_iterations-1):
                    # set all new functions to eval mode
                    for key, value in new_fns_dict.items():
                        value.eval()
                    c_accuracy = self._get_accuracy(program,
                                                    data_loader_val,
                                                    output_type,
                                                    new_fns_dict)[0]
                    c_accuracy_test = self._get_accuracy(program,
                                                         data_loader_test,
                                                         output_type,
                                                         new_fns_dict)[0]

                    accuracies_val.append(c_accuracy)
                    accuracies_test.append(c_accuracy_test)
                    iterations.append(current_iteration)

                    if max_accuracy < c_accuracy:
                        max_accuracy = c_accuracy
                        # store the state_dictionary of the best performing model
                        for new_fn_name, new_fn in new_fns_dict.items():
                            max_accuracy_new_fns_states[new_fn_name] = self._clone_hidden_state(
                                new_fn.state_dict())

                    print("c_accuracy", c_accuracy)

                    # set all new functions to train mode
                    for key, value in new_fns_dict.items():
                        value.train()

                current_iteration += 1

        print("max_accuracy_found_during_training:", max_accuracy)
        # set the state_dictionaries of the new functions to the model with best validation accuracy
        for new_fn_name, new_fn in new_fns_dict.items():
            new_fn.load_state_dict(max_accuracy_new_fns_states[new_fn_name])

        # set all new functions to eval mode
        for key, value in new_fns_dict.items():
            value.eval()

        num_evaluations = accuracies_val.__len__()
        evaluations_np = np.ones((num_evaluations, 3), dtype=np.float32)
        evaluations_np[:, 0] = iterations
        evaluations_np[:, 1] = accuracies_val
        evaluations_np[:, 2] = accuracies_test

        return new_fns_dict, max_accuracy, evaluations_np

    def _learn_neural_network(self,
                              program,
                              output_type,
                              unknown_fns,
                              data_loader_tr: NumpyDataSetIterator,
                              data_loader_val: NumpyDataSetIterator,
                              data_loader_test: NumpyDataSetIterator):
        # ***************** Set up the model *****************
        new_fns_dict, trainable_parameters = self.create_nns(unknown_fns)
        return self.learn_neural_network_(program,
                                          output_type,
                                          new_fns_dict,
                                          trainable_parameters,
                                          data_loader_tr,
                                          data_loader_val,
                                          data_loader_test=data_loader_test)

    def evaluate_(self,
                  program: str,
                  output_type: ProgramOutputType,
                  unknown_fns_def: List[Dict] = None,
                  io_examples_tr=None,
                  io_examples_val=None,
                  io_examples_test=None,
                  dbg_learn_parameters=True):
        """
        Independent from the synthesizer
        :param program:
        :param output_type:
        :param unknown_fns_def:
        :param io_examples_tr:
        :param io_examples_val:
        :param io_examples_test:
        :param dbg_learn_parameters: if False, it's not going to learn parameters
        :return:
        """
        # either io_examples_val is a tuple, or a list of tuples. handle accordingly.
        # a = issubclass(NumpyDataSetIterator, NumpyDataSetIterator)
        data_loader_tr = self._get_data_loader(io_examples_tr)
        data_loader_val = self._get_data_loader(io_examples_val)
        data_loader_test = self._get_data_loader(io_examples_test)
        assert(type(output_type) == ProgramOutputType)

        new_fns_dict = {}
        val_accuracy, test_accuracy = list(), list()
        cox_scores, cox_grads = list(), list()
        repeat = self.data_dict['repeat']
        is_lganm = self.data_dict['dict_name'].lower() == 'lganm'
        for _ in range(repeat):
            evaluations_np = np.ones((1, 1))
            if unknown_fns_def is not None and unknown_fns_def.__len__() > 0:
                if not dbg_learn_parameters:
                    new_fns_dict, _ = self.create_nns(unknown_fns_def)
                else:
                    new_fns_dict, val_acc, evaluations_np = self._learn_neural_network(program,
                                                                                       output_type,
                                                                                       unknown_fns_def,
                                                                                       data_loader_tr,
                                                                                       data_loader_val,
                                                                                       data_loader_test)
            val_acc = self._get_accuracy(program,
                                         data_loader_val,
                                         output_type,
                                         new_fns_dict,
                                         compute_grad=is_lganm)

            test_acc = self._get_accuracy(program,
                                          data_loader_test,
                                          output_type,
                                          new_fns_dict)
            val_accuracy.append(val_acc[0])
            test_accuracy.append(test_acc[0])
            if output_type == ProgramOutputType.HAZARD:
                cox_scores.append(val_acc[1])
                cox_grads.append(val_acc[2])

        val_accuracy = sum(val_accuracy) / len(val_accuracy)
        test_accuracy = sum(test_accuracy) / len(test_accuracy)
        if output_type == ProgramOutputType.HAZARD:
            cox_scores = list(zip(*cox_scores))
            cox_grads = np.asarray(cox_grads)
            cox_index = list(range(cox_grads.shape[1]))
            cox_index = list(self.data_dict['clinical_meta']['causal'].keys())
            cox_dir = self.data_dict['res_dir']
            # print(cox_index, cox_grads.shape)
            cox_utils = MetricUtils.coxsum(cox_index, cox_grads)
            cox_utils.vis_plot(cox_scores,
                               pathlib.Path(cox_dir),
                               self.data_dict['metric_scores'])
            print(cox_utils.summary(pathlib.Path(cox_dir)))
        print("validation accuracy=", val_accuracy)
        print("test accuracy=", test_accuracy)
        return {"accuracy": val_accuracy, "new_fns_dict": new_fns_dict,
                "test_accuracy": test_accuracy, "evaluations_np": evaluations_np}

    # program=st, output_type=output_type, unkSortMap=unkSortMap, io_examples=self.ioExamples
    def evaluate(self, program, output_type_s, unkSortMap=None,
                 io_examples_tr=None, io_examples_val=None, io_examples_test=None, dbg_learn_parameters=True) -> dict:

        is_graph = type(output_type_s) == AST.PPGraphSort

        program_str = ReprUtils.repr_py(program)
        output_type = self.get_program_output_type(
            io_examples_val, output_type_s)

        unknown_fns_def = _get_unknown_fns_definitions(unkSortMap, is_graph)
        res = self.evaluate_(program=program_str, output_type=output_type, unknown_fns_def=unknown_fns_def,
                             io_examples_tr=io_examples_tr, io_examples_val=io_examples_val,
                             io_examples_test=io_examples_test,
                             dbg_learn_parameters=dbg_learn_parameters)
        return res

    def get_program_output_type(self, io_examples_val, output_sort):
        if self.data_dict['out_type'] == 'hazard':
            output_type = ProgramOutputType.HAZARD
        elif self.data_dict['out_type'] == 'integer':
            output_type = ProgramOutputType.INTEGER
        else:
            raise TypeError('invalid output type {}'.format(self.data_dict['out_type']))

        # if issubclass(type(io_examples_val), NumpyDataSetIterator):
        #     val_labels = io_examples_val.targets
        #     label_shape = val_labels.shape
        #     label_max_value = val_labels.max()
        #     label_min_value = val_labels.min()
        # elif type(io_examples_val) == tuple:
        #     label_shape = io_examples_val[1].shape
        #     label_max_value = io_examples_val[1].max()
        #     label_min_value = io_examples_val[1].min()
        # else:
        #     label_shape = io_examples_val[0][1].shape
        #     label_max_value = io_examples_val[0][1].max()
        #     label_min_value = io_examples_val[0][1].min()

        # # print('label shape', label_shape)
        # # deduce output by the target examples
        # # dim = 1, Real => Regression
        # # dim = 1, Int => Classification
        # # dim = 1, Real, [0, 1] => 1d Classification
        # if type(output_sort) == AST.PPGraphSort:
        #     output_type = ProgramOutputType.INTEGER
        # elif type(output_sort.param_sort) == AST.PPBool:
        #     output_type = ProgramOutputType.SOFTMAX if label_max_value > 1. else ProgramOutputType.SIGMOID
        # elif type(output_sort.param_sort) == AST.PPReal:
        #     # if the label has two dim: RFS and RFSyears
        #     if len(label_shape) == 3 and label_shape[-1] == 2:
        #         output_type = ProgramOutputType.HAZARD
        #     else:
        #         output_type = ProgramOutputType.INTEGER
        # else:
        #     raise TypeError('invalid output type {}'.format(type(output_sort)))
        return output_type


def _get_unknown_fns_definitions(unkSortMap, is_graph=False):
    # TODO: double-check. may need re-writing.
    unk_fns_interpreter_def_list = []

    for unk_fn_name, unk_fn in unkSortMap.items():

        # ******** Process output activation ***************
        fn_output_sort = unk_fn.rtpe
        output_dim = fn_output_sort.shape[1].value
        output_type = fn_output_sort.param_sort
        if is_graph:
            output_activation = None
        elif type(fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 2:

            if type(output_type) == AST.PPReal or type(output_type) == AST.PPInt:
                output_activation = None
            elif type(output_type) == AST.PPBool and output_dim == 1:
                output_activation = torch.sigmoid
            elif type(output_type) == AST.PPBool and output_dim > 1:
                output_activation = nn.Softmax(dim=1)
            else:
                raise NotImplementedError()

            print(fn_output_sort, type(output_type))
        # the only other possibility
        elif not (type(fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 4):
            raise NotImplementedError()

        fn_input_sort = unk_fn.args[0]
        if type(fn_input_sort) == AST.PPListSort:
            if is_graph:
                input_dim = fn_input_sort.param_sort.shape[1].value
                uf = {"type": "GCONVNew", "name": unk_fn_name,
                      "input_dim": input_dim}
                unk_fns_interpreter_def_list.append(uf)
            else:
                raise NotImplementedError()
                # input_list_item_sort = fn_input_sort.param_sort
                # # make sure the items in the list are tensors
                # assert(type(input_list_item_sort) == AST.PPTensorSort)
                # input_dim = fn_input_sort.param_sort.shape[1].value
                # hidden_dim = 100
                # # uf = {"type": "RNN", "name": unk_fn_name, "input_dim": input_dim, "hidden_dim": hidden_dim,
                # #       "output_dim": output_dim, "output_activation": output_activation}
                # uf = {"type": "MLP", "name": unk_fn_name, "input_dim": input_dim,
                #       "output_dim": output_dim, "output_activation": output_activation}
                # unk_fns_interpreter_def_list.append(uf)
        elif type(fn_input_sort) == AST.PPTensorSort and fn_input_sort.shape.__len__() == 4:
            if type(fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 4:
                # it's a cnn
                input_dim = fn_input_sort.shape[2].value
                input_ch = fn_input_sort.shape[1].value
                uf = {"type": "CNN", "name": unk_fn_name,
                      "input_dim": input_dim, "input_ch": input_ch}
                unk_fns_interpreter_def_list.append(uf)
            elif type(fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 2:
                # FROM CNN's features to a vector using an MLP
                input_dim = fn_input_sort.shape[1].value * \
                    fn_input_sort.shape[2].value * fn_input_sort.shape[3].value
                uf = {"type": "MLP", "name": unk_fn_name, "input_dim": input_dim,
                      "output_dim": output_dim, "output_activation": output_activation}
                unk_fns_interpreter_def_list.append(uf)
            else:
                raise NotImplementedError()

        elif type(fn_input_sort) == AST.PPTensorSort and fn_input_sort.shape.__len__() == 2:
            input_dim = fn_input_sort.shape[1].value
            if unk_fn.args.__len__() == 2:
                fn_input2_sort = unk_fn.args[1]
                input_dim += fn_input2_sort.shape[1].value
            uf = {"type": "MLP", "name": unk_fn_name, "input_dim": input_dim,
                  "output_dim": output_dim, "output_activation": output_activation}
            unk_fns_interpreter_def_list.append(uf)
        else:
            raise NotImplementedError()

        # uf = {"type": "CNN", "name": "nn_fun_1", "input_dim": 28,
        #      "input_ch": 1, "output_dim": 1, "output_activation": torch.sigmoid, "is_last": False}
        # unk_fns_interpreter_def_list.append(uf)

    return unk_fns_interpreter_def_list
