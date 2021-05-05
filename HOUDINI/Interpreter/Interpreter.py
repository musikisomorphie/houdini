import sys
import math
import random
import pathlib
import numpy as np
from enum import Enum
from functools import reduce
from collections import OrderedDict
from scipy import stats
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
from HOUDINI.Library.Utils import NNUtils
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
                 evaluate_every_n_percent=10.):
        #  batch_size=150,
        #  epochs=1,
        #  lr=0.02,
        #  evaluate_every_n_percent=1.):
        # print(settings)
        # self.settings = settings
        self.library = library
        self.data_dict = settings.data_dict
        self.batch_size = settings.batch_size
        self.epochs = settings.epochs
        self.lr = settings.learning_rate
        # epochs is changed according to data size, but this stays the same
        self.original_num_epochs = self.epochs

        if self.data_dict['dict_name'] != 'portec':
            self.var_num = self.data_dict['mid_size'] - 1
        else:
            self.var_num = self.data_dict['mid_size'] - 2
        self.evaluate_every_n_percent = evaluate_every_n_percent

    @classmethod
    def create_nns(cls, unknown_fns):
        """
        Creates the NN functions.
        :return: a tuple: ({fn_name: NN_Object}, trainable_parameters)
        """
        trainable_parameters = {'do': list(),
                                'non-do': list()}
        new_fns_dict = dict()
        for uf in unknown_fns:
            fns_nn = NNUtils.get_nn_from_params_dict(uf)
            new_fns_dict[uf["name"]] = fns_nn[0]
            c_trainable_params = fns_nn[1]
            if "freeze" in uf and uf["freeze"]:
                print("freezing the weight of {}".format(uf["name"]))
                continue
            if uf['type'] == 'DO':
                trainable_parameters['do'] += list(c_trainable_params)
            else:
                trainable_parameters['non-do'] += list(c_trainable_params)
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

    def _compute_grad(self,
                      x_in,
                      program,
                      global_vars):

        x_in = torch.autograd.Variable(x_in,
                                       requires_grad=True)
        x_pred = eval(program, global_vars)(x_in)
        if type(x_pred) == tuple:
            x_pred = x_pred[0]
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
        return x_grads

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
                # y = torch.reshape(y, (batch * env, -1))
                y = torch.cat(torch.split(y, 1, dim=1), dim=0).squeeze(dim=1)
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
                      compute_grad=False,
                      compute_prob=False):
        global_vars = {"lib": self.library}
        global_vars = {**global_vars, **new_fns_dict}

        if issubclass(type(data_loader), NumpyDataSetIterator):
            num_datapoints = data_loader.num_datapoints
        else:
            num_datapoints = reduce(
                lambda a, b: a + b.num_datapoints, data_loader, 0.)

        c_num_matching_datapoints = 0
        mse = list()  # atm, only accumulated, if the output is a real number
        debug_y = list()
        y_pred_all, y_all = list(), list()
        grad_all, prob_all = list(), list()
        for y_pred, y, x_in in self._predict_data(program, data_loader, new_fns_dict):
            # for i in range(12):
            #     dt_num = y.shape[0] // 12
            #     if not torch.all(y[i * dt_num: (i + 1) * dt_num, -1] == i):
            #         print('false env')
            #         print(y[i * dt_num: (i + 1) * dt_num])
            if type(y_pred) == tuple:
                y_pred_output, pred_prob = y_pred
            else:
                pred_prob = None

            # in case of an int, also calcualte the sqrt(RMSE)
            if output_type == ProgramOutputType.INTEGER:
                torch_mse = F.mse_loss(y_pred_output, y, reduction='none')
            elif output_type == ProgramOutputType.HAZARD:
                torch_mse = Loss.cox_ph_loss(
                    y_pred_output, y, reduction='none')
                y_pred_all.append(y_pred_output)
                y_all.append(y)
            else:
                raise NotImplementedError()

            np_mse = torch_mse.detach().cpu().numpy()
            # np.split is different to torch.split
            mse.append(np.split(np_mse, self.data_dict['env_num'], axis=0))

            # np_y = y.detach().cpu().numpy()
            # debug_y.append(np.split(np_y, self.data_dict['env_num'], axis=0))

            if compute_grad:
                x_grads = self._compute_grad(x_in.clone(),
                                             program,
                                             global_vars)
                x_grads = torch.reshape(x_grads, (-1, x_grads.shape[-1]))
                x_grads_norm = x_grads.norm(dim=0)
                grad_all.append(x_grads_norm.detach().clone().cpu().numpy())

        grad_mean = None
        if compute_grad:
            grad_all = np.asarray(grad_all)
            grad_mean = np.mean(grad_all, axis=0)
            # outcome = self.data_dict['target']
            # parents = set(self.data_dict['truth'])
            # # number of all variables: causal var candidates + outcomet
            # par_cand = list(range(grad_mean.shape[0] + 1))
            # del par_cand[outcome]

            # grad_idx = np.argsort(-grad_mean)
            # grad_idx = grad_idx[:len(parents)]
            # cand_set = set(np.array(par_cand)[grad_idx])
            # jacob = len(parents.intersection(cand_set)) / \
            #     len(parents.union(cand_set))

            # self.data_dict['grads'] = grad_mean
            # print(grad_mean, grad_idx, cand_set, outcome, parents, jacob)

        mse = list(zip(*mse))
        mse = [np.concatenate(mse_env, axis=0) for mse_env in mse]

        # debug_y = list(zip(*debug_y))
        # for debug_id, debug in enumerate(debug_y):
        #     debug = np.concatenate(debug, axis=0)
        #     assert np.all(debug == debug_id)

        # returning -rmse, so that we select the best performance.
        if output_type == ProgramOutputType.HAZARD:
            y_all_np = torch.cat(y_all, dim=0).cpu().detach().numpy()
            y_pred_all_np = torch.cat(
                y_pred_all, dim=0).cpu().detach().numpy()
            cox_metric = Metric.coxph(y_pred_all_np, y_all_np)
            cox_scores = cox_metric.eval_surv(y_pred_all_np, y_all_np)

            g_in = torch.tensor(
                list(self.data_dict['clinical_meta']['causal'].values()))
            g_in = g_in.unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
            cox_grads = self._compute_grad(g_in,
                                           program,
                                           global_vars)

            cox_grads = cox_grads.cpu().numpy()
            cox_grads = np.squeeze(cox_grads)
            return mse, cox_grads, cox_scores
        elif output_type == ProgramOutputType.INTEGER:
            return mse, grad_mean, None
        else:
            raise NotImplementedError()

    def _clone_hidden_state(self, state):
        result = OrderedDict()
        for key, val in state.items():
            result[key] = val.clone()
        return result

    def learn_neural_network(self,
                             program,
                             output_type,
                             unknown_fns,
                             data_loader_tr: NumpyDataSetIterator,
                             data_loader_val: NumpyDataSetIterator,
                             data_loader_test: NumpyDataSetIterator):

        if output_type == ProgramOutputType.INTEGER:
            criterion = torch.nn.MSELoss(reduction='none')
        elif output_type == ProgramOutputType.SIGMOID:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif output_type == ProgramOutputType.HAZARD:
            criterion = Loss.cox_ph_loss
        else:
            # combines log_softmax and cross-entropy
            criterion = F.cross_entropy

        # data_loader_tr is either a dataloader or a list of dataloaders
        if issubclass(type(data_loader_tr), NumpyDataSetIterator):
            num_datapoints_tr = data_loader_tr.num_datapoints
        else:
            num_datapoints_tr = reduce(
                lambda a, b: a + b.num_datapoints, data_loader_tr, 0.)

        num_iterations_in_1_epoch = num_datapoints_tr // self.batch_size + \
            (0 if num_datapoints_tr % self.batch_size == 0 else 1)
        num_iterations = num_iterations_in_1_epoch * self.epochs
        evaluate_every_n_iters = int(num_iterations_in_1_epoch / self.var_num)
        evaluate_every_n_iters = 1 if evaluate_every_n_iters == 0 else evaluate_every_n_iters
        eval_num = math.ceil(num_iterations_in_1_epoch /
                             evaluate_every_n_iters)
        # print("evaluate_every_n_iters=", evaluate_every_n_iters)

        # ***************** Train *****************
        max_accuracy = -sys.float_info.max
        # a dictionary of state_dicts for each new neural module
        max_accuracy_new_fns_states = dict()
        sota_acc_new_fns_states = dict()
        sota_acc = -sys.float_info.max
        sota_var = 0
        sota_grad = list()
        sota_idx = None
        sota_mse = None
        accuracies_val = list()
        accuracies_test = list()
        iterations = list()

        new_fns_dict, trainable_parameters = self.create_nns(unknown_fns)
        global_vars = {"lib": self.library}
        global_vars = {**global_vars, **new_fns_dict}
        parm_all = trainable_parameters['do'] + trainable_parameters['non-do']
        parm_do = trainable_parameters['do']
        optim_all = torch.optim.Adam(parm_all,
                                     lr=self.lr,
                                     weight_decay=0.001)

        # set all new functions to train mode
        for _, value in new_fns_dict.items():
            value.train()

        current_iteration = 0
        reject_var = set()
        accept_var = set()
        epoch = 0
        has_trained_more = False
        is_lganm = self.data_dict['dict_name'].lower() == 'lganm'
        # for epoch in range(self.epochs):
        while len(accept_var) + len(reject_var) < 11:
            print("Starting epoch ", epoch)
            iter_in_one_epoch = 0
            prob_all, metric_all = 0, list()
            for y_pred, y, x_in in self._predict_data(program, data_loader_tr, new_fns_dict):
                # for i in range(12):
                #     dt_num = y.shape[0] // 12
                #     if not np.all(y[i * dt_num: (i + 1) * dt_num, -1].detach().cpu().numpy() == i):
                #         print('false env')
                #         print(y[i * dt_num: (i + 1) * dt_num])
                if type(y_pred) == tuple:
                    y_pred = y_pred[0]

                # wei = torch.ones(11 * 64, requires_grad=False).cuda()
                # wei[10:64 :11*64] = 0.1

                # loss = wei * criterion(y_pred, y)
                loss = criterion(y_pred, y)
                optim_all.zero_grad()
                (loss.mean()).backward(retain_graph=True)
                optim_all.step()

                # if iter_in_one_epoch % evaluate_every_n_iters == 0:
                # if parm_do and iter_in_one_epoch == num_iterations_in_1_epoch - 1:
                # print(prob_all)

                iter_in_one_epoch += 1
                current_iteration += 1

            for _, value in new_fns_dict.items():
                value.eval()

            if epoch == 7:
                val_mse, sota_grad = self._get_accuracy(program,
                                                        data_loader_val,
                                                        output_type,
                                                        new_fns_dict,
                                                        compute_grad=is_lganm)[:2]
                prob_idx = np.argsort(sota_grad).tolist()
                print(sota_grad, prob_idx)

            else:
                val_mse = self._get_accuracy(program,
                                             data_loader_val,
                                             output_type,
                                             new_fns_dict)[0]

            tst_mse = self._get_accuracy(program,
                                         data_loader_test,
                                         output_type,
                                         new_fns_dict)[0]

            val_acc = -np.mean(val_mse)
            val_var = np.var(val_mse)
            tst_acc = -np.mean(tst_mse)
            tst_var = np.var(tst_mse)

            if sota_acc < val_acc:
                sota_acc = val_acc
                sota_var = val_var
                sota_mse = val_mse
                for new_fn_name, new_fn in new_fns_dict.items():
                    sota_acc_new_fns_states[new_fn_name] = self._clone_hidden_state(
                        new_fn.state_dict())

            if parm_do and epoch > 7:
                wass_dis, cur_mean = self._wass(val_mse, sota_mse)
                if wass_dis > -8 * sota_acc or sota_grad[sota_idx] > 10:
                    if not has_trained_more:
                        has_trained_more = True
                        continue
                    else:
                        has_trained_more = False
                        accept_var.add(sota_idx)
                        for new_fn_name, new_fn in new_fns_dict.items():
                            new_fn.load_state_dict(
                                sota_acc_new_fns_states[new_fn_name])
                else:
                    if has_trained_more:
                        has_trained_more = False
                    reject_var.add(sota_idx)
                    for new_fn_name, new_fn in new_fns_dict.items():
                        sota_acc_new_fns_states[new_fn_name] = self._clone_hidden_state(
                            new_fn.state_dict())

                print(sota_idx, sota_acc, wass_dis,
                      accept_var, reject_var)
                print(cur_mean, sota_grad)

            if parm_do and epoch >= 7:
                with torch.no_grad():
                    for idx in prob_idx:
                        if not ((idx in accept_var) or (idx in reject_var)):
                            sota_idx = idx
                            # sota_grad = val_grad[idx]
                            tmp = parm_do[0].detach().clone()
                            tmp[0, idx] -= 10
                            parm_do[0].copy_(tmp)
                            break

            # set all new functions to train mode
            for _, value in new_fns_dict.items():
                value.train()

            accuracies_val.append(val_acc)
            accuracies_test.append(tst_acc)
            iterations.append(current_iteration)

            if max_accuracy < val_acc:
                max_accuracy = val_acc
                # store the state_dictionary of the best performing model
                for new_fn_name, new_fn in new_fns_dict.items():
                    max_accuracy_new_fns_states[new_fn_name] = self._clone_hidden_state(
                        new_fn.state_dict())

            epoch += 1
        print('sota_accuracy_found_during_training:', sota_acc)
        # TODO: need to finetune here
        # TODO: need to finetune here
        # TODO: need to finetune here
        # set the state_dictionaries of the new functions to the model with best validation accuracy
        # for new_fn_name, new_fn in new_fns_dict.items():
        #     new_fn.load_state_dict(max_accuracy_new_fns_states[new_fn_name])

        # set all new functions to eval mode
        for key, value in new_fns_dict.items():
            value.eval()

        num_evaluations = accuracies_val.__len__()
        evaluations_np = np.ones((num_evaluations, 3), dtype=np.float32)
        evaluations_np[:, 0] = iterations
        evaluations_np[:, 1] = accuracies_val
        evaluations_np[:, 2] = accuracies_test

        var_list = list(range(12))
        var_list.remove(self.data_dict['target'])
        var_np = np.array(var_list)
        self.data_dict['reject'] = var_np[list(reject_var)]
        self.data_dict['accept'] = var_np[list(accept_var)]
        par_set = set(self.data_dict['truth'])
        cnd_set = set(self.data_dict['accept'].tolist())
        self.data_dict['jacob'] = len(par_set.intersection(cnd_set)) / \
            len(par_set.union(cnd_set))
        self.data_dict['fwer'] = not cnd_set.issubset(par_set)
        self.data_dict['likelihood'] = parm_do[0][0].detach().cpu().numpy()

        return new_fns_dict, max_accuracy, evaluations_np

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
        probs = list()
        repeat = self.data_dict['repeat']
        is_lganm = self.data_dict['dict_name'].lower() == 'lganm'
        for _ in range(repeat):
            evaluations_np = np.ones((1, 1))
            if unknown_fns_def is not None and unknown_fns_def.__len__() > 0:
                if not dbg_learn_parameters:
                    new_fns_dict, _ = self.create_nns(unknown_fns_def)
                else:
                    new_fns_dict, val_acc, evaluations_np = self.learn_neural_network(program,
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
            val_accuracy.append(np.mean(val_acc[0]))
            test_accuracy.append(np.var(test_acc[0]))
            if output_type == ProgramOutputType.HAZARD:
                cox_grads.append(val_acc[1])
                cox_scores.append(val_acc[2])

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

        self.data_dict['grads'] = val_acc[1]
        print('validation accuracy: {}'.format(val_accuracy))
        print('test accuracy: {}'.format(test_accuracy))
        return {'accuracy': val_accuracy, 'new_fns_dict': new_fns_dict,
                'test_accuracy': test_accuracy, 'evaluations_np': evaluations_np}

    # program=st, output_type=output_type, unkSortMap=unkSortMap, io_examples=self.ioExamples
    def evaluate(self,
                 program,
                 output_type_s,
                 unkSortMap=None,
                 io_examples_tr=None,
                 io_examples_val=None,
                 io_examples_test=None,
                 dbg_learn_parameters=True) -> dict:

        is_graph = type(output_type_s) == AST.PPGraphSort

        program_str = ReprUtils.repr_py(program)
        output_type = self.get_program_output_type(io_examples_val,
                                                   output_type_s)

        unknown_fns_def = self.get_unknown_fns_definitions(unkSortMap,
                                                           is_graph)

        res = self.evaluate_(program=program_str,
                             output_type=output_type,
                             unknown_fns_def=unknown_fns_def,
                             io_examples_tr=io_examples_tr,
                             io_examples_val=io_examples_val,
                             io_examples_test=io_examples_test,
                             dbg_learn_parameters=dbg_learn_parameters)
        return res

    def get_program_output_type(self,
                                io_examples_val,
                                output_sort):

        if self.data_dict['out_type'] == 'hazard':
            output_type = ProgramOutputType.HAZARD
        elif self.data_dict['out_type'] == 'integer':
            output_type = ProgramOutputType.INTEGER
        else:
            raise TypeError('invalid output type {}'.format(
                self.data_dict['out_type']))
        return output_type

    def _wass(self, res, cur_res):
        # bat_size = res.shape[0] // self.data_dict['env_num']
        wass_dis = list()
        # mean_list = list()
        cur_res = np.array(cur_res)
        cur_mean = np.mean(cur_res)
        cur_std = np.var(cur_res)
        for env in range(self.data_dict['env_num']):
            res_env = res[env]
            # cur_env = cur_res[env]
            wdist = (np.mean(res_env) - cur_mean) ** 2 + \
                (np.std(res_env) - cur_std) ** 2
            # wdist = (np.mean(res_env) - np.mean(cur_env)) ** 2 + \
            #     (np.sqrt(np.var(res_env)) - np.sqrt(np.var(cur_env))) ** 2
            wass_dis.append(wdist)
        # print(max(wass_dis))
        return max(wass_dis), np.mean(np.array(res))

    def _conf_test(self, res, lab=None, rand_id=None, name='org'):
        bat_size = res.shape[0] // self.data_dict['env_num']
        p_vals_f, p_vals_t = list(), list()
        for env in range(self.data_dict['env_num']):
            msk = np.ones(res.shape[0], dtype=bool)
            msk[env * bat_size: (env + 1) * bat_size] = False
            x = res[msk].squeeze().cpu().numpy()
            y = res[~msk].squeeze().cpu().numpy()
            # lab_msk = lab[~msk]
            # if not torch.all(lab_msk == lab_msk[0, 0]) and lab is not None:
            #     print('false env', lab_msk)
            # if rand_id in (1, 4, 7, 3):
            # print(name, np.mean(x), np.mean(y), np.var(x), np.var(y))
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            F = np.var(x, ddof=1) / np.var(y, ddof=1)
            p = stats.f.cdf(F, len(x)-1, len(y)-1)
            p_vals_f.append(2*min(p, 1-p))
            p_vals_t.append(stats.ttest_ind(x, y, equal_var=False).pvalue)
            # break
        p_value_f = min(p_vals_f) * self.data_dict['env_num']
        p_value_t = min(p_vals_t) * self.data_dict['env_num']
        print(p_value_f, p_value_t)
        # res_split = torch.split(res, bat_size, dim=0)

        # if rand_id is None:
        #     res_stack = torch.stack(res_split, dim=0)
        #     res_mean = torch.mean(res_stack, dim=1)
        #     # res_var = torch.var(res_stack, dim=1)
        #     min_id = torch.argmin(res_mean).detach()
        #     max_id = torch.argmax(res_mean).detach()
        #     rand_id = min_id

        # rand_id = 0
        # x = [res for res_id, res in enumerate(res_split) if res_id != rand_id]
        # y = [res for res_id, res in enumerate(res_split) if res_id == rand_id]
        # # print(type(x), type(y))
        # if type(x) == list:
        #     x = torch.cat(x, dim=0)
        # x = x.squeeze().detach().cpu().numpy()

        # if type(y) == list:
        #     y = torch.cat(y, dim=0)
        # y = y.squeeze().detach().cpu().numpy()

        # print(x.shape, y.shape, rand_id)
        return 2 * min(p_value_f, p_value_t)

    def _get_metric_msk(self,
                        inp,
                        lab,
                        criterion,
                        program,
                        global_vars,
                        parm_do,
                        reject_var,
                        accept_var):

        with torch.no_grad():
            parm = parm_do.squeeze().detach().clone()
            parm = torch.sigmoid(parm)
            parm_id = torch.argsort(parm)
            parm_id = parm_id.cpu().numpy().tolist()
            exclude_id = None
            for pid in parm_id:
                if not (pid in reject_var or pid in accept_var):
                    # print(pid)
                    exclude_id = pid
                    break

            # parm = parm.detach().cpu().numpy()
            # parm_id = random.choices(list(range(parm.shape[-1])),
            #                          k=parm.shape[-1],
            #                          weights=parm)
            # print(parm_id)
            # parm_soft = torch.softmax(parm_sig, dim=0).detach().clone()
            # parm_id = torch.argsort(parm)

            # print(parm, parm_id)
            # parm_id = list(range(parm.shape[-1]))
            # random.shuffle(parm_id)
            # exclude_id = None
            # for pid in parm_id:
            #     if pid not in reject_var:
            #         # print(pid)
            #         exclude_id = pid
            #         break

            # if exclude_id is None:
            #     return None, None

            # msk = torch.ones(inp.shape[-1], requires_grad=False)
            # if torch.cuda.is_available():
            #     msk = msk.cuda()
            # msk[exclude_id] = 0
            # lab_pred = eval(program, global_vars)(inp * msk)[0]
            # loss = criterion(lab_pred, lab)
            # metric = loss.detach().clone()

        return 0, exclude_id

        # rand_id = random.randrange(
        #     self.data_dict['env_num'])
        # p_org = self._conf_test(metric_org)
        # p_msk = self._conf_test(metric_msk)

    def get_unknown_fns_definitions(self, unkSortMap, is_graph=False):
        # TODO: double-check. may need re-writing.
        unk_fns_interpreter_def_list = []

        for unk_fn_name, unk_fn in unkSortMap.items():

            # ******** Process output activation ***************
            fn_input_sort = unk_fn.args[0]
            fn_output_sort = unk_fn.rtpe
            output_dim = fn_output_sort.shape[1].value
            output_type = fn_output_sort.param_sort
            print(fn_input_sort, fn_output_sort, type(output_type))

            if type(fn_input_sort) == AST.PPTensorSort and fn_input_sort.shape.__len__() == 2:
                input_dim = fn_input_sort.shape[1].value
                if input_dim == output_dim:
                    uf = {'type': 'DO',
                          'name': unk_fn_name,
                          'input_dim': input_dim,
                          'dt_name': self.data_dict['dict_name']}
                else:
                    uf = {'type': 'MLP',
                          'name': unk_fn_name,
                          'input_dim': input_dim,
                          'output_dim': output_dim,
                          'bias': False if self.data_dict['out_type'] == 'hazard' else True}
                unk_fns_interpreter_def_list.append(uf)
            else:
                raise NotImplementedError()
        return unk_fns_interpreter_def_list

    def _temp(self, parm_do, parm_non_do):
        with torch.no_grad():
            tmp = parm_do[0].detach().clone()
            tmp[:] = 0
            parm_do[0].copy_(tmp)
            tmp1 = parm_non_do[0].detach().clone()
            tmp1[:] = 1
            parm_non_do[0].copy_(tmp)
