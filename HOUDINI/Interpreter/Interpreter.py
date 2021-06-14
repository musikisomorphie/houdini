import sys
import math
import random
import pathlib
import numpy as np
import copy
from enum import Enum
from functools import reduce
from collections import OrderedDict
from numpy.compat.py3k import is_pathlib_path
from scipy import stats
from typing import List, Dict, Optional, Tuple, Union, Iterator
import torch
import torch.nn.functional as F
from torch import autograd, nn

from Data.DataGenerator import NumpyDataSetIterator
from HOUDINI.Library import Loss, Metric
from HOUDINI.Library.Utils import MetricUtils
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ReprUtils
from HOUDINI.Library import NN
from HOUDINI.Library.Utils import NNUtils
from HOUDINI.Library.OpLibrary import OpLibrary


class ProgramOutputType(Enum):
    MSE = 1,  # using l2 regression, evaluating using round up/down
    SIGMOID = 2,
    SOFTMAX = 3,
    HAZARD = 4


class Interpreter:
    """The core neural network learning algorithm 
    This class is called after obtaining multiple type-safe function candidates

    Attributes:
        settings: the TaskSettings class storing the training parameters
                  and the dataset info. 
        library: the OpLibrary class initializing the higher order functions.
        evaluate_every_n_percent: deprecated.
    """

    def __init__(self,
                 settings,
                 library: OpLibrary,
                 evaluate_every_n_percent: float = 10.):

        self.library = library
        self.data_dict = settings.data_dict
        self.settings = settings

        if self.data_dict['out_type'] == 'hazard':
            self.output_type = ProgramOutputType.HAZARD
            self.criterion = Loss.cox_ph_loss
            self.metric = Loss.cox_ph_loss
        elif self.data_dict['out_type'] == 'mse':
            self.output_type = ProgramOutputType.MSE
            self.criterion = nn.MSELoss(reduction='none')
            self.metric = F.mse_loss
        else:
            raise TypeError('invalid output type {}'.format(
                self.data_dict['out_type']))

    def _create_fns(self, unknown_fns: List[Dict]) -> Tuple[Dict, Dict]:
        """ Instantiate the higher-oder functions, 
        unknown functions with nn, obtain the trainable parameters 
        for direct access during training.

        Args:
            unknow_fns: the list of the unknown functions

        Returns:
            the dict of instantiated nn functions and 
                higher-order functions
            the dict of their trainable parameters
        """
        trainable_parameters = {'do': list(),
                                'non-do': list()}
        prog_fns_dict = dict()
        prog_fns_dict['lib'] = self.library
        for uf in unknown_fns:
            fns_nn = NNUtils.get_nn_from_params_dict(uf)
            prog_fns_dict[uf["name"]] = fns_nn[0]
            c_trainable_params = fns_nn[1]
            if "freeze" in uf and uf["freeze"]:
                print("freezing the weight of {}".format(uf["name"]))
                continue
            if uf['type'] == 'DO':
                trainable_parameters['do'] += list(c_trainable_params)
            else:
                trainable_parameters['non-do'] += list(c_trainable_params)
        return prog_fns_dict, trainable_parameters

    def _get_data_loader(self,
                         io_examples: Tuple,
                         batch: int) -> NumpyDataSetIterator:
        """ Wrap the data with the NumpyDataSetIterator class

        Args:
            io_examples: the tuple of numpy input data (data, label)

        Returns:
            the NumpyDataSetIterator of the input data
        """

        if isinstance(io_examples, tuple) and \
           len(io_examples) == 2:
            output = NumpyDataSetIterator(io_examples[0],
                                          io_examples[1],
                                          batch)
        else:
            raise NotImplementedError('The function that processes '
                                      'the data type {} is not implemented'.format(type(io_examples)))
        return output

    def _clone_weights_state(self, src_dict: Dict, tar_dict: Dict):
        """ Deep copy the state_dict of the learnable weights
        from the source dict to target dict

        Args:
            src_dict: source dictionary
            tar_dict: target dictionary

        Returns:
        """

        for new_fn_name, new_fn in src_dict.items():
            if issubclass(type(new_fn), nn.Module):
                new_state = OrderedDict()
                for key, val in new_fn.state_dict().items():
                    new_state[key] = val.clone()
                tar_dict[new_fn_name] = new_state

    def _set_weights_mode(self, fns_dict: Dict, is_trn: bool):
        """ Switch the weights mode between train and eval

        Args:
            fns_dict: nn function dictionary storing the learnable weights
            is_trn: is train or not 

        Returns:

        """
        for _, fns in fns_dict.items():
            if issubclass(type(fns), nn.Module):
                fns.train() if is_trn else fns.eval()

    def _compute_grad(self,
                      inputs: torch.Tensor,
                      program: str,
                      prog_fns_dict: Dict) -> torch.Tensor:
        """ Compute the gradients of the nn w.r.t the inputs,
        the nn is instantiated by the program string and its 
        corresponding function stored in global vars.

        Args:
            inputs: the input data 
            program: the program string, for example:
                'lib.compose(nn_fun_idef_np_tdidef_58, 
                             lib.cat(lib.do(nn_fun_idef_np_tdidef_59)))'
            prog_fns_dict: the dict with the key of the function in the 
                program string, and the value of the function implmentation

        Returns:
            the gradients of the program w.r.t the inputs
        """

        inputs = autograd.Variable(inputs, requires_grad=True)
        outputs = eval(program, prog_fns_dict)(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        grad_outputs = torch.ones(outputs.shape)
        if torch.cuda.is_available():
            grad_outputs = grad_outputs.cuda()
        grads = autograd.grad(outputs=outputs,
                              inputs=inputs,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
        return grads.detach()

    def _update_sota(self,
                     sota_acc: np.ndarray,
                     sota_grad: np.ndarray,
                     sota_fns_dict: Dict,
                     prog_fns_dict: Dict,
                     val_mse: np.ndarray,
                     val_grad: Optional[np.ndarray] = None,
                     parm_do: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """ update the sota metric by comparing with the metric of current epoch.

        Args:
            sota_acc: the sota accuracy 
            sota_grad: the sota gradients of the program
            sota_fns_dict: the dict storing the sota state of the learnable weights
            prog_fns_dict: the dict storing the current state of the learnable weights
            val_mse: the validation mse (usually a n-dim numpy array)
            val_grad: the current gradients of the program
            parm_do: the weights of the do function

        Returns:
            the updated sota metrics
        """

        sota_acc = np.mean(val_mse)
        sota_mse = val_mse
        self._clone_weights_state(prog_fns_dict,
                                  sota_fns_dict)
        if val_grad is not None:
            sota_grad = np.abs(val_grad) * parm_do.cpu().numpy()
        return sota_acc, sota_mse, sota_grad, sota_fns_dict

    def _predict_data(self,
                      data_loader: NumpyDataSetIterator,
                      program: str,
                      prog_fns_dict: Dict) -> Iterator[Tuple]:
        """ The core learning step for each iteration, i.e.
        feed one batch data to the nn and compute the output.

        Args:
            data_loader: the data iterator that can generate a batch of data 
            program: the program string, for example:
                'lib.compose(nn_fun_idef_np_tdidef_58, 
                             lib.cat(lib.do(nn_fun_idef_np_tdidef_59)))'
            prog_fns_dict: the dict with the key of the function in the 
                program string, and the value of the function implmentation

        Returns:
            the tuple of torch tensors: (prediction, groud-truth, input)
        """

        # list of data_loader_iterators
        if isinstance(data_loader, NumpyDataSetIterator):
            data_loader_list = [data_loader]

        # creating a shallow copy of the list of iterators
        dl_iters_list = list(data_loader_list)
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
                x = torch.from_numpy(x)
                x = autograd.Variable(x)

                y = torch.from_numpy(y)
                y = torch.cat(torch.split(y, 1, dim=1), dim=0).squeeze(dim=1)
                y = autograd.Variable(y)

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                y_pred = eval(program, prog_fns_dict)(x.float())
                yield (y_pred, y.float(), x.float())

    def _train_data(self,
                    data_loader: NumpyDataSetIterator,
                    program: str,
                    prog_fns_dict: Dict,
                    optim: torch.optim):
        """ Train the data for one epoch 

        Args:
            data_loader: the data iterator that can generate a batch of data 
            program: the program string, for example:
                'lib.compose(nn_fun_idef_np_tdidef_58, 
                             lib.cat(lib.do(nn_fun_idef_np_tdidef_59)))'
            prog_fns_dict: the dict with the key of the function in the 
                program string, and the value of the function implmentation
            criterion: the loss function
            optim: pytorch optimizer

        Returns:
        """

        for y_pred, y, x_in in self._predict_data(data_loader, program, prog_fns_dict):
            if type(y_pred) == tuple:
                y_pred = y_pred[0]
            loss = self.criterion(y_pred, y)
            optim.zero_grad()
            (loss.mean()).backward(retain_graph=True)
            optim.step()

    def _get_accuracy(self,
                      data_loader: NumpyDataSetIterator,
                      program: str,
                      prog_fns_dict: Dict,
                      compute_grad: bool = False) -> Tuple:
        """ Compute the relevant metric for trained model

        Args:
            data_loader: the data iterator that can generate a batch of data
            program: the program string, for example:
                'lib.compose(nn_fun_idef_np_tdidef_58,
                             lib.cat(lib.do(nn_fun_idef_np_tdidef_59)))'
            prog_fns_dict: the dict with the key of the function in the
                program string, and the value of the function implmentation
            compute_grad: whether compute (mean) gradients or not, False by default 

        Returns:
            the tuple of numpy arrays: (mse, gradients, other scores),
                the other scores can be cox scores if portec, None else.
        """

        y_debug_all, mse_all = list(), list()
        x_all, y_all, y_pred_all = list(), list(), list()
        for y_pred, y, x_in in self._predict_data(data_loader, program, prog_fns_dict):
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            x_all.append(x_in)
            y_all.append(y)
            y_pred_all.append(y_pred)

            mse = self.metric(y_pred,
                              y,
                              reduction='none')
            mse = mse.detach().cpu().numpy()
            # np.split is different to torch.split
            mse = np.split(mse, x_in.shape[1], axis=0)
            mse_all.append(mse)

            y = y.detach().cpu().numpy()
            y = np.split(y, x_in.shape[1], axis=0)
            y_debug_all.append(y)

        grad_out, score_out = None, None
        if self.output_type == ProgramOutputType.HAZARD:
            y_all = torch.cat(y_all, dim=0)
            y_all_np = y_all.detach().cpu().numpy()
            y_pred_all = torch.cat(y_pred_all, dim=0)
            y_pred_all_np = y_pred_all.detach().cpu().numpy()
            cox_metric = Metric.coxph(y_pred_all_np, y_all_np)
            cox_scores = cox_metric.eval_surv(y_pred_all_np, y_all_np)

            g_in = torch.tensor(
                list(self.data_dict['clinical_meta']['causal'].values()))
            g_in = g_in.unsqueeze(dim=0).unsqueeze(dim=0).float()
            if torch.cuda.is_available():
                g_in = g_in.cuda()
            cox_grads = self._compute_grad(g_in,
                                           program,
                                           prog_fns_dict)

            cox_grads = cox_grads.cpu().numpy()
            cox_grads = np.squeeze(cox_grads)

            grad_out = cox_grads
            score_out = cox_scores
        else:
            if self.output_type == ProgramOutputType.MSE:
                if compute_grad:
                    grad_all = list()
                    for x_in in x_all:
                        x_grads = self._compute_grad(x_in.clone(),
                                                     program,
                                                     prog_fns_dict)
                        x_grads = torch.cat(torch.split(
                            x_grads, 1, dim=1), dim=0).squeeze()
                        x_grads_norm = x_grads.abs().mean(dim=0)
                        grad_all.append(x_grads_norm.detach().cpu().numpy())
                    grad_all = np.asarray(grad_all)
                    grad_out = np.mean(grad_all, axis=0)
                    grad_out = grad_out / grad_out.sum()

        mse_out = list(zip(*mse_all))
        mse_out = [np.concatenate(mse_env, axis=0) for mse_env in mse_out]

        # y_debug_all = list(zip(*y_debug_all))
        # for debug_id, debug in enumerate(y_debug_all):
        #     print(debug_id)
        #     debug = np.concatenate(debug, axis=0)
        #     assert np.all(debug == debug_id)

        return mse_out, grad_out, score_out

    def _warm_up(self,
                 program: str,
                 unknown_fns: List[Dict],
                 data_loader_trn: NumpyDataSetIterator,
                 data_loader_val: NumpyDataSetIterator,
                 data_loader_tst: NumpyDataSetIterator):

        sota_acc = sys.float_info.max
        sota_wss = sys.float_info.max
        sota_do = None
        sota_fns_dict = dict()

        for _ in range(1):
            prog_fns_dict, trainable_parameters = self._create_fns(unknown_fns)
            parm_all = trainable_parameters['do'] + \
                trainable_parameters['non-do']
            parm_do = trainable_parameters['do']
            optim_all = torch.optim.Adam(parm_all,
                                         lr=self.settings.learning_rate,
                                         weight_decay=0.001)

            for epoch in range(self.settings.warm_up):
                print('Starting warm-up epoch {}'.format(epoch))
                self._set_weights_mode(prog_fns_dict, is_trn=True)
                self._train_data(data_loader_trn,
                                 program,
                                 prog_fns_dict,
                                 optim_all)

                self._set_weights_mode(prog_fns_dict, is_trn=False)
                val_mse = self._get_accuracy(data_loader_val,
                                             program,
                                             prog_fns_dict)[0]

                wass_dis = self._wass1(val_mse)
                if wass_dis < sota_wss:  # np.mean(val_mse) < sota_acc:
                    sota_tuple = self._update_sota(sota_acc,
                                                   None,
                                                   sota_fns_dict,
                                                   prog_fns_dict,
                                                   val_mse,
                                                   None,
                                                   parm_do[0][0].detach())
                    sota_acc, _, _, sota_fns_dict = sota_tuple
                    sota_wss = wass_dis
                    sota_do = parm_do[0][0].detach().cpu().numpy()
                    print('warm_do: ', sota_do)
                    print('sigmoid: ', torch.sigmoid(
                        parm_do[0][0]).detach().cpu().numpy())
                    print('softmax: ', nn.Softmax(dim=0)
                          (parm_do[0][0]).detach().cpu().numpy())
                    print('sota_acc: {}, sota_wass: {}'.format(
                        sota_acc, sota_wss))

        if self.output_type == ProgramOutputType.HAZARD:
            # TODO#TODO#TODO#TODO#TODO
            grad_grp = list()
            caus_grp = self.data_dict['clinical_meta']['causal_grp']
            # sort the causal group based on the mean gradient weight
            for cau in caus_grp:
                grad_grp.append(np.max(sota_do[cau]))
            sota_ord_idx, _ = zip(*sorted(zip(caus_grp,
                                              grad_grp),
                                          key=lambda t: t[1]))
        elif self.output_type == ProgramOutputType.MSE:
            sota_ord_idx = np.argsort(sota_do).tolist()

        print('Warm-up phase compeleted. \n')

        sota = (sota_acc,
                sota_wss,
                sota_do,
                sota_fns_dict,
                sota_ord_idx)

        return sota

    def learn_neural_network(self,
                             program: str,
                             unknown_fns: List[Dict],
                             data_loader_trn: NumpyDataSetIterator,
                             data_loader_val: NumpyDataSetIterator,
                             data_loader_tst: NumpyDataSetIterator) -> Tuple:

        sota = self._warm_up(program,
                             unknown_fns,
                             data_loader_trn,
                             data_loader_val,
                             data_loader_tst)
        sota_acc, sota_wss, sota_do, sota_fns_dict, sota_ord_idx = sota

        prog_fns_dict, trainable_parameters = self._create_fns(unknown_fns)
        parm_all = trainable_parameters['do'] + trainable_parameters['non-do']
        parm_do = trainable_parameters['do']
        optim_all = torch.optim.Adam(parm_all,
                                     lr=self.settings.learning_rate * 0.5,
                                     weight_decay=0.001)

        for new_fn_name, new_fn in prog_fns_dict.items():
            if issubclass(type(new_fn), nn.Module):
                new_fn.load_state_dict(
                    sota_fns_dict[new_fn_name])

        self._set_weights_mode(prog_fns_dict, is_trn=False)
        _, warm_grad, warm_score = self._get_accuracy(data_loader_val,
                                                      program,
                                                      prog_fns_dict,
                                                      compute_grad=True)

        ###################################################################
        ########################Causal Training############################
        ###################################################################
        reject_var, accept_var = list(), list()
        reject_vars = list()
        accept_vars = list()
        sota_idx = None
        epochs = self.settings.warm_up // 4
        for _ in range(self.settings.var_num):
            with torch.no_grad():
                for idx in sota_ord_idx:
                    if not ((idx in accept_var) or (idx in reject_var)):
                        sota_idx = idx
                        msk = parm_do[1].detach().clone()
                        msk[0, sota_idx] = 0
                        parm_do[1].copy_(msk)
                        break
            for epoch in range(epochs):
                self._set_weights_mode(prog_fns_dict, is_trn=True)
                self._train_data(data_loader_trn,
                                 program,
                                 prog_fns_dict,
                                 optim_all)

                self._set_weights_mode(prog_fns_dict, is_trn=False)
                val_mse = self._get_accuracy(data_loader_val,
                                             program,
                                             prog_fns_dict)[0]

                wass_dis = self._wass1(val_mse)
                if wass_dis < self.settings.lambda_1 * (1 - np.max(sota_do[sota_idx])) * sota_wss:
                    reject_var.append(sota_idx)
                    self._clone_weights_state(prog_fns_dict,
                                              sota_fns_dict)
                    print(sota_idx, sota_acc, wass_dis,
                          accept_var, reject_var)
                    break

                if epoch + 1 == epochs:
                    accept_var.append(sota_idx)
                    for new_fn_name, new_fn in prog_fns_dict.items():
                        if issubclass(type(new_fn), nn.Module):
                            new_fn.load_state_dict(
                                sota_fns_dict[new_fn_name])
                    print(sota_idx, sota_acc, wass_dis,
                          accept_var, reject_var)
            accept_vars.append(copy.deepcopy(accept_var))
            reject_vars.append(copy.deepcopy(reject_var))

        self._set_weights_mode(prog_fns_dict, is_trn=True)
        self._train_data(data_loader_trn,
                         program,
                         prog_fns_dict,
                         optim_all)
        self._set_weights_mode(prog_fns_dict, is_trn=False)
        _, val_grad, val_score = self._get_accuracy(data_loader_val,
                                                    program,
                                                    prog_fns_dict,
                                                    compute_grad=True)
        val_do = parm_do[0][0].detach().cpu().numpy()
        val_msk = parm_do[1][0].detach().cpu().numpy()
        print(val_msk)
        print('Causal phase compeleted. \n')

        # collect all the output
        var_cls = (reject_vars, accept_vars)
        warm_up = (warm_grad, warm_score, 1 / (1 + np.exp(-sota_do)))
        caus_val = (val_grad, val_score, val_msk / (1 + np.exp(-val_do)))
        return prog_fns_dict, var_cls, warm_up, caus_val

    def evaluate_(self,
                  program: str,
                  unknown_fns_def: List[Dict] = None,
                  io_examples_trn=None,
                  io_examples_val=None,
                  io_examples_tst=None,
                  dbg_learn_parameters=True):
        """
        Independent from the synthesizer
        :param program:
        :param output_type:
        :param unknown_fns_def:
        :param io_examples_trn:
        :param io_examples_val:
        :param io_examples_tst:
        :param dbg_learn_parameters: if False, it's not going to learn parameters
        :return:
        """
        data_loader_trn = self._get_data_loader(io_examples_trn,
                                                self.settings.train_size)
        data_loader_val = self._get_data_loader(io_examples_val,
                                                self.settings.val_size)
        data_loader_tst = self._get_data_loader(io_examples_tst,
                                                self.settings.val_size)

        prog_fns_dict = dict()
        val_grads, val_scores, val_dos = list(), list(), list()
        warm_grads, warm_scores, warm_dos = list(), list(), list()
        jacads, fwers = list(), list()
        jacads, fwers
        for _ in range(self.data_dict['repeat']):
            if unknown_fns_def is not None and \
               unknown_fns_def.__len__() > 0:
                if not dbg_learn_parameters:
                    prog_fns_dict, _ = self._create_fns(unknown_fns_def)
                else:
                    prog_fns_dict, var_cls, warm_up, caus_val = self.learn_neural_network(program,
                                                                                          unknown_fns_def,
                                                                                          data_loader_trn,
                                                                                          data_loader_val,
                                                                                          data_loader_tst)
                    rej_vars, acc_vars = var_cls
                    warm_grad, warm_score, warm_do = warm_up
                    val_grad, val_score, val_do = caus_val

            val_grads.append(val_grad.tolist())
            val_scores.append(val_score)
            val_dos.append(val_do.tolist())
            warm_grads.append(warm_grad.tolist())
            warm_scores.append(warm_score)
            warm_dos.append(warm_do.tolist())

            accept_vars, reject_vars = list(), list()
            if self.output_type == ProgramOutputType.HAZARD:
                for acc in acc_vars:
                    accept_vars.append(set(tuple(var) for var in acc))
                causal_var = set(tuple(var) for var in self.data_dict['truth'])
            elif self.output_type == ProgramOutputType.MSE:
                var_list = list(range(self.data_dict['mid_size']))
                var_remove = [self.data_dict['target']]
                for var in sorted(var_remove, reverse=True):
                    var_list.remove(var)
                var_np = np.array(var_list)
                for acc in acc_vars:
                    accept_vars.append(set(var_np[list(acc)].tolist()))
                causal_var = set(self.data_dict['truth'])

            jacad, fwer = list(), list()
            for acc in accept_vars:
                js = (len(causal_var.intersection(acc))) / \
                    (len(causal_var.union(acc)))
                jacad.append(js)
                fwer.append(not acc.issubset(causal_var))
            jacads.append(jacad)
            fwers.append(fwer)

        json_out = {'val_grads': val_grads,
                    'val_dos': val_dos,
                    'warm_grads': warm_grads,
                    'warm_dos': warm_dos,
                    'jacads': jacads,
                    'fwers': fwers,
                    'rej_vars': reject_vars,
                    'acc_vars': accept_vars}

        if self.output_type == ProgramOutputType.HAZARD:
            cox_index = list(self.data_dict['clinical_meta']['causal'].keys())
            cox_dir = self.data_dict['results_dir']

            warm_grads = np.asarray(warm_grads)
            warm_scores = list(zip(*warm_scores))
            warm_utils = MetricUtils.coxsum(cox_index,
                                            warm_grads,
                                            file_nm='portec_warm_up_{}'.format(program))
            warm_utils.vis_plot(warm_scores,
                                pathlib.Path(cox_dir),
                                self.data_dict['metric_scores'])
            print(warm_utils.summary(pathlib.Path(cox_dir)), '\n')

            val_grads = np.asarray(val_grads)
            val_scores = list(zip(*val_scores))
            val_utils = MetricUtils.coxsum(cox_index, 
                                           val_grads,
                                           file_nm='portec_{}'.format(program))
            val_utils.vis_plot(val_scores,
                               pathlib.Path(cox_dir),
                               self.data_dict['metric_scores'])
            print(val_utils.summary(pathlib.Path(cox_dir)))

            json_out['warm_scores'] = warm_scores
            json_out['val_scores'] = val_scores

        if 'json_out' not in self.data_dict:
            self.data_dict['json_out'] = dict()
        self.data_dict['json_out'][program] = json_out
        return {'accuracy': 0., 'prog_fns_dict': prog_fns_dict,
                'test_accuracy': 0., 'evaluations_np': np.ones((1, 1))}

    def evaluate(self,
                 program,
                 output_type_s,
                 unkSortMap=None,
                 io_examples_trn=None,
                 io_examples_val=None,
                 io_examples_tst=None,
                 dbg_learn_parameters=True) -> dict:

        program_str = ReprUtils.repr_py(program)
        unknown_fns_def = self.get_unknown_fns_definitions(
            unkSortMap, program_str)

        res = self.evaluate_(program=program_str,
                             unknown_fns_def=unknown_fns_def,
                             io_examples_trn=io_examples_trn,
                             io_examples_val=io_examples_val,
                             io_examples_tst=io_examples_tst,
                             dbg_learn_parameters=dbg_learn_parameters)
        return res

    def _wass(self, res, cur_res):
        wass_dis = list()
        cur_res = np.array(cur_res)
        cur_mean = np.mean(cur_res)
        cur_std = np.std(cur_res)
        for env in range(self.data_dict['env_num']):
            res_env = res[env]
            # cur_env = cur_res[env]
            wdist = (np.mean(res_env) - cur_mean) ** 2 + \
                (np.std(res_env) - cur_std) ** 2
            # wdist = (np.mean(res_env) - np.mean(cur_env)) ** 2 + \
            #     (np.sqrt(np.var(res_env)) - np.sqrt(np.var(cur_env))) ** 2
            wass_dis.append(np.sqrt(wdist))
        # print(max(wass_dis))
        return max(wass_dis), np.mean(np.array(res))

    def get_unknown_fns_definitions(self, unkSortMap, prog_str):
        print('the string of program: {}'.format(prog_str))
        # TODO: double-check. may need re-writing.
        unk_fns_interpreter_def_list = []

        for unk_fn_name, unk_fn in unkSortMap.items():
            fn_input_sort = unk_fn.args[0]
            fn_output_sort = unk_fn.rtpe
            output_dim = fn_output_sort.shape[1].value

            nn_idx = prog_str.find(unk_fn_name)
            lib_idx = prog_str.rfind('lib.', 0, nn_idx)
            lib_nm = prog_str[lib_idx:nn_idx-1]
            print(lib_nm, unk_fn_name)

            if type(fn_input_sort) == AST.PPTensorSort and fn_input_sort.shape.__len__() == 2:
                input_dim = fn_input_sort.shape[1].value
                if lib_nm == 'lib.do':
                    assert input_dim == output_dim
                    uf = {'type': 'DO',
                          'name': unk_fn_name,
                          'input_dim': input_dim,
                          'dt_name': self.data_dict['dict_name']}
                elif lib_nm == 'lib.conv':
                    uf = {'type': 'CONV',
                          'name': unk_fn_name,
                          'input_dim': input_dim,
                          'output_dim': output_dim}
                else:
                    uf = {'type': 'MLP',
                          'name': unk_fn_name,
                          'input_dim': input_dim,
                          'output_dim': output_dim,
                          'dt_name': self.data_dict['dict_name']}
                unk_fns_interpreter_def_list.append(uf)
            else:
                raise NotImplementedError()
        return unk_fns_interpreter_def_list

    def _wass1(self, res):
        wass_dis = list()
        for env in range(self.data_dict['env_num']):
            res_env = res.pop(env)
            wdist = (np.mean(res_env) - np.mean(res)) ** 2 + \
                (np.std(res_env) - np.std(res)) ** 2
            wass_dis.append(np.sqrt(wdist))
            res.insert(env, res_env)
        return max(wass_dis)
