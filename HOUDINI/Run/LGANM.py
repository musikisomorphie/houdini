import argparse
import sys
import random
import pickle
import sempler
import json
import pathlib
import numpy as np
import networkx as nx
from collections import namedtuple
from enum import Enum
# from pathlib import Path
from typing import Callable, Optional, Dict, Tuple, Union, List

from HOUDINI.Config import config
from HOUDINI.Run.Task import Task
from HOUDINI.Run.Task import TaskSettings
from HOUDINI.Run.TaskSeq import TaskSeqSettings, TaskSeq
from HOUDINI.Run.Utils import get_lganm_io_examples
from HOUDINI.Library.OpLibrary import OpLibrary
from HOUDINI.Library.FnLibrary import FnLibrary
from HOUDINI.Synthesizer.AST import mkFuncSort, mkRealTensorSort, mkListSort, mkBoolTensorSort
sys.path.append('.')

SeqTaskInfo = namedtuple(
    'SeqTaskInfo', ['task_type', 'dataset'])


def print_seq(seq: List[SeqTaskInfo]):
    """print the sequence info

    Args:
        seq: the list of task info sequence
    """
    str_list = []
    for task_info in seq:
        if task_info.task_type == TaskType.Identify:
            c_str = 'idef'
        else:
            raise NotImplementedError()

        c_str += 'L'
        c_str += str(task_info.index)
        str_list.append(c_str)
    print('[ ' + ', '.join(str_list) + ' ]')


def get_seq_from_string(seq_str: str) -> List[SeqTaskInfo]:
    """Get the list of sequence task info
    based on the sequence name

    Args:
        seq_str: the squence name

    Returns:
        the list of sequence task info
    """
    seq = []
    str_list = seq_str.split(', ')
    for c_str in str_list:
        if c_str[:4] == 'idef':
            c_task_type = TaskType.Identify
        else:
            raise NotImplementedError()

        c_dataset = Dataset.LGANM
        seq.append(SeqTaskInfo(task_type=c_task_type,
                               dataset=c_dataset))

    return seq


def get_task_settings(data_dict: Dict,
                      dbg_learn_parameters: bool,
                      synthesizer: str,
                      confounder: List[int]) -> TaskSettings:
    """Get the TaskSettings namedtuple, which
    stores important learning parmeters

    Args:
        data_dict: the dict storing 
        dbg_learn_parameters: True if learn weights False otherwise
        synthesizer: enumerative or evolutionary

    Returns:
        the Tasksettings namedtuple 
    """

    if len(confounder) == 0:
        lambda_1 = 9
        lambda_2 = 0.08
    elif len(confounder) == 1:
        lambda_1 = 18
        lambda_2 = 0.09
    elif len(confounder) == 2:
        lambda_1 = 9
        lambda_2 = 0.16
    else:
        raise NotImplementedError('the coeff is not implemented '
                                  'for {} confounder(s)'.format(len(confounder)))

    task_settings = TaskSettings(
        train_size=64,
        val_size=64,
        training_percentages=[100],
        N=5000,
        M=1,
        K=1,
        synthesizer=synthesizer,
        dbg_learn_parameters=dbg_learn_parameters,
        learning_rate=0.02,
        var_num=data_dict['envs'][0].shape[1] - 1 - len(confounder),
        warm_up=8,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_cau=10.,
        data_dict=data_dict)
    return task_settings


def get_seq_info(seq_string: str) -> Dict:
    """Get sequence info

    Args:
        seq_string: the key name of the sequence

    Returns:
        the sequence info dict
    """
    seq_dict = {'idef': {'sequences': ['idefL'],
                         'prefixes': ['idef'],
                         'num_tasks': 1}}

    if not seq_string in seq_dict.keys():
        raise NameError('the seq {} are not valid'.format(seq_string))
    return seq_dict[seq_string]


class TaskType(Enum):
    Identify = 1


class Dataset(Enum):
    LGANM = 1


class IdentifyTask(Task):
    def __init__(self,
                 lganm_dict: Dict,
                 settings: TaskSettings,
                 seq: TaskSeq,
                 dbg_learn_parameters: bool,
                 confounder: List[int]):
        """The class of the causal(parent) variable
        identification task

        Args:
            lganm_dict: dict storing lganm data info
            settings: the namedtuple storing important 
                learning parameters
            seq: the task squence class
            dbg_learn_parameters: True if learn weights 
                False otherwise
            confounder: the list of confounders
        """

        # self.parent = lganm_dict['truth']
        self.confounder = confounder
        self.outcome = lganm_dict['target']
        self.envs = lganm_dict['envs']
        self.dt_dim = self.envs[0].shape[1]
        input_type = mkListSort(mkRealTensorSort(
            [1, self.dt_dim - 1 - len(confounder)]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def get_io_examples(self):
        return get_lganm_io_examples(self.envs,
                                     self.confounder,
                                     self.outcome,
                                     self.dt_dim)

    def name(self):
        return 'idef_Vars'

    def sname(self):
        return 'idef'


class InferSequence(TaskSeq):
    def __init__(self,
                 lganm_dict: Dict,
                 name: str,
                 list_task_info: List[SeqTaskInfo],
                 dbg_learn_parameters: bool,
                 seq_settings: TaskSeqSettings,
                 task_settings: TaskSettings,
                 lib: OpLibrary,
                 confounder: List[int]):
        """The class of the sequence task

        Args:
            lganm_dict: dict storing lganm data info
            name: task name
            list_task_info: list of the SeqTaskInfo namedtuple
            seq: the task squence class
            dbg_learn_parameters: True if learn weights 
                False otherwise
            seq_settings: the settings for the learning sequence
            task_settings: the task settings storing important 
                learning parameters
            lib: the library of higher order functions (Op)
            confounder: the list of confounders
        """

        self._name = name
        tasks = []
        for _, task_info in enumerate(list_task_info):
            if task_info.task_type == TaskType.Identify:
                tasks.append(IdentifyTask(lganm_dict,
                                          task_settings,
                                          self,
                                          dbg_learn_parameters,
                                          confounder))

        super().__init__(tasks,
                         seq_settings,
                         lib)

    def name(self):
        return self._name

    def sname(self):
        return self._name


def main(lganm_dict: Dict,
         task_id: int,
         seq_str: str,
         seq_name: str,
         synthesizer: str,
         confounder: List[int]):
    """The main function that runs the lganm experiments

    Args:
        lganm_dict: dict storing lganm data info
        task_id: the id of the task
        list_task_info: list of the SeqTaskInfo namedtuple
        seq_str: the prefix name that retrieves the seq info dict
        seq_name: name of the sequence
        synthesizer: enumerative or evolutionary
        confounder: the list of confounders
    """

    seq_settings = TaskSeqSettings(update_library=True,
                                   results_dir=settings['results_dir'])

    task_settings = get_task_settings(lganm_dict,
                                      settings['dbg_learn_parameters'],
                                      synthesizer,
                                      confounder)

    lib = OpLibrary(['do', 'compose', 'map',
                     'repeat', 'cat'])

    seq_tasks_info = get_seq_from_string(seq_str)
    print_seq(seq_tasks_info)

    seq = InferSequence(lganm_dict,
                        seq_name,
                        seq_tasks_info,
                        settings['dbg_learn_parameters'],
                        seq_settings,
                        task_settings,
                        lib,
                        confounder)
    seq.run(task_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--synthesizer',
                        choices=['enumerative', 'evolutionary'],
                        default='enumerative',
                        help='Synthesizer type. (default: %(default)s)')
    parser.add_argument('--lganm-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/LGANM/',
                        metavar='DIR',
                        help='path to the visualization folder')
    parser.add_argument('--exp',
                        choices=['fin', 'abcd'],
                        default='fin',
                        help='Experimental settings defined in AICP. (default: %(default)s)')
    parser.add_argument('--confounder',
                        type=int,
                        choices=[0, 1, 2],
                        default=0,
                        help='The number of hidden confounders. (default: %(default)s)')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='num of repeated experiments')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # python -m HOUDINI.Run.LGANM --repeat 1 --exp fin
    args = parse_args()

    settings = {'results_dir': str(args.lganm_dir / 'Results'),
                # If False, the interpreter doesn't learn the new parameters
                'dbg_learn_parameters': True,
                # enumerative, evolutionary
                'synthesizer': args.synthesizer,
                'seq_string': 'idef'}
    settings['results_dir'] += args.exp
    pathlib.Path(settings['results_dir']).mkdir(
        parents=True, exist_ok=True)

    res_dict = {'id': list(),
                'reject': list(),
                'accept': list(),
                'target': list(),
                'truth': list(),
                'grads': list(),
                'likelihood': list(),
                'jacob': list(),
                'fwer': list(),
                'error': list()}

    seq_info_dict = get_seq_info(settings['seq_string'])
    additional_prefix = '_np_{}'.format(
        'td' if settings['synthesizer'] == 'enumerative' else 'ea')
    prefixes = ['{}{}'.format(prefix, additional_prefix)
                for prefix in seq_info_dict['prefixes']]

    mean_jacob = list()
    mean_fwer = list()
    wrong_list = list()

    pkl_dir = args.lganm_dir / args.exp / 'n_1000'
    for pkl_id, pkl_file in enumerate(pkl_dir.glob('*.pickle')):
        if pkl_id > 600:
            continue
    # for pkl_id in ['185', '239', '53', '99', '96', '55', '225', '97', '98', '285', '284', '209', '258', '52', '238', '232', '236', '248', '54', '152', '75', '237', '259', '249', '195', '108']:
        # pkl_file = args.lganm_dir / args.exp / \
            # 'n_1000' / '{}.pickle'.format(pkl_id)
        with open(str(pkl_file), 'rb') as pl:
            lganm_dict = pickle.load(pl)
            if len(lganm_dict['truth']) - args.confounder == 0:
                print('This experiment does not have parents, thus ignore.')
                continue
            lganm_dict['truth'] = list(lganm_dict['truth'])
            print('\n\n all the parents: {}, outcome: {}'.format(
                lganm_dict['truth'], lganm_dict['target']))
            lganm_dict['confounder'] = lganm_dict['truth'][:args.confounder]
            lganm_dict['truth'] = lganm_dict['truth'][args.confounder:]
            print('remaining parents: {}, confounder: {} \n\n'.format(
                lganm_dict['truth'], lganm_dict['confounder']))
            lganm_parm = {'dict_name': 'lganm',
                          'repeat': args.repeat,
                          'mid_size': lganm_dict['envs'][0].shape[1],
                          'out_type': 'integer',
                          'env_num': len(lganm_dict['envs'])}
            lganm_dict.update(lganm_parm)

        for sequence_idx, sequence in enumerate(seq_info_dict['sequences']):
            for task_id in range(seq_info_dict['num_tasks']):
                main(lganm_dict,
                     task_id,
                     sequence,
                     prefixes[sequence_idx],
                     settings['synthesizer'],
                     lganm_dict['confounder'])
                mean_jacob.append(lganm_dict['jacob'])
                mean_fwer.append(lganm_dict['fwer'])
                if lganm_dict['jacob'] != 1:
                    res_dict['error'].append(pkl_file.stem)

                for key in res_dict:
                    if key not in ('error', 'id'):
                        res_dict[key].append(lganm_dict[key])
                res_dict['id'].append(pkl_file.stem)
                print(res_dict['accept'][-1],
                      res_dict['truth'][-1], res_dict['target'][-1])
        print('the jacarrd accuracy: ', sum(mean_jacob) / len(mean_jacob))
        print('the jacarrd accuracy: ', sum(mean_jacob) / len(mean_jacob))
        print('the fwer accuracy: ', sum(mean_fwer) / len(mean_fwer))
    pkl_nm = pathlib.Path(settings['results_dir']) / 'n_1000_res.pickle'
    with open(str(pkl_nm), 'wb') as pl:
        pickle.dump(res_dict, pl)

    print(sum(mean_fwer) / len(mean_fwer))
    print(sum(mean_jacob) / len(mean_jacob))
    print(res_dict['error'])
