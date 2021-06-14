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

from src import utils as icp_utils
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
        lambda_1=5,
        lambda_2=0.08,
        lambda_cau=10.,
        data_dict=data_dict)
    return task_settings


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

    lib = OpLibrary(['do', 'compose',
                     'repeat', 'cat', 'conv'])

    seq_tasks_info = get_seq_from_string(seq_str)

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
                        metavar='DIR')
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

    settings = {'results_dir': str(args.lganm_dir / 'Results' / args.exp / 'proposed'),
                # If False, the interpreter doesn't learn the new parameters
                'dbg_learn_parameters': True,
                # enumerative, evolutionary
                'synthesizer': args.synthesizer,
                'seq_string': 'idef'}
    pathlib.Path(settings['results_dir']).mkdir(
        parents=True, exist_ok=True)

    seq_info_dict = get_seq_info(settings['seq_string'])
    additional_prefix = '_np_{}'.format(
        'td' if settings['synthesizer'] == 'enumerative' else 'ea')
    prefixes = ['{}{}'.format(prefix, additional_prefix)
                for prefix in seq_info_dict['prefixes']]

    json_out = dict()
    jacads, fwers, errors = list(), list(), list()
    pkl_dir = args.lganm_dir / args.exp / 'n_1000'
    for pkl_id, pkl_file in enumerate(pkl_dir.glob('*.pickle')):
        if pkl_id > 400:
            continue
        with open(str(pkl_file), 'rb') as pl:
            lganm_dict = pickle.load(pl)
            lganm_dict['truth'] = list(lganm_dict['truth'])
            dag_wei = lganm_dict['case'].sem.W
            assert (dag_wei.shape[1] == dag_wei.shape[0])
            assert np.all(np.asarray(lganm_dict['truth']) < dag_wei.shape[0])
            assert lganm_dict['target'] < dag_wei.shape[1]
            print(dag_wei.shape)
            print('all the parents: {}, outcome: {}'.format(
                lganm_dict['truth'], lganm_dict['target']))
            lganm_dict['confounder'] = list()
            print('remaining parents: {}, confounder: {} \n\n'.format(
                lganm_dict['truth'], lganm_dict['confounder']))
            lganm_parm = {'dict_name': 'lganm',
                          'repeat': args.repeat,
                          'mid_size': lganm_dict['envs'][0].shape[1],
                          'out_type': 'mse',
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
                res_dict = lganm_dict['json_out']
                for prog_str, prog_dict in res_dict.items():
                    if prog_str not in json_out:
                        json_out[prog_str] = {'jacads': list(),
                                              'fwers': list(),
                                              'errors': list(),
                                              'rejects': list(),
                                              'accepts': list()}
                    json_out[prog_str]['jacads'].extend(prog_dict['jacads'])
                    json_out[prog_str]['fwers'].extend(prog_dict['fwers'])
                    # print(json_out[prog_str]['jacads'])
                    # json_out[prog_str]['rejects'].append(prog_dict['rej_vars'])
                    # json_out[prog_str]['accepts'].append(prog_dict['acc_vars'])
                    # jacads.extend(lganm_dict['json_out']['jacads'])
                    # fwers.extend(lganm_dict['json_out']['fwers'])
                    # json_out[prog_str][pkl_file.stem] = lganm_dict[prog_str]['json_out']
                    if not np.all(np.asarray(prog_dict['jacads']) == 1.):
                        json_out[prog_str]['errors'].append(pkl_file.stem)
                    # print('\nprogram: {}'.format(prog_str))
                    # print('Jaccard Similarity (JS): {}.'.format(
                    #     sum(json_out[prog_str]['jacads']) / len(json_out[prog_str]['jacads'])))
                    # print('Family-wise error rate (FWER): {}'.format(
                    #     sum(json_out[prog_str]['fwers']) / len(json_out[prog_str]['fwers'])))
                    # print('errors: {}'.format(json_out[prog_str]['errors']))
    for prog_str in json_out:
        # print(jacads)
        jacads = np.asarray(json_out[prog_str]['jacads'])
        fwers = np.asarray(json_out[prog_str]['fwers'])
        json_out[prog_str]['jacads_mean'] = np.mean(jacads, axis=0).tolist()
        json_out[prog_str]['jacads_std'] = np.std(jacads, axis=0).tolist()
        json_out[prog_str]['fwers_mean'] = np.mean(fwers, axis=0).tolist()
        json_out[prog_str]['fwers_std'] = np.std(fwers, axis=0).tolist()
        print(jacads.shape, fwers.shape)
        print('\nprogram: {}'.format(prog_str))
        print('Jaccard Similarity (JS) mean: {}, std: {}.'.format(
            np.mean(jacads, axis=0), np.std(jacads, axis=0)))
        print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
            np.mean(fwers, axis=0), np.std(fwers, axis=0)))

    json_file = pathlib.Path(settings['results_dir']) / 'lganm_table.json'
    with open(str(json_file), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=4)
