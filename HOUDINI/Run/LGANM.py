import argparse
import sys
import random
import pickle
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
from HOUDINI.Run.Utils import get_lganm_io_examples
from HOUDINI.Library.OpLibrary import OpLibrary
from HOUDINI.Library.FnLibrary import FnLibrary
from HOUDINI.Synthesizer.AST import mkFuncSort, mkRealTensorSort, mkListSort, mkBoolTensorSort
sys.path.append('.')

SequenceTaskInfo = namedtuple(
    'SequenceTaskInfo', ['task_type', 'dataset'])


class TaskType(Enum):
    Identify = 1


class Dataset(Enum):
    LGANM = 1


class IdentifyTask(Task):
    def __init__(self,
                 lganm_dict,
                 settings,
                 seq,
                 dbg_learn_parameters):
        self.parents = lganm_dict['truth']
        self.outcome = lganm_dict['target']
        self.envs = lganm_dict['envs']

        input_dim = self.envs[0].shape[1] - 1
        input_type = mkRealTensorSort([1, input_dim])
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def get_io_examples(self):
        return get_lganm_io_examples(self.envs,
                                     self.parents,
                                     self.outcome)

    def name(self):
        return 'idef_Vars'

    def sname(self):
        return 'idef'


def get_task_name(task_info):
    if task_info.task_type == TaskType.Identify:
        c_str = 'idef'
    else:
        raise NotImplementedError()

    c_str += 'L'
    c_str += str(task_info.index)
    return c_str


def print_sequence(sequence):
    str_list = []
    for task_info in sequence:
        str_list.append(get_task_name(task_info))

    print('[ ' + ', '.join(str_list) + ' ]')


def get_sequence_from_string(sequence_str):
    '''
    :param sequence_str: recD3, countT2, recT1, ...  (no initial/closing brackets)
    '''
    sequence = []
    str_list = sequence_str.split(', ')
    for c_str in str_list:
        if c_str[:4] == 'idef':
            c_task_type = TaskType.Identify
        else:
            raise NotImplementedError()

        c_dataset = Dataset.LGANM
        # print(c_str)
        sequence.append(SequenceTaskInfo(task_type=c_task_type,
                                         dataset=c_dataset))

    return sequence


class InferSequence(TaskSeq):
    def __init__(self,
                 lganm_dict,
                 name,
                 list_task_info,
                 dbg_learn_parameters,
                 seq_settings,
                 task_settings,
                 lib):
        self._name = name
        tasks = []
        for _, task_info in enumerate(list_task_info):
            if task_info.task_type == TaskType.Identify:
                tasks.append(IdentifyTask(lganm_dict,
                                          task_settings,
                                          self,
                                          dbg_learn_parameters))

        super().__init__(tasks,
                         seq_settings,
                         lib)

    def name(self):
        return self._name

    def sname(self):
        return self._name


def get_sequence_info(seq_string):
    seq_dict = {'idef': {'sequences': ['idefL'],
                         'prefixes': ['idef'],
                         'num_tasks': 1}}

    if not seq_string in seq_dict.keys():
        raise NameError('the seq {} are not valid'.format(seq_string))
    return seq_dict[seq_string]


def get_task_settings(data_dict,
                      dbg_mode,
                      dbg_learn_parameters,
                      synthesizer=None):
    '''
    :param dbg_mode:
    :param synthesizer_type: None, enumerative, evolutionary
    :return:
    '''
    if not dbg_mode:
        task_settings = TaskSettings(
            train_size=128,
            val_size=128,
            batch_size=128,
            training_percentages=[2, 10, 20, 50, 100],
            N=10000,
            M=50,
            K=50,
            epochs=30,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters,
            learning_rate=0.02,
            data_dict=data_dict)
    else:
        task_settings = TaskSettings(
            train_size=128,
            val_size=128,
            batch_size=128,
            training_percentages=[100],
            N=1000,
            M=8,
            K=8,
            epochs=1,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters,
            learning_rate=0.02,
            data_dict=data_dict)
    return task_settings


def mk_default_lib():
    lib = OpLibrary(['compose', 'repeat', 'map_l',
                     'fold_l', 'conv_l', 'zeros'])
    return lib


def main(lganm_dict,
         task_id,
         sequence_str,
         sequence_name,
         synthesizer):

    seq_settings = TaskSeqSettings(update_library=True,
                                   results_dir=settings['results_dir'])

    task_settings = get_task_settings(lganm_dict,
                                      settings['dbg_mode'],
                                      settings['dbg_learn_parameters'],
                                      synthesizer=synthesizer)
    lib = mk_default_lib()

    seq_tasks_info = get_sequence_from_string(sequence_str)
    print_sequence(seq_tasks_info)

    seq = InferSequence(lganm_dict,
                        sequence_name,
                        seq_tasks_info,
                        settings['dbg_learn_parameters'],
                        seq_settings,
                        task_settings,
                        lib)
    seq.run(task_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--synthesizer',
                        choices=['enumerative', 'evolutionary'],
                        default='enumerative',
                        help='Synthesizer type. (default: %(default)s)')
    parser.add_argument('--dbg',
                        action='store_true',
                        help='If set, the sequences run for a tiny amount of data')
    parser.add_argument('--lganm-dir',
                        type=Path,
                        default='/home/histopath/Data/LGANM/',
                        metavar='DIR',
                        help='path to the visualization folder')
    parser.add_argument('--exp',
                        choices=['fin', 'int_srt', 'abcd'],
                        default='fin',
                        help='Experimental settings defined in AICP. (default: %(default)s)')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='num of repeated experiments')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # python -m HOUDINI.Run.LGANM --dbg
    args = parse_args()

    settings = {
        'results_dir': 'Results',  # str(sys.argv[1])
        # If False, the interpreter doesn't learn the new parameters
        'dbg_learn_parameters': True,
        'dbg_mode': args.dbg,  # If True, the sequences run for a tiny amount of data
        'synthesizer': args.synthesizer,  # enumerative, evolutionary
        'seq_string': 'idef'  # 'ls'  # cs1, cs2, cs3, ls
    }

    seq_info_dict = get_sequence_info(settings['seq_string'])

    # num_tasks = seq_info_dict['num_tasks']
    additional_prefix = '_np_{}'.format(
        'td' if settings['synthesizer'] == 'enumerative' else 'ea')
    prefixes = ['{}{}'.format(prefix, additional_prefix)
                for prefix in seq_info_dict['prefixes']]

    pkl_file = args.lganm_dir / args.exp / 'n_1000' / '{}.pickle'.format(241)
    with open(str(pkl_file), 'rb') as pl:
        lganm_dict = pickle.load(pl)
        lganm_dict.update({'dict_name': 'lganm'})
        lganm_dict.update({'repeat': args.repeat})
        lganm_dict.update({'mid_size': lganm_dict['envs'][0].shape[1] - 1 })
        print(lganm_dict['case'].sem)
        print(lganm_dict['case'].target)
        print(lganm_dict['case'].truth)
        for env_id, env in enumerate(lganm_dict['envs']):
            print(env_id)
            print(env.shape, np.mean(env, axis=0), np.var(env, axis=0))

    for sequence_idx, sequence in enumerate(seq_info_dict['sequences']):
        for task_id in range(seq_info_dict['num_tasks']):
            lganm_dict['res_dir'] = Path(settings['results_dir']) / \
                prefixes[sequence_idx]
            lganm_dict['res_dir'].mkdir(parents=True, exist_ok=True)
            main(lganm_dict=lganm_dict,
                 task_id=task_id,
                 sequence_str=sequence,
                 sequence_name=prefixes[sequence_idx],
                 synthesizer=settings['synthesizer'])
