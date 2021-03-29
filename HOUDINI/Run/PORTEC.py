import argparse
import sys
import random
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

SequenceTaskInfo = namedtuple(
    'SequenceTaskInfo', ['task_type', 'dataset'])


class TaskType(Enum):
    Classify = 1
    Predict = 2


class Dataset(Enum):
    PORTEC = 1


class ClassifyTask(Task):
    def __init__(self,
                 portec_dict_dict,
                 settings,
                 seq,
                 dbg_learn_parameters):

        self.feat = portec_dict['clinical_meta']['causal']
        self.label = portec_dict['clinical_meta']['outcome'][0]
        self.file = portec_dict['file']
        input_type = mkRealTensorSort([1, len(self.feat)])
        output_type = mkBoolTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def get_io_examples(self):
        return get_portec_io_examples(self.file,
                                      self.feat,
                                      self.label)

    def name(self):
        return 'classify_RFS'

    def sname(self):
        return 'cr'


class PredictTask(Task):
    def __init__(self,
                 portec_dict,
                 settings,
                 seq,
                 dbg_learn_parameters):

        self.feat = portec_dict['clinical_meta']['causal']
        self.label = portec_dict['clinical_meta']['outcome'][1]
        self.file = portec_dict['file']
        input_type = mkRealTensorSort([1, len(self.feat)])
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def get_io_examples(self):
        return get_portec_io_examples(self.file,
                                      self.feat,
                                      self.label)

    def name(self):
        return 'predict_RFS'

    def sname(self):
        return 'pr'


def get_task_name(task_info):
    if task_info.task_type == TaskType.Classify:
        c_str = 'clas'
    else:
        c_str = 'pred'

    c_str += 'P'
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
        if c_str[:4] == 'clas':
            c_task_type = TaskType.Classify
        else:
            c_task_type = TaskType.Predict

        c_dataset = Dataset.PORTEC

        sequence.append(SequenceTaskInfo(task_type=c_task_type,
                                         dataset=c_dataset))

    return sequence


class InferSequence(TaskSeq):
    def __init__(self,
                 portec_dict,
                 name,
                 list_task_info,
                 dbg_learn_parameters,
                 seq_settings,
                 task_settings,
                 lib):
        self._name = name
        tasks = []
        for _, task_info in enumerate(list_task_info):
            if task_info.task_type == TaskType.Classify:
                tasks.append(ClassifyTask(portec_dict,
                                          task_settings,
                                          self,
                                          dbg_learn_parameters))

            if task_info.task_type == TaskType.Predict:
                tasks.append(PredictTask(portec_dict,
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
    seq_dict = {'clas': {'sequences': ['clasP'],
                         'prefixes': ['clas'],
                         'num_tasks': 1},
                'pred': {'sequences': ['predP'],
                         'prefixes': ['pred'],
                         'num_tasks': 1},
                'clpd': {'sequences': ['clasP', 'predP'],
                         'prefixes': ['clpd_cl', 'clpd_pd'],
                         'num_tasks': 2}}

    if not seq_string in seq_dict.keys():
        raise NameError('the seq {} are not valid'.format(seq_string))
    return seq_dict[seq_string]


def get_task_settings(dbg_mode,
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
            training_percentages=[2, 10, 20, 50, 100],
            N=10000,
            M=50,
            K=50,
            epochs=30,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters
        )
    else:
        task_settings = TaskSettings(
            train_size=64,
            val_size=64,
            training_percentages=[100],
            N=50,
            M=2,
            K=2,
            epochs=10,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters
        )
    return task_settings


def mk_default_lib():
    lib = OpLibrary(['compose', 'repeat', 'map_l',
                     'fold_l', 'conv_l', 'zeros'])
    return lib


def main(portec_dict,
         task_id,
         sequence_str,
         sequence_name,
         synthesizer):

    seq_settings = TaskSeqSettings(update_library=True,
                                   results_dir=settings['results_dir'])

    task_settings = get_task_settings(settings['dbg_mode'],
                                      settings['dbg_learn_parameters'],
                                      synthesizer=synthesizer)
    lib = mk_default_lib()

    seq_tasks_info = get_sequence_from_string(sequence_str)
    print_sequence(seq_tasks_info)

    seq = InferSequence(portec_dict,
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
    parser.add_argument('--taskseq',
                        choices=['clas', 'pred', 'cspd'],
                        required=True,
                        help='Task Sequence')
    parser.add_argument('--dbg',
                        action='store_true',
                        help='If set, the sequences run for a tiny amount of data'
                        )
    parser.add_argument('--dt-file',
                        type=Path,
                        default='/mnt/sda1/Data/PORTEC/PORTEC.sav',
                        metavar='DIR',
                        help='path to the visualization folder')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    portec_dict = config('HOUDINI/Yaml/PORTEC.yaml')
    portec_dict = portec_dict['PORTEC']
    portec_dict.update({'file': args.dt_file})

    settings = {
        'results_dir': 'Results',  # str(sys.argv[1])
        # If False, the interpreter doesn't learn the new parameters
        'dbg_learn_parameters': True,
        'dbg_mode': args.dbg,  # If True, the sequences run for a tiny amount of data
        'synthesizer': args.synthesizer,  # enumerative, evolutionary
        'seq_string': args.taskseq  # 'ls'  # cs1, cs2, cs3, ls
    }

    seq_info_dict = get_sequence_info(settings['seq_string'])

    # num_tasks = seq_info_dict['num_tasks']
    additional_prefix = '_np_{}'.format(
        'td' if settings['synthesizer'] == 'enumerative' else 'ea')
    prefixes = ['{}{}'.format(prefix, additional_prefix)
                for prefix in seq_info_dict['prefixes']]

    for sequence_idx, sequence in enumerate(seq_info_dict['sequences']):
        for task_id in range(seq_info_dict['num_tasks']):
            main(portec_dict=portec_dict,
                 task_id=task_id,
                 sequence_str=sequence,
                 sequence_name=prefixes[sequence_idx],
                 synthesizer=settings['synthesizer'])
