import argparse
import sys
import json
import pathlib
import numpy as np
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple, Union, List

from HOUDINI.Config import config
from HOUDINI.Run.Task import Task
from HOUDINI.Run.Task import TaskSettings
from HOUDINI.Run.TaskSeq import TaskSeqSettings, TaskSeq
from HOUDINI.Run.Utils import get_portec_io_examples
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
        if c_str[:4] == 'surv':
            c_task_type = TaskType.Surv
        else:
            raise NotImplementedError()

        c_dataset = Dataset.PORTEC

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
    seq_dict = {'surv': {'sequences': ['survP'],
                         'prefixes': ['surv'],
                         'num_tasks': 1}}

    if not seq_string in seq_dict.keys():
        raise NameError('the seq {} are not valid'.format(seq_string))
    return seq_dict[seq_string]


def get_task_settings(data_dict: Dict,
                      dbg_learn_parameters: bool,
                      synthesizer: str) -> TaskSettings:
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
        N=200,
        M=1,
        K=1,
        synthesizer=synthesizer,
        dbg_learn_parameters=dbg_learn_parameters,
        learning_rate=0.02,
        var_num=data_dict['clinical_meta']['causal_num'],
        warm_up=8,
        lambda_1=5,
        lambda_2=0.08,
        lambda_cau=10.,
        data_dict=data_dict)
    return task_settings


class TaskType(Enum):
    Surv = 1


class Dataset(Enum):
    PORTEC = 1


class SurvTask(Task):
    def __init__(self,
                 portec_dict,
                 settings,
                 seq,
                 dbg_learn_parameters):

        self.file = portec_dict['file']
        self.causal = list(portec_dict['clinical_meta']['causal'].keys())
        self.outcome = portec_dict['clinical_meta']['outcome']
        self.flter = list(portec_dict['clinical_meta']['filter'].keys())
        input_type = mkListSort(mkRealTensorSort([1, len(self.causal)]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         seq,
                         dbg_learn_parameters)

    def get_io_examples(self):
        return get_portec_io_examples(self.file,
                                      self.causal,
                                      self.outcome,
                                      self.flter[0])

    def name(self):
        return 'surv_RFS'

    def sname(self):
        return 'surv'


class InferSequence(TaskSeq):
    def __init__(self,
                 portec_dict: Dict,
                 name: str,
                 list_task_info: List[SeqTaskInfo],
                 dbg_learn_parameters: bool,
                 seq_settings: TaskSeqSettings,
                 task_settings: TaskSettings,
                 lib: OpLibrary):
        """The class of the sequence task

        Args:
            portec_dict: dict storing portec data info
            name: task name
            list_task_info: list of the SeqTaskInfo namedtuple
            seq: the task squence class
            dbg_learn_parameters: True if learn weights
                False otherwise
            seq_settings: the settings for the learning sequence
            task_settings: the task settings storing important
                learning parameters
            lib: the library of higher order functions (Op)
        """

        self._name = name
        tasks = []
        for _, task_info in enumerate(list_task_info):
            if task_info.task_type == TaskType.Surv:
                tasks.append(SurvTask(portec_dict,
                                      task_settings,
                                      self,
                                      dbg_learn_parameters))
            else:
                raise NotImplementedError()

        super().__init__(tasks,
                         seq_settings,
                         lib)

    def name(self):
        return self._name

    def sname(self):
        return self._name


def main(portec_dict: Dict,
         task_id: int,
         seq_str: str,
         seq_name: str,
         synthesizer: str):
    """The main function that runs the portec experiments

    Args:
        portec_dict: dict storing portec data info
        task_id: the id of the task
        list_task_info: list of the SeqTaskInfo namedtuple
        seq_str: the prefix name that retrieves the seq info dict
        seq_name: name of the sequence
        synthesizer: enumerative or evolutionary
        confounder: the list of confounders
    """

    seq_settings = TaskSeqSettings(update_library=True,
                                   results_dir=settings['results_dir'])

    task_settings = get_task_settings(portec_dict,
                                      settings['dbg_learn_parameters'],
                                      synthesizer=synthesizer)

    lib = OpLibrary(['do', 'compose', 'map',
                     'repeat', 'cat'])

    seq_tasks_info = get_seq_from_string(seq_str)

    seq = InferSequence(portec_dict,
                        seq_name,
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
    parser.add_argument('--portec-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/PORTEC/',
                        metavar='DIR')
    parser.add_argument('--confounder',
                        type=str,
                        choices=['immu', 'mole', 'path', 'immu_cd8', 'immu_cd103'],
                        default='immu',
                        help='the experiments with confounders. (default: %(default)s)')
    parser.add_argument('--dt-file',
                        type=Path,
                        default='/mnt/sda1/Data/PORTEC/PORTEC.sav',
                        metavar='DIR',
                        help='path to the visualization folder')
    parser.add_argument('--repeat',
                        type=int,
                        default=8,
                        help='num of repeated experiments')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # python -m HOUDINI.Run.PORTEC --dt-file /home/histopath/Data/PORTEC/PORTEC12-1-2-21_prep.sav --dbg
    args = parse_args()

    settings = {'results_dir': str(args.portec_dir / 'Results' / args.confounder),
                # If False, the interpreter doesn't learn the new parameters
                'dbg_learn_parameters': True,
                'synthesizer': args.synthesizer,  # enumerative, evolutionary
                'seq_string': 'surv'}

    seq_info_dict = get_seq_info(settings['seq_string'])
    additional_prefix = '_np_{}'.format(
        'td' if settings['synthesizer'] == 'enumerative' else 'ea')
    prefixes = ['{}{}'.format(prefix, additional_prefix)
                for prefix in seq_info_dict['prefixes']]

    portec_dict = config('HOUDINI/Yaml/PORTEC.yaml')
    portec_dict = portec_dict[args.confounder]
    mid_size = len(portec_dict['clinical_meta']['causal'].keys()) + \
        len(portec_dict['clinical_meta']['outcome'])
    portec_parm = {'dict_name': 'portec',
                   'file': args.portec_dir / 'PORTEC12-1-2-21_prep.sav',
                   'repeat': args.repeat,
                   'mid_size': mid_size,
                   'out_type': 'hazard',
                   'env_num': 2,
                   'results_dir': args.portec_dir / 'Results' / args.confounder}
    portec_dict.update(portec_parm)
    pathlib.Path(portec_dict['results_dir']).mkdir(
        parents=True, exist_ok=True)

    for sequence_idx, sequence in enumerate(seq_info_dict['sequences']):
        for task_id in range(seq_info_dict['num_tasks']):
            main(portec_dict,
                 task_id,
                 sequence,
                 prefixes[sequence_idx],
                 settings['synthesizer'])

    jacads = np.asarray(portec_dict['json_out']['jacads'])
    fwers = np.asarray(portec_dict['json_out']['fwers'])
    portec_dict['json_out']['jacads_mean'] = np.mean(jacads)
    portec_dict['json_out']['jacads_std'] = np.std(jacads)
    portec_dict['json_out']['fwers_mean'] = np.mean(fwers)
    portec_dict['json_out']['fwers_std'] = np.std(fwers)
    print('\nJaccard Similarity (JS) mean: {}, std: {}.'.format(
        np.mean(jacads), np.std(jacads)))
    print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
        np.mean(fwers), np.std(fwers)))

    json_file = portec_dict['results_dir'] / 'portec_table.json'
    with open(str(json_file), 'w', encoding='utf-8') as f:
        json.dump(portec_dict['json_out'], f, ensure_ascii=False, indent=4)
