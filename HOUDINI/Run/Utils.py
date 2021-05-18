import numpy as np
import math
import random
import pyreadstat
import pickle
import pathlib
import itertools
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from Data.DataGenerator import NumpyDataSetIterator, ListNumpyDataSetIterator


def mk_tag(tag: str, content: str, cls: List[str] = [], attribs: Dict = {}):
    cls_str = ' '.join(cls)
    if len(cls_str) > 0:
        cls_str = 'class = "%s"' % cls_str

    attrib_str = ' '.join(['%s="%s"' % (k, v) for k, v in attribs.items()])

    return '<%s %s %s>%s</%s>\n' % (tag, cls_str, attrib_str, content, tag)


def mk_div(content: str, cls: List[str] = []):
    return mk_tag('div', content, cls)


def append_to_file(a_file, content):
    with open(a_file, "a") as fh:
        fh.write(content)


def write_to_file(a_file, content):
    with open(a_file, "w") as fh:
        fh.write(content)


def pad_array(env,
              env_len):
    assert env_len >= env.shape[0], \
        'the max len {} should >= data len {}'.format(env_len, env.shape[0])
    # print('divmod result:', q, r)
    q, r = divmod(env_len, env.shape[0])
    if q > 1:
        env = np.repeat(env, q, axis=0)
    if r > 0:
        id_env = list(range(env.shape[0]))
        np.random.shuffle(id_env)
        env = np.concatenate((env, env[id_env[:r]]), axis=0)
    return env


def iterate_diff_training_sizes(train_io_examples, training_data_percentages):
    # assuming all lengths are represented equally
    if issubclass(type(train_io_examples), NumpyDataSetIterator) or \
            type(train_io_examples) == list and issubclass(type(train_io_examples[0]), NumpyDataSetIterator):
        num_of_training_dp = train_io_examples[0].inputs.shape[0]
        # raise NotImplementedError("uhm?!")
        yield train_io_examples, num_of_training_dp
        return
    num_of_training_dp = train_io_examples[0][0].shape[0] if type(train_io_examples) == list else \
        train_io_examples[0].shape[0]

    for percentage in training_data_percentages:
        c_num_items = (percentage * num_of_training_dp) // 100
        if type(train_io_examples) == list:
            c_tr_io_examples = [(t[0][:c_num_items], t[1][:c_num_items])
                                for t in train_io_examples]
            return_c_num_items = c_num_items * train_io_examples.__len__()
        else:
            c_tr_io_examples = (
                train_io_examples[0][:c_num_items], train_io_examples[1][:c_num_items])
            return_c_num_items = c_num_items

        yield c_tr_io_examples, return_c_num_items


def get_portec_io_examples(portec_file,
                           feat,
                           label,
                           fltr=None):
    df, _ = pyreadstat.read_sav(str(portec_file))
    if fltr is None:
        df = df.fillna(0)
    else:
        df = df.loc[df[fltr] == 1]
        df = df.dropna(subset=feat)
        df = df.dropna(subset=label)
        # df = df.fillna(0)
    portec = list()
    df_input, df_lab = list(), list()
    max_len = 0
    for i in range(2):
        df_ptc = df.loc[df['PortecStudy'] == i + 1]
        df_ptc = np.concatenate((df_ptc[feat].values,
                                 df_ptc[label].values), axis=-1)
        portec.append(df_ptc)
        max_len = max(max_len, df_ptc.shape[0])

    for ptc_id, ptc in enumerate(portec):
        ptc = pad_array(ptc, max_len)
        df_input.append(ptc[:, :-len(label)])
        df_lab.append(ptc[:, -len(label):])
        # df_input[-1][:, -1] = ptc_id
        df_lab[-1][:, -1] = ptc_id

    df_trn = (np.stack(df_input, 1),
              np.stack(df_lab, 1))
    print(df_trn[0].shape, df_trn[1].shape)
    df_val, df_tst = df_trn, df_trn
    return df_trn, df_val, df_tst


def get_lganm_io_examples(lganm_envs: List[np.ndarray],
                          confounder: List[int],
                        #   parent: List[int],
                          outcome: int,
                          dt_dim: int,
                          max_len: int = 6000) -> Tuple[Tuple, Tuple, Tuple]:
    """Obtain the lganm data 

    Args:
        lganm_envs: dict storing the lganm data 
            collected from different environments
        confounder: the list of confounders
        parent: the parents (ground-truth) of the outcome var
        outcome: outcome variable
        dt_dim: the data feature dimension including 
            candidate causal and outcome variable(s)
        max_len: the maximum number of data in each env

    Returns:
        the train, val, test lganm data 
    """

    dt_input, dt_lab = list(), list()
    lab_msk = np.zeros(dt_dim, dtype=bool)
    lab_msk[outcome] = True
    cfd_msk = np.ones(dt_dim, dtype=bool)
    cfd_msk[outcome] = False
    if confounder:
        cfd_msk[confounder] = False
    for env_id, env in enumerate(lganm_envs):
        print(env.shape)
        if env.shape[0] < max_len:
            env = pad_array(env, max_len)
        else:
            env = env[:max_len]
        dt_input.append(env[:, cfd_msk])
        dt_lab.append(env[:, lab_msk])
        # dt_lab[-1][:]= env_id

    dt_trn = (np.stack(dt_input, 1),
              np.stack(dt_lab, 1))
    dt_val, dt_tst = dt_trn, dt_trn
    return dt_trn, dt_val, dt_tst


def sav_to_csv(sav_file,
               csv_file):
    df, _ = pyreadstat.read_sav(str(sav_file))
    df.to_csv(str(csv_file))


def prep_sav(sav_file):
    df, meta = pyreadstat.read_sav(str(sav_file))
    df_ind = (df.index.astype('int64') + 1).tolist()
    random.shuffle(df_ind)
    df['log2_id'] = np.log2(df_ind)
    df['POLE'] = df['TCGA_4groups']
    df.loc[df['POLE'] == 1, 'POLE'] = 1
    df.loc[df['POLE'] == 2, 'POLE'] = 0
    df.loc[df['POLE'] == 3, 'POLE'] = 0
    df.loc[df['POLE'] == 4, 'POLE'] = 0

    df['MMRd'] = df['TCGA_4groups']
    df.loc[df['MMRd'] == 1, 'MMRd'] = 0
    df.loc[df['MMRd'] == 2, 'MMRd'] = 1
    df.loc[df['MMRd'] == 3, 'MMRd'] = 0
    df.loc[df['MMRd'] == 4, 'MMRd'] = 0

    df['p53_mutant'] = df['TCGA_4groups']
    df.loc[df['p53_mutant'] == 1, 'p53_mutant'] = 0
    df.loc[df['p53_mutant'] == 2, 'p53_mutant'] = 0
    df.loc[df['p53_mutant'] == 3, 'p53_mutant'] = 1
    df.loc[df['p53_mutant'] == 4, 'p53_mutant'] = 0

    df['EBRT'] = df['ReceivedTx']
    df.loc[df['EBRT'] == 1, 'EBRT'] = 1
    df.loc[df['EBRT'] == 2, 'EBRT'] = 0

    df['VBT'] = df['ReceivedTx']
    df.loc[df['VBT'] == 1, 'VBT'] = 0
    df.loc[df['VBT'] == 2, 'VBT'] = 1

    new_sav = sav_file.parents[0] / (sav_file.stem + '_prep.sav')
    print(new_sav)
    pyreadstat.write_sav(df, str(new_sav))
    new_csv = sav_file.parents[0] / (sav_file.stem + '_prep.csv')
    print(new_csv)
    df.to_csv(str(new_csv))


def load_pickle(path):
    with open(path, 'rb') as apath:
        dt_dict = pickle.load(apath)
    return dt_dict


def compute_aicp_results(path):
    with open(path, 'rb') as apath:
        aicp = pickle.load(apath)
    aicp_cases = aicp[0]
    aicp_results = list(itertools.chain.from_iterable(aicp[1]))

    repeat = len(aicp_results) // len(aicp_cases)
    jacobs = list()
    fwer = list()
    is_false = 0
    for aicp_idx, aicp_res in enumerate(aicp_results):
        case_idx = aicp_idx // repeat
        parents = set(aicp_cases[case_idx].truth)
        par_aicp = set(aicp_res.estimate)
        jacob = len(parents.intersection(par_aicp)) / \
            len(parents.union(par_aicp))
        jacobs.append(jacob)
        fwer.append(not par_aicp.issubset(parents))

        if parents != par_aicp:
            is_false += 1

    print(sum(jacobs) / len(jacobs))
    print(sum(fwer) / len(fwer))
    print(is_false)

    # for aicp_id, aicp_case in enumerate(aicp_cases):
    #     if aicp_id >= 10:
    #         break
    #     print(aicp_case.target, aicp_case.truth)
    # for aicp_id, aicp_res in enumerate(aicp_results):
    #         for res_id, res in enumerate(aicp_res):
    #             if res_id <= 10:
    #                 print(res.estimate)
    #                 # break

    return aicp


def main():
    # sav_file = Path('/home/histopath/Data/PORTEC/PORTEC12-1-2-21.sav')
    # csv_file = sav_file.with_suffix('.csv')
    # sav_to_csv(sav_file, csv_file)
    # prep_sav(sav_file)
    # pkl_path = pathlib.Path(
    #     '/home/histopath/Model/LGANM/Results/fin/n_1000_res.pickle')
    # res_dict = load_pickle(pkl_path)
    # print(sum(res_dict['jacob']) / len(res_dict['jacob']))
    # print(sum(res_dict['fwer']) / len(res_dict['fwer']))
    # print(res_dict['error'])
    # print(len(res_dict['jacob']))
    aicp_path = pathlib.Path('/home/histopath/Data/LGANM/fin/n_1000.pickle')
    aicp_tests = compute_aicp_results(aicp_path)


if __name__ == '__main__':
    main()
