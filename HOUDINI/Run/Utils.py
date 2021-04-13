import numpy as np
import math
import pyreadstat
import pickle
from pathlib import Path
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

    for ptc in portec:
        ptc = pad_array(ptc, max_len)
        df_input.append(ptc[:, :-len(label)])
        df_lab.append(ptc[:, -len(label):])

    df_trn = (np.stack(df_input, 1),
              np.stack(df_lab, 1))
    print(df_trn[0].shape, df_trn[1].shape)
    df_val, df_tst = df_trn, df_trn
    return df_trn, df_val, df_tst


def get_lganm_io_examples(lganm_envs,
                          parents,
                          outcome,
                          dt_dim,
                          trn_batch,
                          val_batch,
                          tst_batch):

    max_len = 0
    dt_input, dt_lab = list(), list()
    msk = np.ones(dt_dim, dtype=bool)
    msk[outcome] = False
    for env in lganm_envs:
        max_len = max(max_len, env.shape[0])

    for env in lganm_envs:
        env = pad_array(env, max_len)
        dt_input.append(env[:, msk])
        dt_lab.append(env[:, ~msk])

    dt_trn = (np.stack(dt_input, 1),
              np.stack(dt_lab, 1))
    dt_val, dt_tst = dt_trn, dt_trn

    # dt = np.concatenate(lganm_envs, axis=1)
    # msk = np.ones(dt.shape[1], dtype=bool)
    # outs = list(range(outcome, msk.shape[0], dt_dim))
    # msk[outs] = False
    print(msk, dt_trn[1].shape)
    # input_dim = dt_dim - 1
    # dt_trn = ListNumpyDataSetIterator(input_dim,
    #                                   dt[:, msk],
    #                                   dt[:, ~msk],
    #                                   trn_batch)
    # dt_val = ListNumpyDataSetIterator(input_dim,
    #                                   dt[:, msk],
    #                                   dt[:, ~msk],
    #                                   val_batch)
    # dt_tst = ListNumpyDataSetIterator(input_dim,
    #                                   dt[:, msk],
    #                                   dt[:, ~msk],
    #                                   tst_batch)
    # print(msk, ~msk)
    # print(dt_trn[0].shape, dt_trn[1].shape)
    # dt_val, dt_tst = dt_trn, dt_trn
    return dt_trn, dt_val, dt_tst


def sav_to_csv(sav_file,
               csv_file):
    df, _ = pyreadstat.read_sav(str(sav_file))
    df.to_csv(str(csv_file))


def prep_sav(sav_file):
    df, meta = pyreadstat.read_sav(str(sav_file))

    df['log2_id'] = np.log2(df.index.astype('int64') + 1)
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


def main():
    sav_file = Path('/home/histopath/Data/PORTEC/PORTEC12-1-2-21.sav')
    # csv_file = sav_file.with_suffix('.csv')
    # sav_to_csv(sav_file, csv_file)
    prep_sav(sav_file)


if __name__ == '__main__':
    main()
