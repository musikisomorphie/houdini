import numpy as np
import pyreadstat
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from Data.DataGenerator import NumpyDataSetIterator


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
        # # df = df.fillna(0)
    df_trn = (df[feat].values,
              df[label].values)
    print(df_trn[0].shape, df_trn[1].shape)
    df_val, df_tst = df_trn, df_trn
    return df_trn, df_val, df_tst


def get_lganm_io_examples(lganm_envs,
                          parents,
                          outcome):
    dt = np.concatenate(lganm_envs, axis=0)
    msk = np.ones(dt.shape[1], dtype=bool)
    msk[outcome] = False
    dt_trn = (dt[:, msk], dt[:, ~msk])
    # print(msk, ~msk)
    # print(dt_trn[0].shape, dt_trn[1].shape)
    dt_val, dt_tst = dt_trn, dt_trn
    return dt_trn, dt_val, dt_tst


def sav_to_csv(sav_file,
               csv_file):
    df, _ = pyreadstat.read_sav(str(sav_file))
    df.to_csv(str(csv_file))


def main():
    sav_file = Path('/home/histopath/Data/PORTEC/PORTEC12-1-2-21.sav')
    csv_file = sav_file.with_suffix('.csv')
    sav_to_csv(sav_file, csv_file)


if __name__ == '__main__':
    main()
