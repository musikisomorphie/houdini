from operator import index
from Data.DataGenerator import NumpyDataSetIterator, ListNumpyDataSetIterator
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import json
import random
import csv
import pyreadstat
import pickle
import pathlib
import itertools
import scipy.stats
import seaborn as sns
sns.set_theme()


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


def get_portec_io_examples(portec_file: pathlib.Path,
                           causal: List[str],
                           outcome: str,
                           fltr: Optional[List]) -> Tuple[Tuple, Tuple, Tuple]:
    """Obtain the portec data 

    Args:
        portec_file: the *.sav file storing portec data
        causal: the candidate causal variables
        outcome: the outcome data, RFSstatus and RFSyears
        flter: the filter condition that excludes patient data 

    Returns:
        the train, val, test portec data 
    """

    df, _ = pyreadstat.read_sav(str(portec_file))
    if fltr is None:
        df = df.fillna(0)
    else:
        df = df.loc[df[fltr[0]] == int(fltr[1])]
        df = df.dropna(subset=causal)
        df = df.dropna(subset=outcome)

    # the values of the cau_var are the median of that variable,
    # which is used to create binary lable
    cau_var = {'total_area': None,  # median 3.46 (portec 1), 4.02844 (portec 2)
               'log2_av_tot_cd8t': 5,
               'log2_av_tot_cd103t': 6}
    portec = list()
    df_input, df_lab = list(), list()
    max_len = 0
    for i in range(2):
        df_ptc = df.loc[df['PortecStudy'] == i + 1]
        for cau_name, cau_val in cau_var.items():
            if cau_name in df_ptc.columns:
                if cau_val is not None:
                    df_ptc.loc[df[cau_name] <= cau_val, cau_name] = 0
                    df_ptc.loc[df[cau_name] > cau_val, cau_name] = 1
                print(cau_name,
                      np.mean(df_ptc[cau_name]),
                      np.median(df_ptc[cau_name]),
                      np.std(df_ptc[cau_name]))
        df_ptc = np.concatenate((df_ptc[causal].values,
                                 df_ptc[outcome].values), axis=-1)
        portec.append(df_ptc)
        max_len = max(max_len, df_ptc.shape[0])

    for ptc_id, ptc in enumerate(portec):
        ptc = pad_array(ptc, max_len)
        df_input.append(ptc[:, :-len(outcome)])
        df_lab.append(ptc[:, -len(outcome):])
        # df_lab[-1][:] = ptc_id

    df_trn = (np.stack(df_input, 1),
              np.stack(df_lab, 1))
    df_val, df_tst = df_trn, df_trn
    print(df_trn[0].shape, df_trn[1].shape)
    return df_trn, df_val, df_tst


def get_portec_io_examples1(portec_file: pathlib.Path,
                            causal: List[str],
                            outcome: str,
                            study: int = 1) -> Tuple[Tuple, Tuple, Tuple]:
    """Obtain the portec data 

    Args:
        portec_file: the *.sav file storing portec data
        causal: the candidate causal variables
        outcome: the outcome data, RFSstatus and RFSyears
        flter: the filter condition that excludes patient data 

    Returns:
        the train, val, test portec data 
    """

    df, _ = pyreadstat.read_sav(str(portec_file))
    df = df.dropna(subset=causal)
    df = df.dropna(subset=outcome)
    df = df.loc[df['PortecStudy'] == study]
    print(len(df), study)

    # the values of the cau_var are the median of that variable,
    # which is used to create binary lable
    log2_id = 9  # median, 9.17 (portec1),9.06 (portec2)
    portec = list()
    df_input, df_lab = list(), list()
    max_len = 0
    for i in range(2):
        if i == 0:
            df_ptc = df.loc[df['log2_id'] < log2_id]
        else:
            df_ptc = df.loc[df['log2_id'] >= log2_id]
        df_ptc = np.concatenate((df_ptc[causal].values,
                                 df_ptc[outcome].values), axis=-1)
        portec.append(df_ptc)
        max_len = max(max_len, df_ptc.shape[0])
        print(len(df_ptc))

    for ptc_id, ptc in enumerate(portec):
        ptc = pad_array(ptc, max_len)
        df_input.append(ptc[:, :-len(outcome)])
        df_lab.append(ptc[:, -len(outcome):])
        # df_lab[-1][:] = ptc_id

    df_trn = (np.stack(df_input, 1),
              np.stack(df_lab, 1))
    df_val, df_tst = df_trn, df_trn
    print(df_trn[0].shape, df_trn[1].shape)
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
    # if confounder:
    #     cfd_msk[confounder] = False
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


def header_lookup(headers):
    """The header lookup table. Assign the index for each candidate as follow,

    var_id[patient id] = 0
    var_id[survival rate] = 1

    Args:
        headers: the name list of candidate causal variables,
                    outcome, patien id, etc.
    """

    var_id = dict()

    for idx, head in enumerate(headers):
        var_id[head] = idx

    return var_id


def prep_sav(portec_dir, decimal=2):
    # obtain the dict that maps case_id to spot_id
    key_tab = defaultdict(list)
    key_csv = portec_dir / 'PORTEC_key.csv'
    with open(str(key_csv), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        csv_id = None
        for csv_id, val in enumerate(csv_reader):
            # due to the excel to csv accuracy,
            # use decimal 2 for mapping
            case_id = round(float(val[0]), decimal)
            key_tab[case_id].append(str(val[2]).replace(' ', ''))
            key_tab[case_id].append(float(val[0]))
        # make sure no id replica after processing the case ids
        assert len(list(key_tab.keys())) == csv_id + \
            1, '{} {}'.format(len(list(key_tab.keys())), csv_id)

    # obtain the dict that maps spot id to total of tumor and stroma area
    path_tab = defaultdict(list)
    path_csv = portec_dir / 'PORTEC_vk.csv'
    with open(str(path_csv), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        var_id = header_lookup(next(csv_reader))
        print(var_id)
        for key, val in enumerate(csv_reader):
            spot_id = str(val[var_id['Spot Id']]).replace(' ', '')
            spot_val = val[var_id['Spot Valid']].lower() == 'true'
            if not spot_val:
                print('patient {} has {} spot, thus exlude'.format(
                    spot_id,
                    val[var_id['Spot Valid']]))
                continue

            stroma = val[var_id['Stroma Area (mm2)']]
            if not str(stroma).replace('.', '', 1).isdigit():
                print('stroma', key, stroma)
            tumor = val[var_id['Tumor Area (mm2)']]
            if not str(tumor).replace('.', '', 1).isdigit():
                print('tumor', key, tumor, float(tumor))
            path_tab[spot_id].append(float(stroma) + float(tumor))

    sav_file = portec_dir / 'PORTEC12-1-2-21.sav'
    df, meta = pyreadstat.read_sav(str(sav_file))
    df['total_area'] = None
    for index, row in df.iterrows():
        case_id = round(float(row['Case_id']), decimal)
        if case_id not in key_tab:
            # print(index, row['log2_av_tot_cd103v'])
            continue
        if key_tab[case_id][0] not in path_tab:
            # print(index, row['log2_av_tot_cd103v'])
            continue
        # check if the raw case id between two tables are almost equal
        assert np.allclose(row['Case_id'], key_tab[case_id][1])
        total_area = path_tab[key_tab[case_id][0]]
        assert total_area, total_area
        df.at[index, 'total_area'] = sum(total_area)

    # dummy value assign check
    for index, row in df.iterrows():
        case_id = round(float(row['Case_id']), decimal)
        if case_id in key_tab and key_tab[case_id][0] in path_tab:
            total_area = path_tab[key_tab[case_id][0]]
            assert row['total_area'] == sum(total_area)

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

    df_ind = (df.index.astype('int64') + 1).tolist()
    random.shuffle(df_ind)
    df['log2_id'] = np.log2(df_ind)

    new_sav = portec_dir / 'portec_prep.sav'
    print(new_sav)
    pyreadstat.write_sav(df, str(new_sav))
    new_csv = sav_file.parents[0] / 'portec_prep.csv'
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
    jacads = list()
    fwers = list()
    is_false = 0
    for aicp_idx, aicp_res in enumerate(aicp_results):
        case_idx = aicp_idx // repeat
        parents = set(aicp_cases[case_idx].truth)
        par_aicp = set(aicp_res.estimate)
        jacad = len(parents.intersection(par_aicp)) / \
            len(parents.union(par_aicp))
        jacads.append(jacad)
        fwers.append(not par_aicp.issubset(parents))

        if parents != par_aicp:
            is_false += 1
    jacads = np.asarray(jacads)
    fwers = np.asarray(fwers)
    print(np.mean(jacads), np.std(jacads))
    print(np.mean(fwers), np.std(fwers))

    # for aicp_id, aicp_case in enumerate(aicp_cases):
    #     if aicp_id >= 10:
    #         break
    #     print(aicp_case.target, aicp_case.truth)
    # for aicp_id, aicp_res in enumerate(aicp_results):
    #         for res_id, res in enumerate(aicp_res):
    #             if res_id <= 10:
    #                 print(res.estimate)
    # break

    return aicp


def heat_plot(json_file, var_list, exp_nm, func_nm, out_folder):
    with open(str(json_file)) as json_data:
        res_dict = json.load(json_data)
        for _, prog_dict in res_dict.items():
            caus_like = np.asarray(prog_dict['val_dos']).T
            hazd_coef = np.asarray(prog_dict['val_grads']).T
            pear_cau = np.zeros((caus_like.shape[0], caus_like.shape[0]))
            pear_haz = np.zeros((caus_like.shape[0], caus_like.shape[0]))
            pear_tot = np.zeros((caus_like.shape[0], caus_like.shape[0]))
            # corr = scipy.stats.pearsonr(caus_like, hazd_coef)
            mask = np.zeros_like(pear_cau)
            mask[np.triu_indices_from(mask, 1)] = True

            if func_nm == 'spearman':
                func = scipy.stats.spearmanr
            elif func_nm == 'kendall':
                func = scipy.stats.kendalltau
            elif func_nm == 'pearson':
                func = scipy.stats.pearsonr
            else:
                raise NotImplementedError()

            for i in range(caus_like.shape[0]):
                for j in range(hazd_coef.shape[0]):
                    pear_cau[i, j] = func(caus_like[i, ], caus_like[j, ])[0]
                    pear_haz[i, j] = func(hazd_coef[i, ], hazd_coef[j, ])[0]
                    pear_tot[i, j] = func(caus_like[i, ], hazd_coef[j, ])[0]

        df_cau = pd.DataFrame(pear_cau, columns=var_list, index=var_list)
        df_haz = pd.DataFrame(pear_haz, columns=var_list, index=var_list)
        df_tot = pd.DataFrame(pear_tot, columns=var_list, index=var_list)

        font = FontProperties()
        font.set_family('Helvetica')
        # font.set_name('Helvetica')

        fig, ax = plt.subplots(1, 2, sharey=True)
        s_0 = sns.heatmap(df_cau, annot=True, cbar=False, mask=mask, center=0, cmap="RdBu_r",
                          ax=ax[0], vmin=-1, vmax=1)
        s_0.set_xticklabels(s_0.get_xticklabels(), rotation=45,
                            fontsize=14, fontweight='bold', fontproperties=font)
        s_0.set_yticklabels(s_0.get_yticklabels(),
                            fontsize=14, fontweight='bold', fontproperties=font)
        ax[0].set_title('Causal Probability', pad=20, fontsize=24,
                        fontweight='bold', fontproperties=font)

        s_1 = sns.heatmap(df_haz, annot=True, mask=mask, center=0, cmap="RdBu_r",
                          ax=ax[1], vmin=-1, vmax=1)
        s_1.set_xticklabels(s_1.get_xticklabels(), rotation=45,
                            fontsize=14, fontweight='bold', fontproperties=font)
        ax[1].set_title('Hazard Coefficient', pad=20, fontsize=24,
                        fontweight='bold', fontproperties=font)

        fig.set_size_inches(32, 18, forward=True)
        plt.savefig(str(out_folder / '{}_{}_split.svg'.format(exp_nm, func_nm)))

        fig, ax = plt.subplots(1, 1)
        s_0 = sns.heatmap(np.abs(df_tot), annot=True, cbar=True, center=0, cmap="RdBu_r",
                          ax=ax, vmin=-1, vmax=1)
        s_0.set_xticklabels(s_0.get_xticklabels(), rotation=45,
                            fontsize=14, fontweight='bold', fontproperties=font)
        s_0.set_yticklabels(s_0.get_yticklabels(),
                            fontsize=14, fontweight='bold', fontproperties=font)

        # plt.show()
        fig.set_size_inches(32, 18, forward=True)
        plt.savefig(str(out_folder / '{}_{}_mixed.svg'.format(exp_nm, func_nm)))


def main():
    # sav_file = pathlib.Path('/raid/jiqing/Data/PORTEC/PORTEC12-1-2-21.sav')
    # csv_file = sav_file.with_suffix('.csv')
    # sav_to_csv(sav_file, csv_file)

    # sav_file = pathlib.Path('/home/histopath/Data/PORTEC/')
    # prep_sav(sav_file)

    # aicp_path = pathlib.Path(
    #     '/home/histopath/Data/PORTEC/Results_Nature/abcd_0/aicp/n_1000.pickle')
    # aicp_tests = compute_aicp_results(aicp_path)

    # funcs = ['pearson', 'spearman', 'kendall']
    # exps = ('path', 'mole', 'immu_cd8')
    # var_lists = (['Myometrial invasion', 'Grade', 'LVSI', 'Vital area', 'Patient ID'],
    #              ['Myometrial invasion', 'Grade', 'LVSI', 'L1CAM', 'POLE',
    #               'MMRd', 'p53mutant', 'Vital area', 'Patient ID'],
    #              ['Myometrial invasion', 'Grade', 'LVSI', 'L1CAM', 'POLE',
    #               'MMRd', 'p53mutant', 'CD8+', 'Vital area', 'Patient ID'])
    # res_dir = pathlib.Path('/home/histopath/Data/PORTEC/Results/')
    # out_dir = pathlib.Path(
    #     '/home/histopath/Data/PORTEC/Results_Nature/heatmap')
    # for e_id, exp in enumerate(exps):
    #     for func_nm in funcs:
    #         json_file = res_dir / exp / 'proposed_64' / 'portec_table.json'
    #         heat_plot(json_file, var_lists[e_id], exp, func_nm, out_dir)

    exps = ('path', 'mole', 'immu_cd8')
    met_nms = ('c-id', 'brie', 'bino')
    stg_nms = ['warm-up', 'complete']
    res_dir = pathlib.Path('/home/histopath/Data/PORTEC/Results/')
    for e_id, exp in enumerate(exps):
        json_file = res_dir / exp / 'proposed' / 'portec_table.json'
        with open(str(json_file)) as json_data:
            res_dict = list(json.load(json_data).values())[0]
            metrics = [res_dict['warm_scores'], res_dict['val_scores']]
            for ms_id, mets in enumerate(metrics):
                print(exp, stg_nms[ms_id])
                for m_id, met in enumerate(mets):
                    mn, q1, median, q3, mx = np.percentile(
                        np.asarray(met), [0, 25, 50, 75, 100])
                    met_sort = np.sort(np.asarray(met))
                    met_sort = met_sort[(
                        met_sort >= q1 - 1.5 * (q3 - q1)) & (met_sort <= q3 + 1.5 * (q3 - q1))]
                    mn = met_sort[0]
                    mx = met_sort[-1]
                    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(mn, q1 - mn,
                                                                             median - q1, q3 - median, mx - q3, mx))
                print('\n')


if __name__ == '__main__':
    main()
