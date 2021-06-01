import json
import timeit
import time
import pickle
import numpy as np
import multiprocessing
import sempler
import os
import pyreadstat
import argparse
import pathlib
from src import utils, icp, population_icp
from HOUDINI import Config
from HOUDINI.Library import Loss, Metric
from HOUDINI.Run.Utils import get_portec_io_examples, get_portec_io_examples1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--portec-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/PORTEC/',
                        metavar='DIR')

    parser.add_argument('--method',
                        choices=['icp', 'nicp'],
                        default='icp',
                        help='the baseline method. (default: %(default)s)')

    parser.add_argument('--exp',
                        type=str,
                        choices=['immu', 'mole', 'path',
                                 'immu_cd8', 'immu_cd103',
                                 'path_sanity1', 'path_sanity2'],
                        default='immu_cd8',
                        help='the ablation experiments. (default: %(default)s)')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.01,
                        help='num of repeated experiments')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # python -m HOUDINI.Run.LGANM --repeat 1 --exp fin
    args = parse_args()
    res_dir = args.portec_dir / 'Results' / args.exp / args.method
    pathlib.Path(res_dir).mkdir(parents=True,
                                exist_ok=True)
    # maximum 8 causal variables for all exps
    max_pred = 8
    alpha = args.alpha if args.method == 'icp' else args.alpha * 10

    json_out = dict()
    jacads, fwers, errors = list(), list(), list()
    portec_dict = Config.config('HOUDINI/Yaml/PORTEC.yaml')
    portec_dict = portec_dict[args.exp]

    pfile = args.portec_dir / 'portec_prep.sav'
    pcausal = list(portec_dict['clinical_meta']['causal'].keys())
    poutcome = portec_dict['clinical_meta']['outcome']
    pflter = portec_dict['clinical_meta']['filter']
    cau_bin = ('log2_av_tot_cd8t', 5)
    df, _ = pyreadstat.read_sav(str(pfile))
    df = df.loc[df[pflter[0]] == int(pflter[1])]
    df = df.dropna(subset=pcausal)
    df = df.dropna(subset=poutcome)
    if cau_bin[0] in df.columns:
        df.loc[df[cau_bin[0]] <= cau_bin[1], cau_bin[0]] = 0
        df.loc[df[cau_bin[0]] > cau_bin[1], cau_bin[0]] = 1
    df = df[pcausal + poutcome + ['PortecStudy']]
    print(df.head())
    start = timeit.default_timer()
    results = icp.icp_baseline(df,
                               0,
                               alpha=alpha,
                               max_predictors=max_pred,
                               method=args.method,
                               dataset='portec')
    stop = timeit.default_timer()
    accept_var = set(results.estimate)
    # flatten the list of causal var lists 
    causal_var = [cau for cau_vars in portec_dict['truth'] for cau in cau_vars]
    causal_var = set(causal_var)
    print(accept_var)
    print(causal_var)
    jacad = (len(causal_var.intersection(accept_var))) / \
        (len(causal_var.union(accept_var)))
    fwer = not accept_var.issubset(causal_var)
    jacads.append(jacad)
    fwers.append(fwer)
    if jacad != 1.:
        errors.append(args.exp)
    jacads = np.asarray(jacads)
    fwers = np.asarray(fwers)
    json_out['jacads_mean'] = np.mean(jacads)
    json_out['jacads_std'] = np.std(jacads)
    json_out['fwers_mean'] = np.mean(fwers)
    json_out['fwers_std'] = np.std(fwers)
    json_out['errors'] = errors
    json_out['accept_var'] = list(accept_var)
    json_out['causal_var'] = list(causal_var)
    json_out['running_time'] = stop - start
    
    print('\nJaccard Similarity (JS) mean: {}, std: {}.'.format(
        np.mean(jacads), np.std(jacads)))
    print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
        np.mean(fwers), np.std(fwers)))
    print('running time: {}'.format(stop - start))
    json_file = pathlib.Path(res_dir) / 'lganm_table.json'
    with open(str(json_file), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=4)
