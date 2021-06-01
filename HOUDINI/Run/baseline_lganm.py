import json
import timeit
import os
import pickle
import numpy as np
from src import utils, icp, population_icp
import multiprocessing
import sempler
import os
import argparse
import pathlib
import joblib
os.sched_getaffinity(0)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lganm-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/LGANM/',
                        metavar='DIR')

    parser.add_argument('--method',
                        choices=['icp', 'nicp'],
                        default='icp',
                        help='the baseline method. (default: %(default)s)')

    parser.add_argument('--cores',
                        type=int,
                        default=-1,
                        help='num of cpu cores')

    parser.add_argument('--exp',
                        choices=['fin', 'abcd'],
                        default='abcd',
                        help='Experimental settings defined in AICP. (default: %(default)s)')

    args = parser.parse_args()

    return args


def identify_cau(pkl_file, alpha, max_pred, method):
    with open(str(pkl_file), 'rb') as pl:
        print(pkl_file)
        lganm_dict = pickle.load(pl)
        results = icp.icp_baseline(lganm_dict['envs'],
                                   lganm_dict['target'],
                                   alpha=alpha,
                                   max_predictors=max_pred,
                                   method=method,
                                   dataset='lganm')
        accept_var = results.estimate
        return accept_var, lganm_dict['truth'], pkl_file.stem


if __name__ == '__main__':
    # python -m HOUDINI.Run.LGANM --repeat 1 --exp fin
    lganm_dirs = ('/home/histopath/Data/LGANM/',)
                #   '/home/histopath/Data/LGANM_hidden_1/',
                #   '/home/histopath/Data/LGANM_hidden_2/')
    args = parse_args()
    for ldir in lganm_dirs:
        for mtd in ('nicp', 'icp'):        
            args.method = mtd
            args.lganm_dir = pathlib.Path(ldir)
            print(args.method, args.lganm_dir)
            res_dir = args.lganm_dir / 'Results' / args.exp / args.method
            pathlib.Path(res_dir).mkdir(parents=True,
                                        exist_ok=True)
            
            if args.cores == -1:
                num_cores = os.cpu_count() - 1 
            else:
                num_cores = min(os.cpu_count() - 1,  abs(args.cores))

            # maximum 7 causal preds for both exps,
            max_pred = 7
            if args.exp == 'fin':
                alpha = 0.0002
            elif args.exp == 'abcd':
                alpha = 0.002
            
            pkl_dir = args.lganm_dir / args.exp / 'n_1000'
            pkl_files = list(pkl_dir.glob('*.pickle'))[:250]
            start = timeit.default_timer()
            results = joblib.Parallel(n_jobs=args.cores)(
                joblib.delayed(identify_cau)(pkl_file, alpha, max_pred, args.method) for pkl_file in pkl_files)
            stop = timeit.default_timer()
            # print(results)

            json_out = dict()
            jacads, fwers, errors = list(), list(), list()
            for res in results:
                accept_var, causal_var, file_nm = res
                jacad = (len(causal_var.intersection(accept_var))) / \
                    (len(causal_var.union(accept_var)))
                fwer = not accept_var.issubset(causal_var)
                jacads.append(jacad)
                fwers.append(fwer)
                if jacad != 1.:
                    errors.append(file_nm)

            jacads = np.asarray(jacads)
            fwers = np.asarray(fwers)
            json_out['jacads_mean'] = np.mean(jacads)
            json_out['jacads_std'] = np.std(jacads)
            json_out['fwers_mean'] = np.mean(fwers)
            json_out['fwers_std'] = np.std(fwers)
            json_out['errors'] = errors
            json_out['running_time'] = stop - start

            print('\nJaccard Similarity (JS) mean: {}, std: {}.'.format(
                np.mean(jacads), np.std(jacads)))
            print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
                np.mean(fwers), np.std(fwers)))
            print('running time: {}'.format(stop - start))

            json_file = pathlib.Path(res_dir) / 'lganm_table.json'
            with open(str(json_file), 'w', encoding='utf-8') as f:
                json.dump(json_out, f, ensure_ascii=False, indent=4)





    # for pkl_id, pkl_file in enumerate(pkl_dir.glob('*.pickle')):
    #     if pkl_id > 5:
    #         continue
    #     with open(str(pkl_file), 'rb') as pl:
    #         lganm_dict = pickle.load(pl)
    #         results = icp.icp_baseline(lganm_dict['envs'],
    #                                    lganm_dict['target'],
    #                                    alpha=alpha,
    #                                    max_predictors=max_pred,
    #                                    method=args.method,
    #                                    dataset='lganm')
    #         accept_var = results.estimate
    #         print(accept_var, lganm_dict['truth'])
    #         causal_var = set(lganm_dict['truth'])
    #         jacad = (len(causal_var.intersection(accept_var))) / \
    #             (len(causal_var.union(accept_var)))
    #         fwer = not accept_var.issubset(causal_var)
    #         jacads.append(jacad)
    #         fwers.append(fwer)
    #         if jacad != 1.:
    #             errors.append(pkl_file.stem)
    #         print('Jaccard Similarity (JS): {}.'.format(
    #             sum(jacads) / len(jacads)))
    #         print('Family-wise error rate (FWER): {}'.format(
    #             sum(fwers) / len(fwers)))
    #         print('errors: {}'.format(errors))
    # jacads = np.asarray(jacads)
    # fwers = np.asarray(fwers)
    # json_out['jacads_mean'] = np.mean(jacads)
    # json_out['jacads_std'] = np.std(jacads)
    # json_out['fwers_mean'] = np.mean(fwers)
    # json_out['fwers_std'] = np.std(fwers)
    # json_out['errors'] = errors
    # print('\nJaccard Similarity (JS) mean: {}, std: {}.'.format(
    #     np.mean(jacads), np.std(jacads)))
    # print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
    #     np.mean(fwers), np.std(fwers)))

    # json_file = pathlib.Path(res_dir) / 'lganm_table.json'
    # with open(str(json_file), 'w', encoding='utf-8') as f:
    #     json.dump(json_out, f, ensure_ascii=False, indent=4)
