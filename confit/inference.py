import os

import torch
import pandas as pd
import numpy as np
from stat_utils import compute_stat, spearman
import argparse


parser = argparse.ArgumentParser(description='inference!')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--shot', type=int, help='training size')
parser.add_argument('--no_retrival', action='store_true', help='whether use retrival')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='retrieval alpha')
args = parser.parse_args()


if os.path.exists(f'results/{args.dataset}/summary.csv'):
    summary = pd.read_csv(f'results/{args.dataset}/summary.csv', index_col=0)
else:
    summary = pd.DataFrame(None)


if os.path.exists(f'predicted/{args.dataset}/pred.csv'):
    pred = pd.read_csv(f'predicted/{args.dataset}/pred.csv', index_col=0)
    pred = pred.drop_duplicates(subset='PID')
    if not args.no_retrival:
        elbo = pd.read_csv(f'data/{args.dataset}/vae_elbo.csv', index_col=0)
    seed_list = []
    for i in range(1, 6):
        if f'{i}' in pred.columns:
            seed_list.append(f'{i}')
    temp = pred[seed_list]
    temp = temp.mean(axis=1)
    pred = pd.concat([pred, temp], axis=1)
    pred = pred.rename(columns={0: 'avg'})
    test = pd.read_csv(f'data/{args.dataset}/test.csv', index_col=0)
    avg = pred[['avg', 'PID']]
    label = test[['PID', 'log_fitness']]
    perf = pd.merge(avg, label, on='PID')

    if not args.no_retrival:
        perf = pd.merge(perf, elbo, on='PID')
        perf['retrival'] = args.alpha * perf['avg'] + (1 - args.alpha) * perf['elbo']
        score = list(perf['retrival'])
        gscore = list(perf['log_fitness'])
        score = np.asarray(score)
        gscore = np.asarray(gscore)
        sr = spearman(score, gscore)
        out = pd.DataFrame({'spearman': sr, 'shot': args.shot}, index=[f'{args.dataset}'])
        summary = pd.concat([summary, out], axis=0)
    else:

        score = list(perf['avg'])
        gscore = list(perf['log_fitness'])
        score = np.asarray(score)
        gscore = np.asarray(gscore)
        sr = spearman(score, gscore)
        out = pd.DataFrame({'spearman': sr, 'shot': args.shot}, index=[f'{args.dataset}'])
        summary = pd.concat([summary, out], axis=0)

    if not os.path.isdir(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')

    summary.to_csv(f'results/{args.dataset}/summary.csv')
