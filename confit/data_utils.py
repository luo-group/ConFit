import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from scipy.stats import bootstrap
import numpy as np
import os
from Bio import SeqIO


class Mutation_Set(Dataset):
    def __init__(self, data, fname, tokenizer, sep_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = sep_len
        self.seq, self.attention_mask = tokenizer(list(self.data['seq']), padding='max_length',
                                                  truncation=True,
                                                  max_length=self.seq_len).values()
        wt_path = os.path.join('data', fname, 'wt.fasta')
        for seq_record in SeqIO.parse(wt_path, "fasta"):
            wt = str(seq_record.seq)
        target = [wt]*len(self.data)
        self.target, self.tgt_mask = tokenizer(target, padding='max_length', truncation=True,
                                               max_length=self.seq_len).values()
        self.score = torch.tensor(np.array(self.data['log_fitness']))
        self.pid = np.asarray(data['PID'])

        if type(list(self.data['mutated_position'])[0]) != str:
            self.position = [[u] for u in self.data['mutated_position']]

        else:

            temp = [u.split(',') for u in self.data['mutated_position']]
            self.position = []
            for u in temp:
                pos = [int(v) for v in u]
                self.position.append(pos)

    def __getitem__(self, idx):
        return [self.seq[idx], self.attention_mask[idx], self.target[idx],self.tgt_mask[idx] ,self.position[idx], self.score[idx], self.pid[idx]]

    def __len__(self):
        return len(self.score)

    def collate_fn(self, data):
        seq = torch.tensor(np.array([u[0] for u in data]))
        att_mask = torch.tensor(np.array([u[1] for u in data]))
        tgt = torch.tensor(np.array([u[2] for u in data]))
        tgt_mask = torch.tensor(np.array([u[3] for u in data]))
        pos = [torch.tensor(u[4]) for u in data]
        score = torch.tensor(np.array([u[5] for u in data]), dtype=torch.float32)
        pid = torch.tensor(np.array([u[6] for u in data]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid


def sample_data(dataset_name, seed, shot, frac=0.2):
    '''
    sample the train data and test data
    :param seed: sample seed
    :param frac: the fraction of testing data, default to 0.2
    :param shot: the size of training data
    '''

    data = pd.read_csv(f'data/{dataset_name}/data.csv', index_col=0)
    test_data = data.sample(frac=frac, random_state=seed)
    train_data = data.drop(test_data.index)
    kshot_data = train_data.sample(n=shot, random_state=seed)
    assert len(kshot_data) == shot, (
        f'expected {shot} train examples, received {len(train_data)}')

    kshot_data.to_csv(f'data/{dataset_name}/train.csv')
    test_data.to_csv(f'data/{dataset_name}/test.csv')


def split_train(dataset_name):
    '''
    five equal split training data, one of which will be used as validation set when training ConFit
    '''
    train = pd.read_csv(f'data/{dataset_name}/train.csv', index_col=0)
    tlen = int(np.ceil(len(train) / 5))
    start = 0
    for i in range(1, 5):
        csv = train[start:start + tlen]
        start += tlen
        csv.to_csv(f'data/{dataset_name}/train_{i}.csv')
    csv = train[start:]
    csv.to_csv(f'data/{dataset_name}/train_{5}.csv')





def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]

def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    sr = (sr,)
    ci = list(bootstrap(sr, np.mean).confidence_interval)
    return mean, std, ci






