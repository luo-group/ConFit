import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


def compute_score(model, seq, mask, wt, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()
    m_id = tokenizer.mask_token_id

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id

    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        wt_i = wt[i]
        seq_i = seq[i]
        scores[i] = torch.sum(score_i[mut_pos+1, seq_i[mut_pos+1]])-torch.sum(score_i[mut_pos+1, wt_i[mut_pos+1]])

    return scores, logits


def BT_loss(scores, golden_score):
    loss = torch.tensor(0.)
    loss = loss.cuda()
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss


def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss