import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
import os
import argparse
from pathlib import Path
import accelerate
from accelerate import Accelerator

from data_utils import Mutation_Set, split_train, sample_data
from stat_utils import spearman, compute_score, BT_loss, KLloss
import gc
import warnings
import time
import yaml
warnings.filterwarnings("ignore")



def train(model, model_reg, trainloder, optimizer, tokenizer, lambda_reg):

    model.train()

    total_loss = 0.

    for step, data in enumerate(trainloder):
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        golden_score = data[5]
        score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)
        score = score.cuda()

        l_BT = BT_loss(score, golden_score)

        out_reg = model_reg(wt, wt_mask)
        logits_reg = out_reg.logits
        l_reg = KLloss(logits, logits_reg, seq, mask)

        loss = l_BT + lambda_reg*l_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def evaluate(model, testloader, tokenizer, accelerator, istest=False):
    model.eval()
    seq_list = []
    score_list = []
    gscore_list = []
    with torch.no_grad():
        for step, data in enumerate(testloader):
            seq, mask = data[0], data[1]
            wt, wt_mask = data[2], data[3]
            pos = data[4]
            golden_score = data[5]
            pid = data[6]
            if istest:
                pid = pid.cuda()
                pid = accelerator.gather(pid)
                for s in pid:
                    seq_list.append(s.cpu())

            score, logits = compute_score(model, seq, mask, wt, pos, tokenizer)

            score = score.cuda()
            score = accelerator.gather(score)
            golden_score = accelerator.gather(golden_score)
            score = np.asarray(score.cpu())
            golden_score = np.asarray(golden_score.cpu())
            score_list.extend(score)
            gscore_list.extend(golden_score)
    score_list = np.asarray(score_list)
    gscore_list = np.asarray(gscore_list)
    sr = spearman(score_list, gscore_list)

    if istest:
        seq_list = np.asarray(seq_list)

        return sr, score_list, seq_list
    else:
        return sr


def main():
    parser = argparse.ArgumentParser(description='ConFit train, set hyperparameters')
    parser.add_argument('--config', type=str, default='48shot_config.yaml',
                        help='the config file name')
    parser.add_argument('--dataset', type=str, help='the dataset name')
    parser.add_argument('--sample_seed', type=int, default=0, help='the sample seed for dataset')
    parser.add_argument('--model_seed', type=int, default=1, help='the random seed for the pretrained model initiate')
    args = parser.parse_args()
    dataset = args.dataset

    #read in config
    with open(f'{args.config}', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    batch_size = int(int(config['batch_size'])/int(config['gpu_number']))


    accelerator = Accelerator()

    ### creat model
    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    elif config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    elif config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    for pm in model_reg.parameters():
        pm.requires_grad = False
    model_reg.eval()    #regularization model


    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=["query", "value"]
    )

    model = get_peft_model(basemodel, peft_config)

    # create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['ini_lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2*int(config['max_epochs']), eta_min=float(config['min_lr']))
    if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model_reg = accelerator.prepare(model_reg)

    accelerator.print(f'===================dataset:{dataset}, preparing data=============')

    # sample data
    if accelerator.is_main_process:
        sample_data(dataset, args.sample_seed, int(config['shot']))
        split_train(dataset)

    with accelerator.main_process_first():
        train_csv = pd.DataFrame(None)
        test_csv = pd.read_csv(f'data/{dataset}/test.csv')
        for i in range(1, 6):
            if i == args.model_seed:
                val_csv = pd.read_csv(f'data/{dataset}/train_{i}.csv')   #using 1/5 train data as validation set
            temp_csv = pd.read_csv(f'data/{dataset}/train_{i}.csv')
            train_csv = pd.concat([train_csv, temp_csv], axis=0)


    #creat dataset and dataloader
    trainset = Mutation_Set(data=train_csv, fname=dataset, tokenizer=tokenizer)
    testset = Mutation_Set(data=test_csv, fname=dataset,  tokenizer=tokenizer)
    valset = Mutation_Set(data=val_csv, fname=dataset,  tokenizer=tokenizer)
    with accelerator.main_process_first():
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, shuffle=True)
        testloader = DataLoader(testset, batch_size=2, collate_fn=testset.collate_fn)
        valloader = DataLoader(valset, batch_size=2, collate_fn=testset.collate_fn)

    trainloader = accelerator.prepare(trainloader)
    testloader = accelerator.prepare(testloader)
    valloader = accelerator.prepare(valloader)
    accelerator.print('==============data preparing done!================')
    # accelerator.print("Current allocated memory:", torch.cuda.memory_allocated())
    # accelerator.print("cached:", torch.cuda.memory_reserved())


    best_sr = -np.inf
    endure = 0
    best_epoch = 0

    for epoch in range(int(config['max_epochs'])):
        loss = train(model, model_reg, trainloader, optimizer, tokenizer, float(config['lambda_reg']))
        accelerator.print(f'========epoch{epoch}; training loss :{loss}=================')
        sr = evaluate(model, valloader, tokenizer, accelerator)
        accelerator.print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
        scheduler.step()
        if best_sr > sr:
            endure += 1
        else:
            endure = 0
            best_sr = sr
            best_epoch = epoch

            if not os.path.isdir(f'checkpoint/{dataset}'):
                if accelerator.is_main_process:
                    os.makedirs(f'checkpoint/{dataset}')
            save_path = os.path.join('checkpoint', f'{dataset}',
                                     f'seed{args.model_seed}')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
        if sr == 1.0:
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break
        if endure > int(config['endure_time']):
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break

    # inference on the test sest
    accelerator.print('=======training done!, test the performance!========')
    save_path = Path(os.path.join('checkpoint', f'{dataset}', f'seed{args.model_seed}'))
    del basemodel
    del model
    accelerator.free_memory()

    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    if config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    if config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    model = PeftModel.from_pretrained(basemodel, save_path)
    model = accelerator.prepare(model)
    sr, score, pid = evaluate(model, testloader, tokenizer, accelerator, istest=True)
    pred_csv = pd.DataFrame({f'{args.model_seed}': score, 'PID': pid})
    if accelerator.is_main_process:
        if not os.path.isdir(f'predicted/{dataset}'):
            os.makedirs(f'predicted/{dataset}')
        if os.path.exists(f'predicted/{dataset}/pred.csv'):
            pred = pd.read_csv(f'predicted/{dataset}/pred.csv', index_col=0)
            pred = pd.merge(pred, pred_csv, on='PID')
        else:
            pred = pred_csv
        pred.to_csv(f'predicted/{dataset}/pred.csv')
    accelerator.print(f'=============the test spearman correlation for early stop: {sr}==================')


if __name__ == "__main__":
    main()





