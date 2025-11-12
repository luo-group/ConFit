"""Training entry-point for ConFit on the GFP fluorescence dataset."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer

from .gfp_data import GFPDataset, get_gfp_dfs


try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from peft.utils.other import fsdp_auto_wrap_policy

    PEFT_AVAILABLE = True
    PEFT_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - handled at runtime
    PEFT_AVAILABLE = False
    PEFT_IMPORT_ERROR = exc


def bt_loss(scores: torch.Tensor, golden_score: torch.Tensor) -> torch.Tensor:
    """Compute the Bradley-Terry loss used in ConFit."""

    loss = torch.zeros(1, device=scores.device)
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss = loss + torch.log1p(torch.exp(scores[j] - scores[i]))
            else:
                loss = loss + torch.log1p(torch.exp(scores[i] - scores[j]))
    return loss


def kl_loss(
    logits: torch.Tensor, logits_reg: torch.Tensor, seq: torch.Tensor, att_mask: torch.Tensor
) -> torch.Tensor:
    """KL loss between the finetuned model and the frozen teacher."""

    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    batch_loss = torch.zeros(1, device=logits.device)
    for i in range(seq.shape[0]):
        seq_len = att_mask[i].sum()
        if seq_len <= 1:
            continue
        arange = torch.arange(seq_len, device=logits.device)
        reg = probs_reg[i, arange, seq[i, :seq_len]]
        pred = probs[i, arange, seq[i, :seq_len]]
        batch_loss = batch_loss + criterion(reg.log(), pred)
    return batch_loss


def compute_mutation_scores(
    model: EsmForMaskedLM,
    seq: torch.Tensor,
    mask: torch.Tensor,
    wt: torch.Tensor,
    positions: Sequence[torch.Tensor],
    tokenizer: EsmTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ConFit mutation scores for a batch of sequences."""

    masked_seq = seq.clone()
    mask_token_id = tokenizer.mask_token_id
    for batch_index, mutated_positions in enumerate(positions):
        if mutated_positions.numel() == 0:
            continue
        masked_seq[batch_index, mutated_positions + 1] = mask_token_id

    outputs = model(masked_seq, attention_mask=mask, output_hidden_states=True)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(seq.size(0), device=seq.device)

    for batch_index, mutated_positions in enumerate(positions):
        if mutated_positions.numel() == 0:
            continue
        residue_indices = mutated_positions + 1
        seq_ids = seq[batch_index, residue_indices]
        wt_ids = wt[batch_index, residue_indices]
        diff = log_probs[batch_index, residue_indices, seq_ids] - log_probs[
            batch_index, residue_indices, wt_ids
        ]
        scores[batch_index] = diff.sum()

    return scores, logits


def prepare_dataloaders(
    tokenizer: EsmTokenizer,
    per_device_batch_size: int,
    *,
    dataset_root: str = "dataset",
    max_length: int = 1024,
) -> Tuple[GFPDataset, GFPDataset, GFPDataset, DataLoader, DataLoader, DataLoader]:
    """Prepare ConFit dataloaders for the GFP dataset."""

    train_df, val_df, test_df = get_gfp_dfs(dataset_root)
    train_dataset = GFPDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = GFPDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = GFPDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, per_device_batch_size // 2),
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, per_device_batch_size // 2),
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def evaluate(
    accelerator: Accelerator,
    model: EsmForMaskedLM,
    dataloader: DataLoader,
    tokenizer: EsmTokenizer,
    *,
    return_predictions: bool = False,
    dataset: GFPDataset | None = None,
) -> Tuple[float, pd.DataFrame | None]:
    """Evaluate the model and optionally collect predictions."""

    model.eval()
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            seq, mask, wt, wt_mask, positions, labels, indices = batch
            seq = seq.to(accelerator.device)
            mask = mask.to(accelerator.device)
            wt = wt.to(accelerator.device)
            positions = [p.to(accelerator.device) for p in positions]
            labels = labels.to(accelerator.device)
            indices = indices.to(accelerator.device)

            scores, _ = compute_mutation_scores(model, seq, mask, wt, positions, tokenizer)
            gathered_scores = accelerator.gather(scores)
            gathered_labels = accelerator.gather(labels)
            gathered_indices = accelerator.gather(indices)
            all_scores.append(gathered_scores.cpu())
            all_labels.append(gathered_labels.cpu())
            all_indices.append(gathered_indices.cpu())

    pred_scores = torch.cat(all_scores).numpy()
    golden_scores = torch.cat(all_labels).numpy()
    sr_value = pd.Series(pred_scores).corr(pd.Series(golden_scores), method="spearman")
    sr = float(sr_value) if not math.isnan(sr_value) else 0.0

    if not return_predictions or dataset is None:
        return sr, None

    indices = torch.cat(all_indices).numpy().astype(int)
    rows = []
    for idx, prediction in zip(indices, pred_scores):
        example = dataset.sequence_from_index(idx)
        rows.append(
            {
                "dataset_index": idx,
                "sequence": example.sequence,
                "log_fluorescence": example.fluorescence,
                "predicted_score": float(prediction),
            }
        )
    predictions = pd.DataFrame(rows)
    return sr, predictions


def train_one_epoch(
    accelerator: Accelerator,
    model: EsmForMaskedLM,
    teacher: EsmForMaskedLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: EsmTokenizer,
    lambda_reg: float,
) -> float:
    """Run a single training epoch."""

    model.train()
    total_loss = torch.zeros(1, device=accelerator.device)
    for batch in dataloader:
        seq, mask, wt, wt_mask, positions, labels, _ = batch
        seq = seq.to(accelerator.device)
        mask = mask.to(accelerator.device)
        wt = wt.to(accelerator.device)
        wt_mask = wt_mask.to(accelerator.device)
        positions = [p.to(accelerator.device) for p in positions]
        labels = labels.to(accelerator.device)

        scores, logits = compute_mutation_scores(model, seq, mask, wt, positions, tokenizer)
        with torch.no_grad():
            teacher_logits = teacher(wt, attention_mask=wt_mask).logits
        loss = bt_loss(scores, labels) + lambda_reg * kl_loss(
            logits, teacher_logits, seq, mask
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.detach()

    gathered = accelerator.gather(total_loss)
    return float(gathered.mean().item())


def load_model_and_tokenizer(model_name: str, model_seed: int) -> Tuple[EsmForMaskedLM, EsmForMaskedLM, EsmTokenizer]:
    """Load the specified ESM model and tokenizer."""

    if model_name == "ESM-1v":
        base = EsmForMaskedLM.from_pretrained(f"facebook/esm1v_t33_650M_UR90S_{model_seed}")
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/esm1v_t33_650M_UR90S_{model_seed}")
    elif model_name == "ESM-2":
        base = EsmForMaskedLM.from_pretrained("facebook/esm2_t48_15B_UR50D")
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
    elif model_name == "ESM-1b":
        base = EsmForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    teacher = EsmForMaskedLM.from_pretrained(base.name_or_path)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return base, teacher, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConFit on the GFP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gfp_training.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset",
        help="Directory containing the GFP LMDB files.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    accelerator = Accelerator()
    if "seed" in config:
        accelerator.wait_for_everyone()
        accelerator.seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))

    model_name = config.get("model", "ESM-1b")
    model_seed = int(config.get("model_seed", 1))
    lambda_reg = float(config.get("lambda_reg", 0.1))
    max_epochs = int(config.get("max_epochs", 10))
    max_length = int(config.get("max_length", 1024))

    base_model, teacher_model, tokenizer = load_model_and_tokenizer(model_name, model_seed)

    use_lora = bool(config.get("use_lora", True))
    if use_lora and not PEFT_AVAILABLE:
        accelerator.print(
            "LoRA requested but PEFT/bitsandbytes could not be imported. "
            "Falling back to full-model fine-tuning."
        )
        if PEFT_IMPORT_ERROR is not None:
            accelerator.print(f"PEFT import error: {PEFT_IMPORT_ERROR}")
        use_lora = False

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.MASKED_LM,
            r=int(config.get("lora_r", 8)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.1)),
            target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(base_model, lora_config)
    else:
        model = base_model

    lr = float(config.get("ini_lr", 5e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_lr = float(config.get("min_lr", lr / 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * max_epochs,
        eta_min=min_lr,
    )

    if use_lora and PEFT_AVAILABLE and os.environ.get("ACCELERATE_USE_FSDP"):
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    global_batch_size = int(config.get("batch_size", 16))
    per_device_batch_size = max(1, global_batch_size // accelerator.num_processes)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = prepare_dataloaders(
        tokenizer,
        per_device_batch_size,
        dataset_root=args.dataset_root,
        max_length=max_length,
    )

    model, teacher_model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
        model,
        teacher_model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
    )
    teacher_model.eval()

    output_dir = Path(config.get("output_dir", "checkpoint/gfp"))
    prediction_dir = Path(config.get("prediction_dir", "predicted/gfp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    best_sr = -math.inf
    best_epoch = -1
    best_path = output_dir / "best_model"

    accelerator.print("Starting training...")
    for epoch in range(max_epochs):
        loss = train_one_epoch(
            accelerator,
            model,
            teacher_model,
            train_loader,
            optimizer,
            tokenizer,
            lambda_reg,
        )
        accelerator.print(f"Epoch {epoch}: loss={loss:.4f}")
        val_sr, _ = evaluate(accelerator, model, val_loader, tokenizer)
        accelerator.print(f"Epoch {epoch}: validation Spearman={val_sr:.4f}")
        scheduler.step()

        if val_sr > best_sr:
            best_sr = val_sr
            best_epoch = epoch
            accelerator.print(f"New best checkpoint at epoch {epoch} (Spearman={val_sr:.4f})")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(best_path, save_function=accelerator.save)

    accelerator.print(f"Best validation performance: epoch {best_epoch} (Spearman={best_sr:.4f})")

    accelerator.wait_for_everyone()
    del model
    accelerator.free_memory()

    if use_lora and PEFT_AVAILABLE:
        base_model, _, tokenizer = load_model_and_tokenizer(model_name, model_seed)
        model = PeftModel.from_pretrained(base_model, best_path)
    else:
        model = EsmForMaskedLM.from_pretrained(best_path)
    model = accelerator.prepare(model)

    test_sr, predictions = evaluate(
        accelerator,
        model,
        test_loader,
        tokenizer,
        return_predictions=True,
        dataset=test_dataset,
    )
    accelerator.print(f"Test Spearman correlation: {test_sr:.4f}")

    if predictions is not None and accelerator.is_main_process:
        predictions.to_csv(prediction_dir / "gfp_predictions.csv", index=False)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
