"""Utilities for working with the GFP fluorescence dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tape.datasets import LMDBDataset
from transformers import PreTrainedTokenizer


GFP_AMINO_ACID_VOCABULARY = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    ".",
]
GFP_ALPHABET = {aa: i for i, aa in enumerate(GFP_AMINO_ACID_VOCABULARY)}


def gfp_dataset_to_df(lmdb_path: str) -> pd.DataFrame:
    """Load an LMDB dataset into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    lmdb_path:
        Path to the LMDB file produced by the TAPE GFP benchmark.
    """

    dataset = LMDBDataset(lmdb_path)
    df = pd.DataFrame(list(dataset)[:])
    df["log_fluorescence"] = df.log_fluorescence.apply(lambda x: x[0])
    return df


def get_gfp_dfs(dataset_root: str = "dataset") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test dataframes for the GFP dataset."""

    train_df = gfp_dataset_to_df(f"{dataset_root}/fluorescence_train.lmdb")
    val_df = gfp_dataset_to_df(f"{dataset_root}/fluorescence_valid.lmdb")
    test_df = gfp_dataset_to_df(f"{dataset_root}/fluorescence_test.lmdb")
    return train_df, val_df, test_df


@dataclass(frozen=True)
class GFPExample:
    """Container for a single GFP sequence and metadata."""

    index: int
    sequence: str
    fluorescence: float


class GFPDataset(Dataset):
    """Torch dataset that tokenises GFP sequences for ConFit training."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        *,
        max_length: int = 1024,
        sequence_column: str = "primary",
        label_column: str = "log_fluorescence",
        num_mutations_column: str = "num_mutations",
    ) -> None:
        if sequence_column not in dataframe or label_column not in dataframe:
            raise KeyError(
                "The GFP dataframe must contain columns 'primary' and 'log_fluorescence'."
            )

        if num_mutations_column not in dataframe:
            raise KeyError(
                "The GFP dataframe must contain the 'num_mutations' column so mutation positions can be inferred."
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

        df = dataframe.reset_index(drop=True).copy()
        parent_rows = df[df[num_mutations_column] == 0]
        if parent_rows.empty:
            raise ValueError(
                "Unable to locate the wild-type parent sequence (row with num_mutations == 0)."
            )
        self.parent_sequence = str(parent_rows.iloc[0][sequence_column])
        self.data = df
        self.sequences: List[str] = self.data[sequence_column].astype(str).tolist()
        self.labels = torch.tensor(
            self.data[label_column].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.sample_indices = torch.arange(len(self.data), dtype=torch.long)

        tokenised = tokenizer(
            self.sequences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.sequence_ids = tokenised["input_ids"].to(torch.long)
        self.attention_mask = tokenised["attention_mask"].to(torch.long)

        wt_tokenised = tokenizer(
            [self.parent_sequence] * len(self.data),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.wt_ids = wt_tokenised["input_ids"].to(torch.long)
        self.wt_attention_mask = wt_tokenised["attention_mask"].to(torch.long)

        self.mutated_positions: List[torch.Tensor] = []
        parent_array = np.frombuffer(self.parent_sequence.encode("utf-8"), dtype="S1")
        for sequence in self.sequences:
            seq_array = np.frombuffer(sequence.encode("utf-8"), dtype="S1")
            length = min(len(parent_array), len(seq_array))
            diffs = np.where(parent_array[:length] != seq_array[:length])[0]
            if len(seq_array) > length:
                tail = np.arange(length, len(seq_array), dtype=np.int64)
                diffs = np.concatenate([diffs, tail])
            self.mutated_positions.append(torch.tensor(diffs, dtype=torch.long))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:  # pragma: no cover - simple access
        return (
            self.sequence_ids[idx],
            self.attention_mask[idx],
            self.wt_ids[idx],
            self.wt_attention_mask[idx],
            self.mutated_positions[idx],
            self.labels[idx],
            self.sample_indices[idx],
        )

    def collate_fn(self, batch: Iterable[tuple[torch.Tensor, ...]]):
        seq, mask, wt, wt_mask, pos, labels, indices = zip(*batch)
        seq = torch.stack(seq)
        mask = torch.stack(mask)
        wt = torch.stack(wt)
        wt_mask = torch.stack(wt_mask)
        pos = [torch.as_tensor(p, dtype=torch.long) for p in pos]
        labels = torch.stack(labels)
        indices = torch.stack(indices)
        return seq, mask, wt, wt_mask, pos, labels, indices

    def sequence_from_index(self, idx: int) -> GFPExample:
        """Return metadata for a given dataset index."""

        return GFPExample(
            index=idx,
            sequence=self.sequences[idx],
            fluorescence=float(self.labels[idx].item()),
        )


__all__ = [
    "GFPDataset",
    "GFPExample",
    "GFP_AMINO_ACID_VOCABULARY",
    "GFP_ALPHABET",
    "gfp_dataset_to_df",
    "get_gfp_dfs",
]
