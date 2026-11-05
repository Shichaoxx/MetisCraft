"""Batch records into tensors for DecisionModel.

All sequences are left-padded and ``position_ids`` are derived from the attention mask,
so padding never shifts positions. Batches may mix judgment types.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torch

from metiscraft.config import ReadoutConfig
from metiscraft.data.contract import Record
from metiscraft.judgments import answer_index, target_vector
from metiscraft.modeling.prompts import Encoded, PromptBuilder, choices_of


def left_pad(seqs: Sequence[list[int]], pad_id: int):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    offsets = []
    for i, seq in enumerate(seqs):
        off = width - len(seq)
        ids[i, off:] = torch.tensor(seq, dtype=torch.long)
        mask[i, off:] = 1
        offsets.append(off)
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, position_ids, offsets


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class Collator:
    def __init__(
        self,
        tokenizer,
        readout: ReadoutConfig,
        max_length: int,
        *,
        train: bool = False,
        seed: int = 0,
        ordinal_sigma: float | None = None,
        completion: bool = False,
    ) -> None:
        self.builder = PromptBuilder(tokenizer, max_length, readout.prompt)
        self.readout = readout
        self.train = train
        self.rng = random.Random(seed)
        self.ordinal_sigma = ordinal_sigma
        # completion training appends the answer code; the model then predicts [code, EOS]
        self.completion = completion
        self.eos_id = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    def encode(self, record: Record) -> Encoded:
        r = self.readout
        if r.kind == "answer_token":
            return self.builder.answer_token(record)
        if r.layout == "pairs":
            return self.builder.candidate_pairs(record)
        n = len(choices_of(record))
        order = None
        if self.train and r.shuffle_train and n > 1:
            order = list(range(n))
            self.rng.shuffle(order)
        return self.builder.candidate_list(record, order)

    def __call__(self, rows: Sequence[dict[str, Any] | Record]) -> dict[str, Any]:
        records = [r if isinstance(r, Record) else Record.model_validate(r) for r in rows]
        encoded = [self.encode(r) for r in records]
        sizes = [r.target_size() for r in records]
        n, k = len(records), max(sizes)
        mask = torch.zeros((n, k), dtype=torch.bool)
        for i, size in enumerate(sizes):
            mask[i, :size] = True
        batch: dict[str, Any] = {
            "mask": mask,
            "kinds": [r.question.type for r in records],
            "ids": [r.id for r in records],
        }
        if self.readout.kind == "candidate_scalar" and self.readout.layout == "pairs":
            seqs, rows_, cols = [], [], []
            for i, enc in enumerate(encoded):
                for j, seq in enumerate(enc.pairs):
                    seqs.append(seq)
                    rows_.append(i)
                    cols.append(j)
            ids, am, pos, _ = left_pad(seqs, self.pad_id)
            batch.update(
                input_ids=ids,
                attention_mask=am,
                position_ids=pos,
                pair_row=torch.tensor(rows_, dtype=torch.long),
                pair_col=torch.tensor(cols, dtype=torch.long),
            )
        else:
            seqs = [e.input_ids for e in encoded]
            if self.completion:
                answers = [e.codes[answer_index(r)] for r, e in zip(records, encoded, strict=True)]
                seqs = [s + [a] for s, a in zip(seqs, answers, strict=True)]
                batch["answer_ids"] = torch.tensor([[a, self.eos_id] for a in answers])
            ids, am, pos, offsets = left_pad(seqs, self.pad_id)
            batch.update(input_ids=ids, attention_mask=am, position_ids=pos)
            if self.readout.kind == "answer_token":
                width = max(len(e.codes) for e in encoded)
                codes = torch.zeros((n, width), dtype=torch.long)
                for i, enc in enumerate(encoded):
                    codes[i, : len(enc.codes)] = torch.tensor(enc.codes, dtype=torch.long)
                batch["codes"] = codes
                batch["noul_rows"] = torch.tensor([e.kind == "noul" for e in encoded])
            else:
                positions = torch.zeros((n, k), dtype=torch.long)
                for i, (enc, off) in enumerate(zip(encoded, offsets, strict=True)):
                    positions[i, : len(enc.positions)] = torch.tensor(enc.positions) + off
                batch["positions"] = positions
        if all(r.supervision is not None for r in records):
            targets = torch.zeros((n, k))
            for i, r in enumerate(records):
                vec = target_vector(r, self.ordinal_sigma)
                targets[i, : len(vec)] = torch.tensor(vec)
            batch["targets"] = targets
        return batch
