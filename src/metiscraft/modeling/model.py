"""DecisionModel: a backbone plus one readout (design §6).

Shapes: B batch, T sequence, d hidden, V vocabulary, K targets, R pair sequences.
"""

from __future__ import annotations

import torch
from torch import nn

from metiscraft.config import ReadoutConfig


class ScalarHead(nn.Module):
    """h_k [.., d] -> s_k [..]; kept in fp32 and saved separately from the adapter."""

    def __init__(self, hidden_size: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        nn.init.normal_(self.net[3].weight, std=0.02)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float()).squeeze(-1)


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, readout: ReadoutConfig, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = readout
        self.head = (
            ScalarHead(hidden_size, readout.head.hidden, readout.head.dropout)
            if readout.kind == "candidate_scalar"
            else None
        )

    def hidden(self, input_ids, attention_mask, position_ids) -> torch.Tensor:
        decoder = self.backbone.get_decoder()
        out = decoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return out.last_hidden_state  # [B, T, d], after the final norm

    def forward(self, batch: dict) -> torch.Tensor:
        """Raw scores [B, K]; padded columns are meaningless and masked by the caller."""
        if self.readout.kind == "answer_token":
            h_last = self.hidden(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"]
            )[:, -1]
            logits = self.backbone.get_output_embeddings()(h_last).float()  # [B, V]
            z = logits.gather(1, batch["codes"])  # [B, K]
            noul = batch["noul_rows"]
            if noul.any():
                z = z.clone()
                z[noul, 0] = z[noul, 0] - z[noul, 1]  # logit of yes vs no
            return z[:, : batch["mask"].size(1)]
        if self.readout.layout == "pairs" and "pair_row" in batch:
            h = self.hidden(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
            s = self.head(h[:, -1])  # [R]
            scores = s.new_zeros(batch["mask"].shape)
            scores[batch["pair_row"], batch["pair_col"]] = s
            return scores
        h = self.hidden(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
        idx = batch["positions"].unsqueeze(-1).expand(-1, -1, h.size(-1))  # [B, K, d]
        return self.head(h.gather(1, idx))  # [B, K]
