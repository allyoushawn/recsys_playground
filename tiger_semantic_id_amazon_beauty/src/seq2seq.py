from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class VocabConfig:
    codebook_size: int = 256
    levels: int = 4  # including collision code
    user_vocab_hash: int = 2000

    @property
    def semantic_vocab(self) -> int:
        return self.codebook_size * self.levels


@dataclass
class Seq2SeqConfig:
    d_model: int = 128
    ff: int = 1024
    heads: int = 6
    layers_enc: int = 4
    layers_dec: int = 4
    dropout: float = 0.1
    max_hist_len: int = 20
    batch_size: int = 256
    lr: float = 1e-2


def sid_to_tokens(sid: np.ndarray, codebook_size: int) -> List[int]:
    # Encode (c1,c2,c3,c4) to flat tokens using pos*codebook_size + code
    return [int(p) * codebook_size + int(v) for p, v in enumerate(sid.tolist())]


class TIGERSeqDataset(Dataset):
    def __init__(
        self,
        user_histories: Dict[int, List[int]],  # user_idx -> list of item_idx (train sequences)
        item_sids: np.ndarray,  # [num_items, 4]
        user_hash_size: int = 2000,
        codebook_size: int = 256,
        max_hist_len: int = 20,
    ) -> None:
        self.samples: List[Tuple[int, List[int], List[int]]] = []
        for u, items in user_histories.items():
            if len(items) < 2:
                continue
            # create (input_sequence, target_sid_tokens)
            # Use all but last as input; last as target
            hist = items[-max_hist_len:]
            inp_items = hist[:-1]
            tgt_item = hist[-1]
            user_tok = (u % user_hash_size) + 1  # reserve 0 for PAD
            seq: List[int] = [user_tok]
            for it in inp_items:
                seq.extend(sid_to_tokens(item_sids[it], codebook_size))
            tgt = sid_to_tokens(item_sids[tgt_item], codebook_size)
            self.samples.append((user_tok, seq, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch, pad_id: int = 0):
    users, seqs, tgts = zip(*batch)
    max_len = max(len(s) for s in seqs)
    max_t = max(len(t) for t in tgts)  # expect 4
    B = len(seqs)
    src = torch.full((B, max_len), pad_id, dtype=torch.long)
    tgt = torch.full((B, max_t + 1), pad_id, dtype=torch.long)  # +1 for BOS
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, 0] = 1  # BOS=1
        tgt[i, 1 : 1 + len(t)] = torch.tensor(t, dtype=torch.long)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len,1,d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, ff: int, heads: int, layers_enc: int, layers_dec: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=layers_enc,
            num_decoder_layers=layers_dec,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=False,
        )
        self.lm = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        # Shapes: src [B,S], tgt [B,T]
        src = self.embed(src_tokens).transpose(0, 1)  # [S,B,D]
        tgt = self.embed(tgt_tokens).transpose(0, 1)  # [T,B,D]
        src = self.pos(src)
        tgt = self.pos(tgt)
        T = tgt.shape[0]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)  # [B,T,D]
        logits = self.lm(out)  # [B,T,V]
        return logits


def recall_at_k(gt: List[int], pred: List[int], k: int) -> float:
    s = set(gt)
    return sum(1 for x in pred[:k] if x in s) / max(1, min(len(s), k))


def dcg_at_k(rel: List[int], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel[:k]))


def ndcg_at_k(gt_ranked_items: List[int], pred_items: List[int], k: int) -> float:
    # Binary relevance: 1 if in gt set
    gt = set(gt_ranked_items)
    rel = [1 if x in gt else 0 for x in pred_items[:k]]
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg / idcg

