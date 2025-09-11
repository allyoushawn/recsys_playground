import torch

from tiger_semantic_id_amazon_beauty.src.seq2seq import (
    TIGERSeqDataset,
    VocabConfig,
    Seq2SeqConfig,
    TinyTransformer,
    collate_batch,
)


def test_dataset_and_collate():
    user_hist = {0: [0, 1, 2], 1: [3, 4]}
    # 5 items, each with 4 codes (0..2 small), codes need to be < codebook_size
    sid = torch.tensor(
        [
            [0, 1, 2, 0],
            [1, 2, 3, 0],
            [2, 3, 4, 0],
            [3, 4, 5, 1],
            [4, 5, 6, 0],
        ]
    ).numpy()
    ds = TIGERSeqDataset(user_hist, sid, user_hash_size=50, codebook_size=256, max_hist_len=5)
    assert len(ds) >= 1
    src, tgt = collate_batch([ds[0]])
    assert src.ndim == 2 and tgt.ndim == 2
    assert src.size(0) == 1 and tgt.size(0) == 1


def test_tiny_transformer_forward():
    V = 2048
    model = TinyTransformer(vocab_size=V, d_model=32, ff=64, heads=4, layers_enc=1, layers_dec=1, dropout=0.0)
    src = torch.randint(0, V, (2, 7))  # [B,S]
    tgt = torch.randint(0, V, (2, 5))  # [B,T]
    logits = model(src, tgt)
    assert logits.shape == (2, 5, V)
