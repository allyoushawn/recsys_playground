"""Local coverage for the decoupled Ali-CCP dataset layer (E1).

Validates (a) the data layer imports without pulling torch, and (b) the encode
path turns a normalized arrow batch into well-formed tensors — without needing
the real Ali-CCP data (uses a tiny synthetic table). The full real-data path is
exercised by `datasets/aliccp/smoke.py` on Colab.
"""

import subprocess
import sys

import pytest

pa = pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")

from datasets.aliccp.data import ALICCP_DENSE_FEAT_COLS, ALICCP_SPARSE_COLS
from datasets.aliccp.encode import _precompute_sparse_encode_tables, encode_and_tensorize_arrow


def test_data_layer_is_torch_free():
    """Importing the data layer must NOT import torch (decoupling guarantee)."""
    code = "import sys; import datasets.aliccp.data as d; print('torch' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False", f"torch leaked into datasets.aliccp.data (got {out!r})"


def test_encode_arrow_shapes_and_dtypes():
    sparse, densef = ALICCP_SPARSE_COLS, ALICCP_DENSE_FEAT_COLS
    n = 16
    # tiny synthetic normalized batch: sparse as strings, dense as floats, labels.
    data = {c: [f"v{(i + j) % 5}" for i in range(n)] for j, c in enumerate(sparse)}
    for j, c in enumerate(densef):
        data[c] = [float(i + j) * 0.1 for i in range(n)]
    data["click"] = [i % 2 for i in range(n)]
    data["purchase"] = [0 for _ in range(n)]
    tbl = pa.table(data)

    # vocab: map each observed sparse value to a 1-based id (0 reserved UNK).
    vocabs = {c: {f"v{k}": k + 1 for k in range(5)} for c in sparse}
    enc = _precompute_sparse_encode_tables(vocabs, sparse)

    sparse_t, dense_t, label_t = encode_and_tensorize_arrow(tbl, enc, sparse, densef, "click")
    assert sparse_t.shape == (n, len(sparse))
    assert dense_t.shape == (n, len(densef))
    assert label_t.shape == (n,)
    assert sparse_t.dtype == torch.int32
    assert torch.isfinite(dense_t).all()
    assert int(sparse_t.min()) >= 0


def test_encode_maps_unknown_to_zero():
    sparse = ALICCP_SPARSE_COLS
    densef = ALICCP_DENSE_FEAT_COLS
    data = {c: ["UNSEEN"] for c in sparse}
    for c in densef:
        data[c] = [1.0]
    data["click"] = [1]
    tbl = pa.table(data)
    vocabs = {c: {"known": 1} for c in sparse}  # "UNSEEN" not in vocab -> 0
    enc = _precompute_sparse_encode_tables(vocabs, sparse)
    sparse_t, _, _ = encode_and_tensorize_arrow(tbl, enc, sparse, densef, "click")
    assert int(sparse_t.max()) == 0  # all unknown -> UNK id 0
