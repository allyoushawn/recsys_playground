from tiger_semantic_id_amazon_beauty.src.utils import set_seed
import numpy as np
import random


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(3)
    b = [random.random() for _ in range(3)]
    set_seed(123)
    a2 = np.random.rand(3)
    b2 = [random.random() for _ in range(3)]
    assert np.allclose(a, a2)
    assert b == b2
