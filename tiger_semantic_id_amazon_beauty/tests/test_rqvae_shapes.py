import torch

from tiger_semantic_id_amazon_beauty.src.rqvae import RQVAE, RQVAEConfig, encode_codes


def test_rqvae_forward_shapes():
    cfg = RQVAEConfig(input_dim=16, latent_dim=8, levels=3, codebook_size=16)
    model = RQVAE(cfg)
    x = torch.randn(64, 16)
    x_hat, loss, recon, codes = model(x)
    assert x_hat.shape == x.shape
    assert codes.shape == (64, cfg.levels)
    assert loss.item() >= 0


def test_encode_codes_range():
    cfg = RQVAEConfig(input_dim=16, latent_dim=8, levels=3, codebook_size=8)
    model = RQVAE(cfg)
    x = torch.randn(32, 16)
    codes = encode_codes(model, x)
    assert codes.shape == (32, cfg.levels)
    assert int(codes.max()) < cfg.codebook_size
    assert int(codes.min()) >= 0

