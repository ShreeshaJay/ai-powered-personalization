import numpy as np
import pytest
import torch

from models.rqvae import RQVAE, add_collision_suffix, code_statistics, kmeans


def clustered(n_per=64, k=8, dim=16, offset=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(k, dim, generator=g) * 3 + offset
    x = torch.cat([c + 0.1 * torch.randn(n_per, dim, generator=g) for c in centres])
    return x


def test_collision_suffix_known_answer():
    codes = np.array([[1, 2], [1, 2], [3, 4], [1, 2]])
    sids, size = add_collision_suffix(codes)
    assert sids[:, -1].tolist() == [0, 1, 0, 2]
    assert size == 3
    assert len({tuple(r) for r in sids.tolist()}) == len(codes)


def test_code_statistics_known_answer():
    codes = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])
    stats = code_statistics(codes, codebook_size=4)
    assert stats["levels"][0]["codes_used"] == 2
    assert stats["levels"][1]["distinct_prefixes"] == 3
    assert stats["distinct_full_codes"] == 3
    assert stats["items_sharing_codes_with_another_item"] == pytest.approx(0.5)
    assert stats["largest_collision_group"] == 2
    assert stats["levels"][0]["perplexity"] == pytest.approx(2.0)


def test_kmeans_recovers_separated_clusters():
    x = clustered()
    centroids = kmeans(x, 8, iters=20, generator=torch.Generator().manual_seed(0))
    assign = torch.cdist(x, centroids).argmin(1)
    assert len(torch.unique(assign)) == 8


def test_codebooks_are_buffers_not_parameters():
    model = RQVAE(input_dim=16, encoder_hidden_dim=32, latent_dim=8, num_levels=2, codebook_size=4)
    names = [n for n, _ in model.named_parameters()]
    assert not any("codebook" in n for n in names)
    assert any(n.endswith("codebook") for n, _ in model.named_buffers())


def test_single_straight_through_gradient():
    """The encoder receives the decoder-input gradient once, not once per level."""
    torch.manual_seed(0)
    model = RQVAE(input_dim=16, encoder_hidden_dim=32, latent_dim=8, num_levels=3, codebook_size=4,
                  commitment_weight=0.0)
    model.eval()  # no EMA update
    captured = {}

    def keep_output(module, inputs, output):
        output.retain_grad()
        captured["z_e"] = output

    def keep_input(module, inputs):
        inputs[0].retain_grad()
        captured["dec_in"] = inputs[0]

    model.encoder.register_forward_hook(keep_output)
    model.decoder.register_forward_pre_hook(keep_input)
    out = model(torch.randn(10, 16))
    out["reconstruction_loss"].backward()
    assert torch.allclose(captured["z_e"].grad, captured["dec_in"].grad, atol=1e-7)


def unit_clusters(k=8, n_per=64, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(k, dim, generator=g)
    x = torch.cat([c + 0.15 * torch.randn(n_per, dim, generator=g) for c in centres])
    return x / x.norm(dim=1, keepdim=True)  # item embeddings are L2-normalised


def codes_used_after_training(kmeans_init: bool, dead_code_fraction: float, epochs=40, batch=64):
    torch.manual_seed(0)
    x = unit_clusters()
    model = RQVAE(16, 32, 8, num_levels=1, codebook_size=8, dead_code_fraction=dead_code_fraction)
    if kmeans_init:
        model.initialize_codebooks(x, iters=10, seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(batch):
            out = model(x[idx])
            opt.zero_grad()
            out["total_loss"].backward()
            opt.step()
    return len(np.unique(model.get_codes(x)[:, 0]))


def test_dead_code_reset_keeps_codes_alive():
    with_reset = codes_used_after_training(kmeans_init=True, dead_code_fraction=0.1)
    without_reset = codes_used_after_training(kmeans_init=True, dead_code_fraction=0.0)
    assert with_reset >= 7
    assert with_reset >= without_reset


def test_codes_are_deterministic_given_seed():
    assert codes_used_after_training(True, 0.1, epochs=3) == codes_used_after_training(True, 0.1, epochs=3)
