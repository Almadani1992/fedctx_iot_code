"""
tests/test_model.py
====================
Unit tests for model components, federated protocol, DP, and XAI.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import torch

from models.cnn_backbone import MultiScaleCNNBackbone, DepthwiseSeparableConv1d
from models.transformer_encoder import PersonalisedTransformerEncoder
from models.fedctx_model import FedCTXModel
from privacy.dp_mechanism import DPMechanism, TopKCompressor
from federated.aggregation import fedavg_backbone, aggregate_importance
from data.partitioner import FederatedPartitioner


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_input():
    B, T, F = 4, 32, 47
    return torch.randn(B, T, F)


@pytest.fixture
def model():
    return FedCTXModel(
        in_features=47, n_classes=8,
        embedding_dim=64, cnn_kernels=[3, 5, 7],
        n_heads=4, n_layers=2, seq_len=32, rank=16,
    )


# ── CNN Backbone Tests ────────────────────────────────────────────────────

class TestCNNBackbone:

    def test_output_shape(self, sample_input):
        backbone = MultiScaleCNNBackbone(
            in_features=47, embedding_dim=64, kernels=[3, 5, 7]
        )
        out = backbone(sample_input)
        assert out.shape == (4, 32, 64), \
            f"Expected (4, 32, 64), got {out.shape}"

    def test_single_kernel(self, sample_input):
        backbone = MultiScaleCNNBackbone(
            in_features=47, embedding_dim=64, kernels=[7]
        )
        out = backbone(sample_input)
        assert out.shape == (4, 32, 64)

    def test_depthwise_separable_params(self):
        """Depthwise separable should have fewer params than standard conv."""
        dw = DepthwiseSeparableConv1d(32, 64, kernel_size=7)
        std_conv = torch.nn.Conv1d(32, 64, kernel_size=7)
        dw_params  = sum(p.numel() for p in dw.parameters())
        std_params = sum(p.numel() for p in std_conv.parameters())
        assert dw_params < std_params, \
            "Depthwise separable should use fewer parameters."

    def test_gradient_flow(self, sample_input):
        backbone = MultiScaleCNNBackbone(in_features=47, embedding_dim=64)
        sample_input.requires_grad_(True)
        out = backbone(sample_input)
        out.sum().backward()
        assert sample_input.grad is not None


# ── Transformer Tests ─────────────────────────────────────────────────────

class TestTransformerEncoder:

    def test_output_shape(self):
        x   = torch.randn(4, 32, 64)
        enc = PersonalisedTransformerEncoder(
            embed_dim=64, n_heads=4, n_layers=2, seq_len=32, rank=16
        )
        out = enc(x)
        assert out.shape == x.shape

    def test_attention_weights_stored(self):
        x   = torch.randn(2, 32, 64)
        enc = PersonalisedTransformerEncoder(
            embed_dim=64, n_heads=4, n_layers=2, seq_len=32, rank=16
        )
        enc(x)
        weights = enc.get_all_attention_weights()
        assert len(weights) == 2           # one per layer
        assert weights[0].shape[0] == 2   # batch size


# ── Full Model Tests ──────────────────────────────────────────────────────

class TestFedCTXModel:

    def test_forward_shape(self, model, sample_input):
        logits = model(sample_input)
        assert logits.shape == (4, 8)

    def test_backbone_param_separation(self, model):
        backbone_params = set(id(p) for p in model.backbone_params())
        local_params    = set(id(p) for p in model.local_params())
        assert backbone_params.isdisjoint(local_params), \
            "Backbone and local parameter sets must be disjoint."

    def test_backbone_state_dict_roundtrip(self, model):
        sd_before = {k: v.clone() for k, v in model.backbone_state_dict().items()}
        model.load_backbone_state_dict(sd_before)
        sd_after = model.backbone_state_dict()
        for k in sd_before:
            assert torch.allclose(sd_before[k], sd_after[k]), \
                f"Mismatch in backbone key: {k}"

    def test_count_parameters(self, model):
        info = model.count_parameters()
        assert info["total_params"] == info["backbone_params"] + info["local_params"]
        assert 0 < info["transmitted_pct"] < 100

    def test_embeddings_shape(self, model, sample_input):
        emb = model.get_embeddings(sample_input)
        assert emb.shape == (4, 64)   # (B, embedding_dim)


# ── Differential Privacy Tests ────────────────────────────────────────────

class TestDPMechanism:

    def test_clip_reduces_norm(self):
        dp   = DPMechanism(clip_norm=1.0, noise_multiplier=0.0)
        grads = [torch.randn(100) * 10]
        clipped = dp.clip_and_noise(grads)
        norm = clipped[0].norm().item()
        assert norm <= 1.01, f"Clipped norm {norm:.4f} exceeds clip_norm=1.0"

    def test_noise_added(self):
        dp    = DPMechanism(clip_norm=1.0, noise_multiplier=1.0)
        grads = [torch.zeros(100)]
        noisy = dp.clip_and_noise(grads)
        assert not torch.allclose(noisy[0], torch.zeros(100)), \
            "Noise should modify zero gradient."

    def test_epsilon_increases_with_steps(self):
        dp = DPMechanism(clip_norm=1.0, noise_multiplier=1.1)
        g  = [torch.randn(10)]
        dp.clip_and_noise(g)
        eps1 = dp.compute_epsilon(n_samples=1000, batch_size=64)
        dp.clip_and_noise(g)
        eps2 = dp.compute_epsilon(n_samples=1000, batch_size=64)
        assert eps2 > eps1


# ── TopK Compressor Tests ─────────────────────────────────────────────────

class TestTopKCompressor:

    def test_sparsity(self):
        comp  = TopKCompressor(sparsity=0.1)
        grads = [torch.randn(1000)]
        sparse, masks = comp.compress(grads)
        nnz = (sparse[0] != 0).float().mean().item()
        assert nnz <= 0.15, f"Non-zero ratio {nnz:.3f} exceeds expected sparsity."

    def test_error_feedback(self):
        comp = TopKCompressor(sparsity=0.1)
        g    = [torch.ones(100)]
        s1, _ = comp.compress(g)
        s2, _ = comp.compress(g)
        # Second round should include error from first
        assert s2[0].abs().sum() >= s1[0].abs().sum() * 0.9


# ── Aggregation Tests ─────────────────────────────────────────────────────

class TestAggregation:

    def test_fedavg_backbone_shape(self):
        sd = {"layer.weight": torch.randn(64, 32)}
        updates = [{"layer.weight": torch.randn(64, 32)} for _ in range(3)]
        weights = [1/3, 1/3, 1/3]
        new_sd = fedavg_backbone(updates, weights, sd)
        assert new_sd["layer.weight"].shape == (64, 32)

    def test_weights_must_sum_to_one(self):
        sd = {"w": torch.randn(10)}
        updates = [{"w": torch.randn(10)}]
        with pytest.raises(AssertionError):
            fedavg_backbone(updates, [0.5], sd)

    def test_aggregate_importance(self):
        imps    = [np.ones(47) * i for i in range(1, 4)]
        weights = [1/3, 1/3, 1/3]
        agg     = aggregate_importance(imps, weights)
        assert agg.shape == (47,)
        assert np.isclose(agg.mean(), 2.0)


# ── Partitioner Tests ─────────────────────────────────────────────────────

class TestPartitioner:

    def test_iid_sizes_balanced(self):
        X = np.random.randn(1000, 32, 47).astype(np.float32)
        y = np.random.randint(0, 8, 1000)
        p = FederatedPartitioner(n_clients=8, alpha=None)
        clients = p.partition(X, y)
        sizes = [len(c[1]) for c in clients]
        assert max(sizes) - min(sizes) <= 2

    def test_dirichlet_total_samples(self):
        X = np.random.randn(800, 32, 47).astype(np.float32)
        y = np.random.randint(0, 8, 800)
        p = FederatedPartitioner(n_clients=8, alpha=0.5)
        clients = p.partition(X, y)
        total = sum(len(c[1]) for c in clients)
        assert total == 800

    def test_distribution_matrix_shape(self):
        y = np.random.randint(0, 8, 500)
        p = FederatedPartitioner(n_clients=8, alpha=0.1)
        dist = p.get_distribution_matrix(y)
        assert dist.shape == (8, 8)
        assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-3)
