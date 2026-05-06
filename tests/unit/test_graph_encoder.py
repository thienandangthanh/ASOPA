"""Unit tests for the graph attention encoder building blocks. Light-weight
shape and norm sanity tests; full numeric reproduction is covered by
test_attention_forward (loaded checkpoint)."""

from __future__ import annotations

import torch


def test_multi_head_attention_preserves_shape(fixed_seed):
    from attention_model.graph_encoder import MultiHeadAttention

    n_heads, dim = 8, 128
    mha = MultiHeadAttention(n_heads, input_dim=dim, embed_dim=dim)
    x = torch.randn(2, 7, dim)

    out = mha(x)
    assert out.shape == (2, 7, dim)


def test_skip_connection_adds_residual(fixed_seed):
    from attention_model.graph_encoder import SkipConnection
    from torch import nn

    inner = nn.Linear(4, 4)
    skip = SkipConnection(inner)
    x = torch.randn(3, 5, 4)

    out = skip(x)
    expected = x + inner(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_graph_attention_encoder_returns_node_and_graph_embeddings(fixed_seed):
    from attention_model.graph_encoder import GraphAttentionEncoder

    enc = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=2, normalization="batch")
    x = torch.randn(2, 6, 128)

    node_embed, graph_embed = enc(x)
    assert node_embed.shape == (2, 6, 128)
    # Graph embed is the mean over nodes per batch.
    assert graph_embed.shape == (2, 128)
    assert torch.allclose(graph_embed, node_embed.mean(dim=1), atol=1e-6)
