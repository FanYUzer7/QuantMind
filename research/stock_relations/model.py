"""Controlled stock-identity paths and state-based leave-self-out peer groups.

The original five variants preserve v1 computations. New variants isolate
direct identity use from routing, or remove categorical stock identity entirely.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn


def day_permutation(active, date, seed):
    """Return source-row assignment indexes; inactive rows remain in place."""
    indexes = np.flatnonzero(active)
    key = hashlib.sha256(f"peer-shuffle:{seed}:{date}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    result = np.arange(len(active), dtype=np.int64)
    result[indexes] = rng.permutation(indexes)
    return result


def peer_context(h, query, source, active):
    """h [T,N,H], query/source [T,N,B]; exclude each target's own h exactly."""
    weighted = source * active.unsqueeze(-1)
    mass = weighted.sum(dim=1)
    total = torch.einsum("tnb,tnh->tbh", weighted, h)
    other_mass = mass[:, None, :] - weighted
    other_total = total[:, None, :, :] - weighted[:, :, :, None] * h[:, :, None, :]
    groups = other_total / other_mass.clamp_min(1e-6).unsqueeze(-1)
    groups = torch.where(
        other_mass.unsqueeze(-1) > 1e-6, groups, torch.zeros_like(groups)
    )
    return (query.unsqueeze(-1) * groups).sum(dim=2)


class StockRelationNet(nn.Module):
    def __init__(self, input_dim, market_dim, known_stocks, cfg, variant):
        super().__init__()
        if variant not in (
            "local",
            "market",
            "identity",
            "relation",
            "shuffled",
            "route_only",
            "state_relation",
            "uniform_peers",
        ):
            raise ValueError(variant)
        self.variant, self.cfg = variant, cfg
        hidden, embed = cfg["hidden_dimension"], cfg["embedding_dimension"]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, hidden), nn.GELU()
        )
        self.identity = nn.Embedding(known_stocks + 1, embed, padding_idx=0)
        nn.init.normal_(self.identity.weight, std=0.2)
        with torch.no_grad():
            self.identity.weight[0].zero_()
        self.projection = nn.Parameter(torch.empty(embed, cfg["latent_groups"]))
        nn.init.normal_(self.projection, std=0.4)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + embed + market_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        # Append after the v1 layers so matched seed initialization stays identical.
        self.state_projection = nn.Linear(hidden, cfg["latent_groups"], bias=False)

    def assignments(self, ids):
        return torch.softmax(
            self.identity(ids) @ self.projection / self.cfg["routing_temperature"],
            dim=-1,
        )

    def forward(self, x, market, ids, active, permutation):
        h = self.encoder(x)
        e = self.identity(ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        m = market.unsqueeze(1).expand(-1, x.shape[1], -1)
        context = torch.zeros_like(h)
        regularizer = h.new_zeros(())
        if self.variant in (
            "relation",
            "shuffled",
            "route_only",
            "state_relation",
            "uniform_peers",
        ):
            if self.variant == "state_relation":
                a = torch.softmax(
                    self.state_projection(h) / self.cfg["routing_temperature"], dim=-1
                )
            elif self.variant == "uniform_peers":
                a = h.new_full(
                    (*h.shape[:2], self.cfg["latent_groups"]),
                    1 / self.cfg["latent_groups"],
                )
            else:
                a = self.assignments(ids).unsqueeze(0).expand(x.shape[0], -1, -1)
            source = (
                torch.gather(
                    a, 1, permutation.unsqueeze(-1).expand(-1, -1, a.shape[-1])
                )
                if self.variant == "shuffled"
                else a
            )
            context = peer_context(h, a, source, active)
            # Only present, known stocks enter the routing penalties.
            known = (
                active
                if self.variant in ("state_relation", "uniform_peers")
                else active * ids.ne(0).unsqueeze(0)
            )
            norm = known.sum().clamp_min(1)
            entropy = (
                (-(a.clamp_min(1e-9).log() * a).sum(-1) * known).sum()
                / norm
                / math.log(a.shape[-1])
            )
            usage = (a * known.unsqueeze(-1)).sum((0, 1)) / norm
            balance = (usage * (usage.clamp_min(1e-9) * a.shape[-1]).log()).sum()
            regularizer = (
                self.cfg["routing_entropy_penalty"] * entropy
                + self.cfg["routing_balance_penalty"] * balance
            )
        if self.variant == "local":
            m = torch.zeros_like(m)
        if self.variant in (
            "local",
            "market",
            "route_only",
            "state_relation",
            "uniform_peers",
        ):
            e = torch.zeros_like(e)
        return self.head(torch.cat([h, m, e, context], dim=-1)).squeeze(-1), regularizer


def equal_date_mse(prediction, target, observed):
    count = observed.sum(dim=1)
    daily = ((prediction - target).square() * observed).sum(dim=1) / count.clamp_min(1)
    return daily[count > 0].mean()
