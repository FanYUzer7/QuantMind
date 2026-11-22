"""Mechanism checks for the next stock-relation round, independent of prices."""

import unittest

import torch

from model import StockRelationNet, peer_context


CFG = {
    "hidden_dimension": 4,
    "embedding_dimension": 3,
    "latent_groups": 2,
    "routing_temperature": 0.5,
    "routing_entropy_penalty": 0.001,
    "routing_balance_penalty": 0.001,
}


class IdentityPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.x = torch.randn(2, 4, 6)
        self.market = torch.randn(2, 2)
        self.ids = torch.tensor([0, 1, 2, 3])
        self.active = torch.ones(2, 4, dtype=torch.bool)
        self.permutation = torch.arange(4).repeat(2, 1)

    def run_model(self, model, ids=None):
        return model(
            self.x,
            self.market,
            self.ids if ids is None else ids,
            self.active,
            self.permutation,
        )[0]

    def test_state_and_uniform_paths_ignore_id_lookup(self):
        for variant in ("state_relation", "uniform_peers"):
            model = StockRelationNet(6, 2, 3, CFG, variant)
            before = self.run_model(model)
            with torch.no_grad():
                model.identity.weight.add_(20)
                model.projection.mul_(10)
            after = self.run_model(model, self.ids.flip(0))
            torch.testing.assert_close(before, after)

    def test_route_only_removes_identity_head_but_retains_routing(self):
        model = StockRelationNet(6, 2, 3, CFG, "route_only")
        before = self.run_model(model)
        start = CFG["hidden_dimension"] + self.market.shape[-1]
        with torch.no_grad():
            model.head[0].weight[:, start : start + CFG["embedding_dimension"]].add_(
                100
            )
        torch.testing.assert_close(before, self.run_model(model))
        self.run_model(model).square().sum().backward()
        self.assertGreater(float(model.projection.grad.abs().sum()), 0)

    def test_state_router_learns_without_identity_gradients(self):
        model = StockRelationNet(6, 2, 3, CFG, "state_relation")
        pred, penalty = model(
            self.x, self.market, self.ids, self.active, self.permutation
        )
        (pred.square().sum() + penalty).backward()
        self.assertIsNone(model.identity.weight.grad)
        self.assertIsNone(model.projection.grad)
        self.assertGreater(float(model.state_projection.weight.grad.abs().sum()), 0)

    def test_uniform_context_is_mean_of_other_active_states(self):
        h = torch.randn(1, 4, 3, dtype=torch.float64)
        active = torch.tensor([[True, True, True, False]])
        assignment = torch.full((1, 4, 2), 0.5, dtype=torch.float64)
        context = peer_context(h, assignment, assignment, active)
        torch.testing.assert_close(context[0, 0], h[0, 1:3].mean(0))

    def test_only_shuffle_variant_consumes_source_permutation(self):
        permuted = torch.tensor([[3, 2, 1, 0], [1, 2, 3, 0]])
        for variant in ("relation", "route_only", "state_relation", "uniform_peers"):
            model = StockRelationNet(6, 2, 3, CFG, variant)
            before = self.run_model(model)
            after, _ = model(self.x, self.market, self.ids, self.active, permuted)
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
