"""Tests for the stock-identity diagnostic, without data or checkpoints."""

import unittest

import numpy as np
import torch

from diagnose_stock_identity import (
    apply_known_means,
    equal_date_mse_rows,
    forward_parts,
    leave_one_day_forecasts,
    weighted_stock_means,
)
from model import StockRelationNet, equal_date_mse


CFG = {
    "hidden_dimension": 4,
    "embedding_dimension": 3,
    "latent_groups": 2,
    "routing_temperature": 0.5,
    "routing_entropy_penalty": 0.001,
    "routing_balance_penalty": 0.001,
}


def readable_model(variant):
    torch.manual_seed(0)
    model = StockRelationNet(6, 2, 4, CFG, variant)
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.encoder[0].weight[:4, :4] = torch.eye(4)
        model.encoder[2].weight.zero_()
        model.encoder[2].bias.zero_()
        model.encoder[2].weight[:, :4] = torch.eye(4)
        model.identity.weight.zero_()
        model.identity.weight[1] = torch.tensor([2.0, 0.0, 0.0])
        model.identity.weight[2] = torch.tensor([0.0, 2.0, 0.0])
        model.identity.weight[3] = torch.tensor([0.0, 0.0, 1.0])
        model.projection.zero_()
        model.projection[0, 0] = 4.0
        model.projection[1, 1] = 4.0
        model.head[0].weight.zero_()
        model.head[0].bias.zero_()
        model.head[0].weight[0, 6:9] = 1.0
        model.head[0].weight[1, 9:13] = 1.0
        model.head[2].weight.zero_()
        model.head[2].bias.zero_()
        model.head[2].weight[0, 0] = 1.0
        model.head[2].weight[0, 1] = 1.0
    return model


class IdentityDiagnosticTests(unittest.TestCase):
    def test_weighted_mean_matches_closed_form_and_not_unweighted_mean(self):
        target = np.array(
            [[1.0, 0.0, 5.0], [0.0, 9.0, 9.0], [0.4, 0.0, 0.0]], dtype=np.float64
        )
        observed = np.array(
            [[True, True, False], [True, False, False], [True, True, True]]
        )
        contaminated = target.copy()
        contaminated[~observed] = 99.0
        means, weight_sum, days, _ = weighted_stock_means(contaminated, observed)
        np.testing.assert_allclose(means, [3.8 / 11, 0.0, 0.0])
        np.testing.assert_allclose(
            weight_sum, [0.5 + 1.0 + 1.0 / 3.0, 0.5 + 1.0 / 3.0, 1.0 / 3.0]
        )
        np.testing.assert_array_equal(days, [3, 2, 1])
        unweighted = target[observed[:, 0], 0].mean()
        self.assertGreater(abs(unweighted - means[0]), 0.1)

    def test_gradient_is_zero_only_at_weighted_mean(self):
        target = torch.tensor(
            [[1.0, 0.0, 5.0], [0.0, 9.0, 9.0], [0.4, 0.0, 0.0]], dtype=torch.float64
        )
        observed = torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True]]
        )
        means, _, _, _ = weighted_stock_means(target.numpy(), observed.numpy())
        mu = torch.tensor(means, dtype=torch.float64, requires_grad=True)
        equal_date_mse(mu.expand_as(target), target, observed).backward()
        self.assertLess(float(mu.grad.abs().max()), 1e-8)
        counts = observed.sum(dim=0)
        unweighted = (
            torch.where(observed, target, torch.zeros_like(target)).sum(dim=0) / counts
        )
        mu_unweighted = unweighted.detach().clone().requires_grad_(True)
        equal_date_mse(mu_unweighted.expand_as(target), target, observed).backward()
        self.assertGreater(float(mu_unweighted.grad.abs().max()), 1e-4)

    def test_unknown_identity_and_missing_mean_forecast_zero(self):
        means = np.array([0.4, np.nan, -0.2, 0.1])
        ids = np.array([0, 2, 3, 0])
        np.testing.assert_array_equal(
            apply_known_means(means, ids), [0.0, 0.0, -0.2, 0.0]
        )

    def test_leave_one_day_removes_only_that_stock_day(self):
        target = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float64
        )
        observed = np.array(
            [[True, True, False], [True, False, False], [True, True, True]]
        )
        ids = np.array([1, 2, 3])
        forecast = leave_one_day_forecasts(target, observed, ids)
        np.testing.assert_allclose(forecast[:, 0], [0.1, 0.76, 1.0 / 3.0])
        self.assertEqual(float(forecast[2, 2]), 0.0)
        ids[0] = 0
        hidden = leave_one_day_forecasts(target, observed, ids)
        np.testing.assert_array_equal(hidden[:, 0], 0.0)
        mse, _ = equal_date_mse_rows(forecast, target, observed, np.array([0, 1, 2]))
        self.assertGreater(mse, 0.0)

    def test_original_forward_matches_module_and_does_not_mutate_embedding(self):
        x = torch.randn(2, 4, 6)
        market = torch.randn(2, 2)
        ids = torch.tensor([0, 1, 2, 3])
        active = torch.tensor([[True, True, True, False], [True, True, False, True]])
        permutation = torch.tensor([[0, 2, 1, 3], [3, 0, 2, 1]])
        for variant in ("local", "market", "identity", "relation", "shuffled"):
            model = readable_model(variant)
            before = model.identity.weight.detach().clone()
            with torch.no_grad():
                expected, _ = model(x, market, ids, active, permutation)
                parts = forward_parts(model, x, market, ids, active, permutation)
            torch.testing.assert_close(parts["prediction"], expected)
            torch.testing.assert_close(model.identity.weight, before)

    def test_interventions_edit_only_their_own_path(self):
        x = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ],
            ]
        )
        market = torch.zeros(1, 2)
        ids = torch.tensor([0, 1, 2, 3])
        active = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([[1, 2, 3, 0]])
        model = readable_model("relation")
        with torch.no_grad():
            original = forward_parts(model, x, market, ids, active, permutation)
            zeroed = forward_parts(
                model, x, market, ids, active, permutation, zero_direct_identity=True
            )
            uniform = forward_parts(
                model, x, market, ids, active, permutation, uniform_routing=True
            )
            both = forward_parts(
                model,
                x,
                market,
                ids,
                active,
                permutation,
                zero_direct_identity=True,
                uniform_routing=True,
            )
        torch.testing.assert_close(zeroed["context"], original["context"])
        torch.testing.assert_close(
            zeroed["query_assignment"], original["query_assignment"]
        )
        torch.testing.assert_close(
            zeroed["identity_to_head"], torch.zeros_like(zeroed["identity_to_head"])
        )
        torch.testing.assert_close(
            uniform["identity_to_head"], original["identity_to_head"]
        )
        torch.testing.assert_close(
            uniform["query_assignment"],
            torch.full_like(uniform["query_assignment"], 0.5),
        )
        self.assertGreater(
            float((uniform["context"] - original["context"]).abs().max()), 1e-4
        )
        torch.testing.assert_close(both["context"], uniform["context"])
        torch.testing.assert_close(
            both["identity_to_head"], torch.zeros_like(both["identity_to_head"])
        )
        torch.testing.assert_close(
            zeroed["prediction"][:, 0], original["prediction"][:, 0]
        )
        self.assertGreater(
            float(
                (zeroed["prediction"][:, 1] - original["prediction"][:, 1]).abs().max()
            ),
            1e-4,
        )

    def test_uniform_routing_preserves_leave_self_out_and_is_inert_without_routing(
        self,
    ):
        x = torch.randn(1, 4, 6)
        changed = x.clone()
        changed[:, 0] += 3.0
        market = torch.randn(1, 2)
        ids = torch.arange(1, 5)
        active = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([[3, 2, 0, 1]])
        relation = readable_model("relation")
        with torch.no_grad():
            base = forward_parts(
                relation, x, market, ids, active, permutation, uniform_routing=True
            )
            moved = forward_parts(
                relation,
                changed,
                market,
                ids,
                active,
                permutation,
                uniform_routing=True,
            )
        torch.testing.assert_close(
            moved["context"][:, 0], base["context"][:, 0], atol=1e-4, rtol=1e-4
        )
        self.assertGreater(
            float((moved["context"][:, 1] - base["context"][:, 1]).abs().max()), 1e-5
        )
        identity = readable_model("identity")
        with torch.no_grad():
            plain = forward_parts(identity, x, market, ids, active, permutation)
            routed = forward_parts(
                identity, x, market, ids, active, permutation, uniform_routing=True
            )
        torch.testing.assert_close(routed["prediction"], plain["prediction"])
        self.assertIsNone(routed["query_assignment"])

    def test_uniform_source_permutation_does_not_change_context(self):
        x = torch.tensor(
            [
                [
                    [0.2, 1.0, -0.4, 0.6, 0.0, 0.1],
                    [-0.5, 0.3, 1.2, -0.2, 0.4, 0.0],
                    [0.8, -1.0, 0.1, 0.7, -0.3, 0.2],
                    [0.0, 0.4, -0.6, 1.1, 0.5, -0.1],
                ],
                [
                    [1.0, 0.0, 0.2, -0.4, 0.3, 0.0],
                    [0.1, -0.7, 0.5, 0.8, -0.2, 0.4],
                    [-0.3, 0.9, -0.1, 0.2, 0.6, -0.5],
                    [0.4, 0.2, 0.7, -0.8, 0.1, 0.3],
                ],
            ]
        )
        market = torch.zeros(2, 2)
        ids = torch.arange(1, 5)
        active = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([[1, 2, 3, 0], [2, 0, 3, 1]])
        model = readable_model("relation")
        with torch.no_grad():
            relation = forward_parts(
                model, x, market, ids, active, permutation, uniform_routing=True
            )
            model.variant = "shuffled"
            shuffled = forward_parts(
                model, x, market, ids, active, permutation, uniform_routing=True
            )
            learned = forward_parts(
                model, x, market, ids, active, permutation, uniform_routing=False
            )
        torch.testing.assert_close(shuffled["context"], relation["context"])
        self.assertGreater(
            float((learned["context"] - shuffled["context"]).abs().max()), 1e-4
        )


if __name__ == "__main__":
    unittest.main()
