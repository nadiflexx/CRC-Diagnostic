# tests/src/training/test_mixup.py
"""Tests for mixup_data, mixup_criterion, TemperatureScaler — FIXED."""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(autouse=True)
def mock_heavy_imports():
    with (
        patch("src.training.train_image_classifier.ColonCancerClassifier"),
        patch("src.training.train_image_classifier.MLflowTracker"),
        patch("src.training.train_image_classifier.get_db"),
    ):
        yield


# ── mixup_data ────────────────────────────────────────────────────────────────


class TestMixupData:
    def _fn(self):
        from src.training.train_image_classifier import mixup_data

        return mixup_data

    def test_returns_four_elements(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1])
        assert len(fn(x, y, alpha=0.2)) == 4

    def test_mixed_x_same_shape(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1])
        mixed_x, _, _, _ = fn(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape

    def test_mixed_x_same_dtype(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1])
        mixed_x, _, _, _ = fn(x, y, alpha=0.2)
        assert mixed_x.dtype == x.dtype

    def test_y_a_equals_original_y(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 0])
        _, y_a, _, _ = fn(x, y, alpha=0.2)
        assert torch.equal(y_a, y)

    def test_lam_in_zero_one_range(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1])
        for _ in range(20):
            _, _, _, lam = fn(x, y, alpha=0.4)
            assert 0.0 <= lam <= 1.0

    def test_alpha_zero_gives_lam_one(self):
        fn = self._fn()
        x = torch.rand(4, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1])
        _, _, _, lam = fn(x, y, alpha=0)
        assert lam == 1

    def test_y_b_has_same_length_as_y(self):
        fn = self._fn()
        x = torch.rand(6, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        _, _, y_b, _ = fn(x, y, alpha=0.2)
        assert len(y_b) == len(y)

    def test_lam_sampled_from_beta(self):
        fn = self._fn()
        x = torch.rand(4, 3, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        with patch("numpy.random.beta", return_value=0.7) as mock_beta:
            _, _, _, lam = fn(x, y, alpha=0.3)
        mock_beta.assert_called_once_with(0.3, 0.3)
        assert lam == pytest.approx(0.7)

    def test_batch_size_one_does_not_crash(self):
        fn = self._fn()
        x = torch.rand(1, 3, 8, 8)
        y = torch.tensor([0])
        mixed_x, y_a, y_b, lam = fn(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape

    def test_mixed_x_is_convex_combination(self):
        """With fixed lam=1, mixed_x == x (permutation doesn't matter for lam=1)."""
        fn = self._fn()
        x = torch.rand(4, 3, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        mixed_x, _, _, lam = fn(x, y, alpha=0)
        # lam=1 → mixed = 1*x + 0*x[perm] = x
        assert lam == 1
        torch.testing.assert_close(mixed_x, x)


# ── mixup_criterion ───────────────────────────────────────────────────────────


class TestMixupCriterion:
    def _fn(self):
        from src.training.train_image_classifier import mixup_criterion

        return mixup_criterion

    def test_returns_tensor(self):
        fn = self._fn()
        criterion = MagicMock(side_effect=lambda p, t: torch.tensor(0.5))
        pred = torch.rand(4, 3)
        result = fn(
            criterion,
            pred,
            torch.tensor([0, 1, 2, 0]),
            torch.tensor([1, 0, 1, 2]),
            lam=0.6,
        )
        assert isinstance(result, torch.Tensor)

    def test_weighted_sum_formula(self):
        fn = self._fn()
        returns = [torch.tensor(1.0), torch.tensor(2.0)]
        idx = [0]

        def fake_crit(p, t):
            v = returns[idx[0]]
            idx[0] += 1
            return v

        lam = 0.3
        result = fn(
            fake_crit,
            torch.rand(2, 3),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            lam=lam,
        )
        expected = lam * 1.0 + (1 - lam) * 2.0
        assert result.item() == pytest.approx(expected, abs=1e-5)

    def test_lam_one_uses_only_ya(self):
        fn = self._fn()
        returns = [torch.tensor(0.8), torch.tensor(9.9)]
        idx = [0]

        def fake_crit(p, t):
            v = returns[idx[0]]
            idx[0] += 1
            return v

        result = fn(
            fake_crit,
            torch.rand(2, 3),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            lam=1.0,
        )
        assert result.item() == pytest.approx(0.8, abs=1e-5)

    def test_lam_zero_uses_only_yb(self):
        fn = self._fn()
        returns = [torch.tensor(9.9), torch.tensor(0.4)]
        idx = [0]

        def fake_crit(p, t):
            v = returns[idx[0]]
            idx[0] += 1
            return v

        result = fn(
            fake_crit, torch.rand(2, 3), torch.tensor([0]), torch.tensor([1]), lam=0.0
        )
        assert result.item() == pytest.approx(0.4, abs=1e-5)

    def test_criterion_called_twice(self):
        fn = self._fn()
        crit = MagicMock(return_value=torch.tensor(0.5))
        fn(crit, torch.rand(2, 3), torch.tensor([0, 1]), torch.tensor([1, 0]), lam=0.5)
        assert crit.call_count == 2


# ── TemperatureScaler ─────────────────────────────────────────────────────────


class TestTemperatureScaler:
    def _cls(self):
        from src.training.train_image_classifier import TemperatureScaler

        return TemperatureScaler

    def test_default_temperature_is_one(self):
        ts = self._cls()()
        assert ts.temperature == 1.0

    def test_scale_divides_by_temperature(self):
        ts = self._cls()()
        ts.temperature = 2.0
        logits = torch.tensor([[2.0, 4.0]])
        torch.testing.assert_close(ts.scale(logits), torch.tensor([[1.0, 2.0]]))

    def test_scale_temperature_one_is_identity(self):
        ts = self._cls()()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(ts.scale(logits), logits)

    def test_scale_temperature_half_doubles_logits(self):
        ts = self._cls()()
        ts.temperature = 0.5
        logits = torch.tensor([[1.0, 2.0]])
        torch.testing.assert_close(ts.scale(logits), torch.tensor([[2.0, 4.0]]))

    def _make_loader_and_model(self, n_classes=3, batch_size=4):
        """
        Returns a loader of real tensors and a model whose side_effect
        returns real Tensors (NOT MagicMock).
        """
        loader = [
            (torch.rand(batch_size, 3, 8, 8), torch.tensor([0, 1, 2, 0])),
            (torch.rand(batch_size, 3, 8, 8), torch.tensor([1, 2, 0, 1])),
        ]
        model = MagicMock()
        model.eval = MagicMock()
        # KEY FIX: side_effect makes the mock callable return a real Tensor
        model.side_effect = lambda imgs: torch.rand(imgs.shape[0], n_classes)
        return loader, model

    def test_fit_returns_float(self):
        ts = self._cls()()
        loader, model = self._make_loader_and_model()
        result = ts.fit(model, loader, "cpu")
        assert isinstance(result, float)
        assert 0.5 <= result < 5.0

    def test_fit_stores_best_temperature(self):
        ts = self._cls()()
        loader, model = self._make_loader_and_model()
        temp = ts.fit(model, loader, "cpu")
        assert ts.temperature == temp

    def test_fit_calls_model_eval(self):
        ts = self._cls()()
        loader, model = self._make_loader_and_model()
        ts.fit(model, loader, "cpu")
        model.eval.assert_called_once()

    def test_fit_temperature_in_valid_range(self):
        ts = self._cls()()
        loader = [(torch.rand(4, 3, 8, 8), torch.tensor([0, 0, 0, 0]))]
        model = MagicMock()
        model.eval = MagicMock()
        model.side_effect = lambda imgs: torch.tensor(
            [[10.0, -10.0, -10.0]] * imgs.shape[0]
        )
        ts.fit(model, loader, "cpu")
        assert 0.5 <= ts.temperature < 5.0

    def test_fit_selects_minimum_nll_temperature(self):
        ts = self._cls()()
        loader = [(torch.rand(4, 3, 4, 4), torch.tensor([0, 0, 0, 0]))]
        model = MagicMock()
        model.eval = MagicMock()
        model.side_effect = lambda imgs: torch.tensor(
            [[50.0, -50.0, -50.0]] * imgs.shape[0]
        )
        temp = ts.fit(model, loader, "cpu")
        assert temp <= 1.0
