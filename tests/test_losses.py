from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from topoanchor.losses.anchor_distribution_loss import AnchorDistributionLoss
from topoanchor.losses.metric_loss import SupervisedContrastiveTopologyLoss
from topoanchor.losses.segmentation_losses import DiceCrossEntropyLoss


def test_segmentation_loss_accepts_multiclass_3d() -> None:
    logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4, 4))
    loss = DiceCrossEntropyLoss(include_background=False)(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_supervised_contrastive_loss_skips_empty_positives() -> None:
    z = torch.randn(3, 8, requires_grad=True)
    positive = torch.zeros(3, 3, dtype=torch.bool)
    loss = SupervisedContrastiveTopologyLoss()(z, positive)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None


def test_supervised_contrastive_loss_with_positive_pair() -> None:
    z = torch.randn(3, 8, requires_grad=True)
    positive = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    loss = SupervisedContrastiveTopologyLoss()(z, positive)
    assert torch.isfinite(loss)


def test_anchor_distribution_loss_is_finite_and_differentiable() -> None:
    z = torch.randn(2, 4, requires_grad=True)
    mu = torch.zeros(2, 4)
    var = torch.ones(2, 4)
    loss = AnchorDistributionLoss()(z, mu, var)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
