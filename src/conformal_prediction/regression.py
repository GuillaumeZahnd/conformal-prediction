import torch
from dataclasses import dataclass

from .core import Calibration, conformal_quantile


@dataclass(frozen=True)
class RegressionUncertainty:
    alpha: float
    nb_samples: int
    nb_covered: int
    coverage: float
    interval_width: float  # 2 * qhat: identical for every sample


@torch.no_grad()
def predict_values(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Point predictions, shape (N,), float64, on CPU."""
    model.eval()
    device = next(model.parameters()).device
    out = [
        model(x[i : i + batch_size].to(device)).double().reshape(-1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out)


def calibrate_regression(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, alpha: float) -> Calibration:
    """Compute the half-width of the conformal intervals from held-out labelled data (absolute-error score)."""

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    predictions = predict_values(model, x)
    calibration_scores = (y.double() - predictions).abs()

    qhat = conformal_quantile(calibration_scores, alpha)
    calibration = Calibration(alpha, qhat, len(x))

    return calibration


def evaluate_regression(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    calibration: Calibration,
) -> RegressionUncertainty:
    """Build prediction intervals on `test` and measure coverage and width."""

    predictions = predict_values(model, x)
    lower, upper = predictions - calibration.qhat, predictions + calibration.qhat
    targets = y.double()
    covered = (targets >= lower) & (targets <= upper)

    nb_samples, nb_covered = len(y), int(covered.sum())

    regression_uncertainty =  RegressionUncertainty(
        alpha=calibration.alpha,
        nb_samples=nb_samples,
        nb_covered=nb_covered,
        coverage=nb_covered / nb_samples,
        interval_width=2 * calibration.qhat,
    )

    return regression_uncertainty
