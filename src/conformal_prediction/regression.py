import torch
from dataclasses import dataclass

from .core import calibrate, Calibration


@dataclass(frozen=True)
class RegressionUncertainty:
    """Result of a conformal regression evaluation on a labelled test set.

    Attributes
    ----------
    alpha : float
        Miscoverage level used for the calibration. The conformal guarantee targets coverage ``1 - alpha``.
    nb_samples : int
        Number of evaluated test samples.
    nb_covered : int
        Number of samples whose true value lies inside its prediction interval.
    coverage : float
        Empirical coverage, ``nb_covered / nb_samples``.
    interval_width : float
        Full width of the prediction intervals, ``2 * qhat``, in the units of the target.
        Identical for every sample, since the absolute-error score gives a constant half-width.
    """
    alpha: float
    nb_samples: int
    nb_covered: int
    coverage: float
    interval_width: float


@torch.no_grad()
def predict_values(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 256
) -> torch.Tensor:
    """Point predictions, shape (N,), float64, on CPU.

    Parameters
    ----------
    model : torch.nn.Module
        The regression model.
    x : torch.Tensor
        Input features tensor of shape ``(N, ...)``.
    batch_size : int, optional
        Batch size for inference.

    Returns
    -------
    torch.Tensor
        Point predictions tensor of shape ``(N,)`` on CPU in float64.
    """
    model.eval()
    device = next(model.parameters()).device
    out = [
        model(x[i : i + batch_size].to(device)).double().reshape(-1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out)


def calibrate_regression(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Calibration:
    """Compute the half-width of the conformal intervals from held-out labelled data (absolute-error score).

    Parameters
    ----------
    model : torch.nn.Module
        The regression model.
    x : torch.Tensor
        Calibration input tensor.
    y : torch.Tensor
        Calibration true target tensor.
    alpha : float
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    Calibration
        The resulting calibration object with computed threshold ``qhat``.
    """
    calibration_scores = compute_calibration_scores_regression(model, x, y)
    return calibrate(calibration_scores, alpha)


def compute_calibration_scores_regression(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute absolute error conformity scores for regression calibration.

    Parameters
    ----------
    model : torch.nn.Module
        The regression model.
    x : torch.Tensor
        Input features tensor.
    y : torch.Tensor
        True target tensor.

    Returns
    -------
    torch.Tensor
        Calibration scores tensor of shape ``(n_calibration_samples,)``.
    """
    predictions = predict_values(model, x)
    calibration_scores = (y.double() - predictions).abs()
    return calibration_scores


def evaluate_regression(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    calibration: Calibration,
) -> RegressionUncertainty:
    """Build prediction intervals on `test` and measure coverage and width.

    Parameters
    ----------
    model : torch.nn.Module
        The regression model.
    x : torch.Tensor
        Test input features tensor.
    y : torch.Tensor
        Test true target tensor.
    calibration : Calibration
        Fitted calibration object containing ``qhat``.

    Returns
    -------
    RegressionUncertainty
        Evaluation results including coverage and interval width.
    """

    predictions = predict_values(model, x)
    lower = predictions - calibration.qhat
    upper = predictions + calibration.qhat
    targets = y.double()
    covered = (targets >= lower) & (targets <= upper)

    nb_samples = len(y)
    nb_covered = int(covered.sum())

    regression_uncertainty =  RegressionUncertainty(
        alpha=calibration.alpha,
        nb_samples=nb_samples,
        nb_covered=nb_covered,
        coverage=nb_covered / nb_samples,
        interval_width=2 * calibration.qhat,
    )

    return regression_uncertainty
