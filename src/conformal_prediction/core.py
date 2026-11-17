import math
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Result of the conformal calibration step.

    Attributes
    ----------
    alpha : float
        Miscoverage level used for the calibration, in ``(0, 1)``. The conformal guarantee targets
        coverage ``1 - alpha``.
    qhat : float
        Calibrated threshold: the finite-sample corrected ``1 - alpha`` quantile of the calibration scores.
        ``inf`` when the calibration set is too small for this ``alpha``.
    n_calibration_samples : int
        Number of labelled samples used for the calibration.
    """

    alpha: float
    qhat: float
    n_calibration_samples: int


def conformal_quantile(
    scores: torch.Tensor,
    alpha: float
) -> float:
    """Finite-sample corrected ``1 - alpha`` quantile of the calibration scores.

    Parameters
    ----------
    scores : torch.Tensor
        Calibration scores, shape ``(n_samples,)``. Higher means a worse fit of the label.
    alpha : float
        Miscoverage level, expected in ``(0, 1)``. Not validated here; see `calibrate`.

    Returns
    -------
    float
        The ``k``-th smallest score, with ``k = ceil((n_samples + 1) * (1 - alpha))``.
        ``inf`` if ``k > n_samples``, i.e. the calibration set is too small for this ``alpha``.
    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n_samples = len(scores)
    k = math.ceil((n_samples + 1) * (1 - alpha))
    return float("inf") if k > n_samples else float(scores.sort().values[k - 1])


def calibrate(
    calibration_scores: torch.Tensor,
    alpha: float,
) -> Calibration:
    """Compute the conformal threshold from calibration scores.

    Parameters
    ----------
    calibration_scores : torch.Tensor
        Score of the true label of each calibration sample, shape ``(n_calibration_samples,)``.
    alpha : float
        Miscoverage level, in ``(0, 1)``.

    Returns
    -------
    Calibration
        The calibration result holding ``alpha``, the threshold ``qhat`` and the sample count.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    qhat = conformal_quantile(calibration_scores, alpha)
    n_calibration_samples = len(calibration_scores)
    return Calibration(alpha, qhat, n_calibration_samples)
