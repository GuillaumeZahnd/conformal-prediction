import torch
import math

from conformal_utils import Calibration


def print_regression_calibration_report(calibration: "Calibration") -> None:
    """Print the outcome of the calibration step."""

    print("\n" + "-"*64)
    print("CALIBRATION REPORT")
    print("-"*64)

    print(f"Calibration samples: {calibration.nb_calibration_samples}")

    print(f"alpha: {calibration.alpha}  (target coverage >= {1 - calibration.alpha:.1%})")

    if math.isinf(calibration.qhat):
        print("qhat: inf  (the calibration set is too small for this alpha: intervals are unbounded)")
    else:
        print(f"qhat: {calibration.qhat:.4f}  (in the units of the target)")
        print(f"prediction interval: prediction +/- {calibration.qhat:.4f}")
    print()


def print_regression_task_report(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Print basic regression metrics, independent of the uncertainty report.

    Args:
        predictions: Point predictions of shape (nb_samples,).
        targets: Ground-truth values of shape (nb_samples,).
    """
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = errors.pow(2).mean().sqrt().item()
    r2 = 1 - (errors.pow(2).sum() / (targets - targets.mean()).pow(2).sum()).item()

    print("\n" + "-" * 64)
    print("REGRESSION REPORT")
    print("-" * 64)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.3f}")


def print_regression_uncertainty_report(covered: torch.Tensor, calibration: "Calibration") -> None:
    """
    Print empirical coverage and interval width.

    Args:
        covered: Boolean tensor of shape (nb_samples,). True where the true value lies inside its interval.
        calibration: Result of `calibrate_regression`, providing `alpha` and `qhat`.
    """
    nb_samples = len(covered)
    nb_covered = int(covered.sum())

    print("\n" + "-" * 64)
    print("UNCERTAINTY REPORT")
    print("-" * 64)
    print(f"Target coverage:    >= {1 - calibration.alpha:.1%} (alpha = {calibration.alpha})")
    print(f"Empirical coverage: {nb_covered}/{nb_samples} ({nb_covered / nb_samples:.1%})")
    print(f"Interval width:     {2 * calibration.qhat:.4f}  (identical for every sample)")
