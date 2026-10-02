import math

from .core import Calibration
from .classification import ClassificationUncertainty
from .regression import RegressionUncertainty


def format_classification_uncertainty(u: ClassificationUncertainty) -> str:
    """Render the uncertainty summary as text."""
    lines = [
        "-" * 64,
        "UNCERTAINTY REPORT",
        "-" * 64,
        f"Coverage (true label in prediction set): {u.nb_covered}/{u.nb_samples} ({u.coverage:.1%})",
        "",
        f"Set size summary: "
        f"mean {u.mean_set_size:.2f}, "
        f"median {u.median_set_size}, "
        f"90% of samples in [{u.set_size_q05}, {u.set_size_q95}], "
        f"range [{u.min_set_size}, {u.max_set_size}]",
        "",
        f"{'set size':>8} | {'count':>7} | {'frequency':>9} | {'coverage':>8}",
    ]
    # Most frequent first; sorted() is stable, so ties keep the smaller set size first
    for n in sorted(range(len(u.set_size_frequencies)), key=lambda n: u.set_size_frequencies[n], reverse=True):
        if u.set_size_counts[n] == 0:
            break  # Skip cases that do not occur
        lines.append(
            f"{n:>8} | "
            f"{u.set_size_counts[n]:>7} | "
            f"{u.set_size_frequencies[n]:>9.3f} | "
            f"{u.covered_per_set_size[n] / u.set_size_counts[n]:>8.3f}"
        )
    return "\n".join(lines)


def format_classification_calibration(calibration: Calibration) -> str:
    """Render the outcome of the calibration step as text."""
    lines = [
        "-" * 64,
        "CALIBRATION REPORT",
        "-" * 64,
        f"Calibration samples: {calibration.nb_calibration_samples}",
        f"alpha: {calibration.alpha}  (target coverage >= {1 - calibration.alpha:.1%})",
    ]
    if math.isinf(calibration.qhat):
        lines.append("qhat: inf (the calibration set is too small for this alpha: every class is included)")
    else:
        lines += [
            f"qhat: {calibration.qhat:.4f}",
            f"A class is included in the prediction set if its softmax probability is >= {1 - calibration.qhat:.4f}",
        ]
    return "\n".join(lines)


def format_regression_calibration(calibration: Calibration) -> str:
    """Render the outcome of the calibration step as text."""
    lines = [
        "-" * 64,
        "CALIBRATION REPORT",
        "-" * 64,
        f"Calibration samples: {calibration.nb_calibration_samples}",
        f"alpha: {calibration.alpha}  (target coverage >= {1 - calibration.alpha:.1%})",
    ]
    if math.isinf(calibration.qhat):
        lines.append("qhat: inf  (the calibration set is too small for this alpha: intervals are unbounded)")
    else:
        lines += [
            f"qhat: {calibration.qhat:.4f}  (in the units of the target)",
            f"prediction interval: prediction +/- {calibration.qhat:.4f}",
        ]
    return "\n".join(lines)


def format_regression_uncertainty(u: RegressionUncertainty) -> str:
    """Render the uncertainty summary as text."""
    return "\n".join([
        "-" * 64,
        "UNCERTAINTY REPORT",
        "-" * 64,
        f"Target coverage:    >= {1 - u.alpha:.1%} (alpha = {u.alpha})",
        f"Empirical coverage: {u.nb_covered}/{u.nb_samples} ({u.coverage:.1%})",
        f"Interval width:     {u.interval_width:.4f}  (identical for every sample)",
    ])
