import math
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    alpha: float
    qhat: float
    nb_calibration_samples: int


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """k-th smallest score, with k = ceil((n + 1)(1 - alpha)); inf if the calibration set is too small."""
    nb_samples = len(scores)
    k = math.ceil((nb_samples + 1) * (1 - alpha))
    return float("inf") if k > nb_samples else scores.sort().values[k - 1].item()
