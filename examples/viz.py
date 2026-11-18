import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from conformal_prediction import Calibration


def plot_conformal_calibration(
    calibration_scores: torch.Tensor,
    calibration: Calibration,
    title: str,
) -> None:
    """Plot the calibration scores distribution (with KDE) and the sorted scores 
    with the conformal quantile threshold and x-tick indicator.

    Parameters
    ----------
    calibration_scores : torch.Tensor
        Tensor of calibration scores.
    calibration : Calibration
        Calibration object containing alpha, qhat, and n_calibration_samples.
    """
    scores_np = calibration_scores.detach().cpu().numpy()
    sorted_scores = torch.sort(calibration_scores).values.detach().cpu().numpy()
    n = len(scores_np)

    # Set seaborn style for clean aesthetics
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Histogram distribution with KDE using seaborn
    sns.histplot(
        scores_np, 
        kde=False, 
        stat="density", 
        bins=30, 
        ax=axes[0], 
        color="skyblue", 
        edgecolor="black", 
        alpha=0.6
    )
    axes[0].axvline(
        calibration.qhat,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$\\hat{{q}}$ = {calibration.qhat:.4f}",
    )
    axes[0].set_xlim(left=-0.02)    
    axes[0].set_title("Calibration Scores Distribution & KDE")
    axes[0].set_xlabel("Score Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Right panel: Sorted scores with conformal quantile threshold & explicit x-tick
    x_vals = (torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)).numpy()
    axes[1].plot(x_vals, sorted_scores, marker=".", linestyle="-", color="teal", label="Sorted Scores")
    axes[1].axhline(
        calibration.qhat,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$\\hat{{q}}$ ($1 - \\alpha$ = {1 - calibration.alpha:.2f})",
    )

    # Highlight the exact index k used for finite-sample correction
    k = math.ceil((n + 1) * (1 - calibration.alpha))
    if k <= n:
        x_k = k / (n + 1)
        axes[1].scatter(
            [x_k],
            [calibration.qhat],
            color="darkred",
            s=100,
            zorder=5,
            label=f"k-th score ($k={k}$)",
        )
        
        axes[1].set_xlim(left=-0.02, right=1.02)
        axes[1].tick_params(axis='x', rotation=45)

    axes[1].set_title("Sorted Scores & Conformal Quantile")
    # Fixed: Matplotlib mathtext doesn't support \displaystyle
    axes[1].set_xlabel("Quantile Level Index $\\frac{i}{n+1}$")
    axes[1].set_ylabel("Score Value")
    axes[1].legend()

    plt.tight_layout()
    
    sns.despine(fig=fig)

    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
