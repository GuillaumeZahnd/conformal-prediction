from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import torch


def plot_prediction_set(
    image: torch.Tensor,
    probs: torch.Tensor,
    in_set: torch.Tensor,
    qhat: float,
    alpha: float,
    sample_index: int,
    true_label: int | None = None,
    class_names: tuple[str, ...] | list[str] | None = None,
    k_regularization: int | None = None,
    lambda_regularization: float | None = None,
    output_dir: str | Path = "figures",
) -> None:
    """
    Plot an input image alongside its class probabilities ordered by rank,
    highlighting membership in a RAPS prediction set.

    Args:
        image: Input image tensor of shape (1, H, W) or (3, H, W).
        probs: Softmax probabilities of shape (nb_classes,).
        in_set: Boolean tensor of shape (nb_classes,) from get_prediction_sets.
        qhat: Calibrated RAPS threshold (cumulative penalized score cutoff).
        alpha: Target miscoverage level.
        sample_index: Index of the sample in its dataset.
        true_label: Ground-truth class index, if available.
        class_names: Optional display names, one per class.
        k_regularization: Optional k_reg used in RAPS, shown as a rank divider.
        lambda_regularization: Optional lambda used in RAPS, shown in subtitle.
        output_dir: Directory where the figure is saved.
    """
    probs_cpu = probs.detach().cpu().float()
    in_set_cpu = in_set.detach().cpu().bool()
    nb_classes = probs_cpu.numel()
    names = list(class_names) if class_names is not None else [str(c) for c in range(nb_classes)]

    # Order classes from highest to lowest probability
    order = torch.argsort(probs_cpu, descending=True)
    ordered_probs = probs_cpu[order]
    ordered_in_set = in_set_cpu[order]
    ordered_labels = [names[idx] for idx in order.tolist()]

    color_in_set = "gold"
    color_not_in_set = "lightgray"
    bar_colors = [color_in_set if s else color_not_in_set for s in ordered_in_set.tolist()]

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2.2]}
    )

    # Input image
    img = image.detach().cpu()
    if img.shape[0] == 1:
        ax_img.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax_img.imshow(img.permute(1, 2, 0))
    ax_img.axis("off")

    # Sorted class probabilities
    y_pos = range(nb_classes)
    bars = ax_bar.barh(y_pos, ordered_probs.tolist(), color=bar_colors, zorder=3)

    if true_label is not None:
        true_pos = (order == true_label).nonzero().item()
        bars[true_pos].set_edgecolor("black")
        bars[true_pos].set_linewidth(2.5)

    if k_regularization is not None and 0 < k_regularization < nb_classes:
        ax_bar.axhline(
            k_regularization - 0.5,
            color="tab:blue",
            linestyle="--",
            zorder=4,
            label=f"k_reg cutoff (top {k_regularization})",
        )

    x_upper = max(1.0, float(ordered_probs.max()) * 1.05)
    ax_bar.set_yticks(list(y_pos), ordered_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, x_upper)
    ax_bar.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_bar.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax_bar.set_xlabel("Softmax probability")
    ax_bar.set_ylabel("Class")
    ax_bar.grid(axis="x", which="major", linestyle="-", alpha=0.6, zorder=0)
    ax_bar.grid(axis="x", which="minor", linestyle=":", alpha=0.45, zorder=0)

    handles = [
        Patch(color=color_in_set, label="in prediction set"),
        Patch(color=color_not_in_set, label="not in prediction set"),
    ]
    if true_label is not None:
        handles.append(
            Patch(facecolor="white", edgecolor="black", linewidth=2, label="true label")
        )
    if k_regularization is not None and 0 < k_regularization < nb_classes:
        handles.append(
            plt.Line2D([0], [0], color="tab:blue", linestyle="--", label=f"k_reg = {k_regularization}")
        )
    ax_bar.legend(handles=handles, loc="lower right")

    # Title
    set_names = [names[idx] for idx in order[ordered_in_set].tolist()]
    reg_info = ""
    if k_regularization is not None or lambda_regularization is not None:
        reg_info = f", k={k_regularization}, lambda={lambda_regularization}"
    title = f"RAPS prediction set (alpha={alpha}, qhat={float(qhat):.3f}{reg_info})"
    title += f"\nSample index: {sample_index}"
    title += f"\nPrediction set ({len(set_names)} classes): {{{', '.join(set_names)}}}"
    if true_label is not None:
        covered = bool(in_set_cpu[true_label].item())
        outcome = "covered" if covered else "not covered"
        title += f"\ntrue label: {names[true_label]} ({outcome})"
    fig.suptitle(title)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"demo_raps_sample_{sample_index}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
