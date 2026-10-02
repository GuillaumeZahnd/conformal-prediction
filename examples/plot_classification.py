import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch
from pathlib import Path


def plot_prediction_set(
    image: torch.Tensor,
    probs: torch.Tensor,
    qhat: float,
    alpha: float,
    sample_index: int,
    true_label: int | None = None,
    class_names: tuple[str, ...] | None = None,
) -> None:
    """
    Plot one input next together with its class probabilities, showing which classes are in the prediction set.

    Args:
        image: Input image, of shape (1, H, W) or (3, H, W).
        probs: Softmax probabilities, of shape (nb_classes,).
        qhat: Calibrated conformal threshold. A class is in the prediction set if 1 - probability <= qhat.
        alpha: Target miscoverage level.
        sample_index: Index of the sample in its dataset.
        true_label: Ground-truth class index, if available.
        class_names: Optional display names, one per class.
    """

    probs = probs.detach().cpu()
    nb_classes = len(probs)
    names = class_names or [str(c) for c in range(nb_classes)]
    in_set = (1 - probs) <= qhat
    threshold = max(1 - qhat, 0.0)

    color_in_set = "gold"
    color_not_in_set = "lightgray"

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]})

    # Image
    img = image.detach().cpu()
    if img.shape[0] == 1:
        ax_img.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax_img.imshow(img.permute(1, 2, 0))
    ax_img.axis("off")

    # Class probabilities
    colors = [color_in_set if s else color_not_in_set for s in in_set.tolist()]
    bars = ax_bar.barh(range(nb_classes), probs.tolist(), color=colors)
    if true_label is not None:
        bars[true_label].set_edgecolor("black")
        bars[true_label].set_linewidth(2.5)
    threshold_line = ax_bar.axvline(
        threshold, color="tab:red", linestyle="--", label=f"threshold (1 - qhat = {threshold:.3f})"
    )

    ax_bar.set_yticks(range(nb_classes), names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Softmax probability")
    ax_bar.set_ylabel("Labels")

    handles = [
        Patch(color=color_in_set, label="in prediction set"),
        Patch(color=color_not_in_set, label="not in prediction set")
    ]

    if true_label is not None:
        handles.append(Patch(facecolor="white", edgecolor="black", linewidth=2, label="true label"))
    ax_bar.legend(handles=handles + [threshold_line], loc="best")

    # Title
    set_names = [names[c] for c in in_set.nonzero().flatten().tolist()]
    title = f"Conformal prediction with alpha={alpha}"
    title += f"\nSample index: {sample_index}"
    title += f"\nPrediction set ({len(set_names)} classes): {{{', '.join(set_names)}}}"
    if true_label is not None:
        outcome = "covered" if in_set[true_label] else "not covered"
        title += f"\ntrue label: {names[true_label]} ({outcome})"
    fig.suptitle(title)
    fig.tight_layout()

    # Save
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_classification.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
