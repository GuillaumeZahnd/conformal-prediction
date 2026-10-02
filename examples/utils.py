import torch


def format_classification_task_report(probs: torch.Tensor, targets: torch.Tensor) -> str:
    """Render basic classification metrics: overall accuracy, mean confidence, per-class accuracy."""
    nb_samples, nb_classes = probs.shape
    correct = probs.argmax(dim=1) == targets
    nb_correct = int(correct.sum())

    support = torch.bincount(targets, minlength=nb_classes)
    correct_per_class = torch.bincount(targets[correct], minlength=nb_classes)

    lines = [
        "-" * 64,
        "CLASSIFICATION REPORT",
        "-" * 64,
        f"Accuracy: {nb_correct}/{nb_samples} ({nb_correct / nb_samples:.1%})",
        f"Mean confidence (top-1 probability): {probs.max(dim=1).values.mean().item():.3f}",
        "",
        f"{'class':>5} | {'support':>7} | {'accuracy':>8}",
    ]
    for c in range(nb_classes):
        if support[c] == 0:
            continue  # Skip cases that do not occur
        lines.append(f"{c:>5} | {support[c].item():>7} | {correct_per_class[c].item() / support[c].item():>8.3f}")
    return "\n".join(lines)
        

def format_regression_task_report(predictions: torch.Tensor, targets: torch.Tensor) -> str:
    """Render basic regression metrics: MAE, RMSE, R2."""
    targets = targets.double()
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = errors.pow(2).mean().sqrt().item()
    r2 = 1 - float(errors.pow(2).sum() / (targets - targets.mean()).pow(2).sum())

    lines = [
        "-" * 64,
        "REGRESSION REPORT",
        "-" * 64,
        f"MAE:  {mae:.4f}  (in the units of the target)",
        f"RMSE: {rmse:.4f}  (in the units of the target)",
        f"R2:   {r2:.3f}",
    ]
    return "\n".join(lines)
