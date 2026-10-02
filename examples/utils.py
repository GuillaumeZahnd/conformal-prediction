import torch


def print_classification_task_report(probs: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Print basic classification metrics: overall accuracy, mean confidence, per-class accuracy.

    Args:
        probs: Softmax probabilities of shape (nb_samples, nb_classes).
        targets: Ground-truth class labels of shape (nb_samples,), int64.
    """
    nb_samples, nb_classes = probs.shape
    correct = probs.argmax(dim=1) == targets

    print("\n" + "-"*64)
    print("CLASSIFICATION REPORT")
    print("-"*64)

    nb_correct = int(correct.sum())
    print(f"Accuracy: {nb_correct}/{nb_samples} ({nb_correct / nb_samples:.1%})")
    print(f"Mean confidence (top-1 probability): {probs.max(dim=1).values.mean().item():.3f}")
    print("")

    support = torch.bincount(targets, minlength=nb_classes)
    correct_per_class = torch.bincount(targets[correct], minlength=nb_classes)

    print(f"{'class':>5} | {'support':>7} | {'accuracy':>8}")
    for c in range(nb_classes):
        if support[c] == 0:
            continue  # Skip cases that do not occur
        print(f"{c:>5} | {support[c].item():>7} | {correct_per_class[c].item() / support[c].item():>8.3f}")
        

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
