import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """
    Evaluate the model on the test set.

    Args:
        model: Model to evaluate
        dataloader: Test data loader
        criterion: Loss function
        device: Device (cuda/cpu)

    Returns:
        tuple: (average loss, accuracy percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get the class with the highest probability
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def count_flops(model: nn.Module, device: str) -> None:
    """
    Compute FLOPs and parameter count for the model.

    Args:
        model: Model to analyze
        device: Device (cuda/cpu)
    """

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(
        model,
        (3, 32, 32),
        as_strings=True,
        print_per_layer_stat=False
    )
    print(f"FLOPs      : {flops}")
    print(f"Parameters : {params}")

