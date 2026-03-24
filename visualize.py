import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def plot_training_curves(
    history: dict[str, list[float]],
    title: str,
    save_path: str
) -> None:
    """
    Plot training loss and validation accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss' and 'accuracy',
                 each mapping to a list of per-epoch values.
        title: Plot title
        save_path: File path to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["accuracy"], "r-o", markersize=4, label="Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_tsne(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    title: str,
    save_path: str,
    max_samples: int = 2000
) -> None:
    """
    Generate a t-SNE visualization of the model's learned feature embeddings.

    Extracts features from the penultimate layer and projects them to 2D.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device (cuda/cpu)
        title: Plot title
        save_path: File path to save the figure
        max_samples: Maximum number of samples to use (for speed)
    """
    model.eval()
    features_list = []
    labels_list = []

    # Hook to capture penultimate layer output
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output.detach().cpu())

    # Register hook on the layer before the final classifier
    # Works for ResNet, SimpleCNN, and MobileNet
    hook_handle = None
    if hasattr(model, "model"):
        # ResNet or MobileNet (wrapped in model.model)
        inner = model.model
        if hasattr(inner, "fc"):
            # ResNet: hook on avgpool
            hook_handle = inner.avgpool.register_forward_hook(hook_fn)
        elif hasattr(inner, "classifier"):
            # MobileNet: hook on the dropout before classifier
            hook_handle = inner.classifier[0].register_forward_hook(hook_fn)
    elif hasattr(model, "fc"):
        # SimpleCNN: hook on conv2
        hook_handle = model.conv2.register_forward_hook(hook_fn)

    if hook_handle is None:
        print(f"  ⚠️  Could not attach hook for {title}, skipping t-SNE.")
        return

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)

            feat = hook_output[-1]
            feat = feat.view(feat.size(0), -1)
            features_list.append(feat)
            labels_list.append(labels)

            if sum(f.size(0) for f in features_list) >= max_samples:
                break

    hook_handle.remove()

    features = torch.cat(features_list, dim=0)[:max_samples].numpy()
    labels = torch.cat(labels_list, dim=0)[:max_samples].numpy()

    # Run t-SNE
    print(f"  ⏳ Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap="tab10", s=5, alpha=0.7
    )
    handles, _ = scatter.legend_elements(num=10)
    ax.legend(handles, CIFAR10_CLASSES, loc="best", fontsize=8)
    ax.set_title(f"t-SNE — {title}")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    title: str,
    save_path: str
) -> None:
    """
    Plot a confusion matrix for the given model.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device (cuda/cpu)
        title: Plot title
        save_path: File path to save the figure
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {title}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_flops_accuracy_comparison(
    model_names: list[str],
    accuracies: list[float],
    flops_list: list[str],
    save_path: str
) -> None:
    """
    Plot a grouped bar chart comparing FLOPs and accuracy across models.

    Args:
        model_names: List of model names
        accuracies: List of accuracy values (%)
        flops_list: List of FLOPs strings (e.g., '0.56 GMac')
        save_path: File path to save the figure
    """
    # Parse FLOPs strings to numeric (in millions)
    flops_numeric = []
    for f in flops_list:
        f = f.strip()
        if "GMac" in f:
            flops_numeric.append(float(f.replace("GMac", "").strip()) * 1000)
        elif "MMac" in f:
            flops_numeric.append(float(f.replace("MMac", "").strip()))
        else:
            flops_numeric.append(0.0)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Accuracy bars
    bars1 = ax1.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#4C72B0")
    ax1.set_ylabel("Accuracy (%)", color="#4C72B0")
    ax1.set_ylim(0, 100)

    # FLOPs bars on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, flops_numeric, width, label="FLOPs (MMac)", color="#DD8452")
    ax2.set_ylabel("FLOPs (MMac)", color="#DD8452")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20, ha="right")
    ax1.set_title("Model Comparison — Accuracy vs FLOPs")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {save_path}")