import os
import torch
import torch.nn as nn
import torch.optim as optim

from parameters import get_configs
from models.simple_cnn import SimpleCNN
from models.resnet import ResNet
from models.mobilenet import MobileNet
from train import (
    get_dataloaders,
    train_one_epoch,
    train_with_distillation,
    train_with_soft_labels
)
from test import evaluate, count_flops
from visualize import (
    plot_training_curves,
    plot_tsne,
    plot_confusion_matrix,
    plot_flops_accuracy_comparison
)


def main() -> None:
    """
    Main function — runs all experiments sequentially.
    """
    os.makedirs("plots", exist_ok=True)

    # ─── 1. Load parameters ───
    train_cfg, distill_cfg, smooth_cfg, transfer_cfg = get_configs()
    device = train_cfg.device

    # ─── 2. Load data (32x32 — used for most experiments) ───
    print("Loading data...")
    train_loader, test_loader = get_dataloaders(train_cfg)

    # ─── 3. Transfer Learning ───
    print("=" * 50)
    print("A. TRANSFER LEARNING")
    print("=" * 50)

    for option in [1, 2]:
        print(f"--- Option {option} ---")

        # Option 1: resize to 224x224 (ImageNet compatible, freeze early layers)
        # Option 2: keep 32x32 (modified conv1, fine-tune all layers)
        if option == 1:
            tl_train_loader, tl_test_loader = get_dataloaders(
                train_cfg, resize=224
            )
        else:
            tl_train_loader, tl_test_loader = get_dataloaders(
                train_cfg, resize=0
            )

        model = ResNet(option=option, pretrained=True).to(device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "accuracy": []}
        for epoch in range(train_cfg.epochs):
            train_loss = train_one_epoch(
                model, tl_train_loader, optimizer, criterion, device
            )
            val_loss, accuracy = evaluate(
                model, tl_test_loader, criterion, device
            )
            history["train_loss"].append(train_loss)
            history["accuracy"].append(accuracy)
            print(
                f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Accuracy: {accuracy:.2f}%"
            )

        tag = f"TransferLearning_Option{option}"
        plot_training_curves(history, tag, f"plots/{tag}_curves.png")

        print(f"Option {option} FLOPs:")
        count_flops(model, device)

    # ─── 4. Knowledge Distillation ───
    print("=" * 50)
    print("B. KNOWLEDGE DISTILLATION")
    print("=" * 50)

    # ─── 4.1 SimpleCNN Baseline ───
    print("--- 1. SimpleCNN Baseline ---")
    simple_cnn = SimpleCNN().to(device)
    optimizer = optim.Adam(simple_cnn.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history_cnn = {"train_loss": [], "accuracy": []}
    for epoch in range(train_cfg.epochs):
        train_loss = train_one_epoch(
            simple_cnn, train_loader, optimizer, criterion, device
        )
        val_loss, accuracy = evaluate(
            simple_cnn, test_loader, criterion, device
        )
        history_cnn["train_loss"].append(train_loss)
        history_cnn["accuracy"].append(accuracy)
        print(
            f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    plot_training_curves(history_cnn, "SimpleCNN_Baseline", "plots/SimpleCNN_Baseline_curves.png")

    print("SimpleCNN FLOPs:")
    count_flops(simple_cnn, device)

    # ─── 4.2 ResNet from scratch — without Label Smoothing ───
    print("--- 2a. ResNet (scratch, no label smoothing) ---")
    resnet = ResNet(option=2, pretrained=False).to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history_resnet = {"train_loss": [], "accuracy": []}
    for epoch in range(train_cfg.epochs):
        train_loss = train_one_epoch(
            resnet, train_loader, optimizer, criterion, device
        )
        val_loss, accuracy = evaluate(
            resnet, test_loader, criterion, device
        )
        history_resnet["train_loss"].append(train_loss)
        history_resnet["accuracy"].append(accuracy)
        print(
            f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    plot_training_curves(history_resnet, "ResNet_NoSmoothing", "plots/ResNet_NoSmoothing_curves.png")

    # ─── 4.3 ResNet from scratch — with Label Smoothing ───
    print("--- 2b. ResNet (scratch, with label smoothing) ---")
    resnet_smooth = ResNet(option=2, pretrained=False).to(device)
    optimizer = optim.Adam(resnet_smooth.parameters(), lr=train_cfg.learning_rate)
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=smooth_cfg.epsilon)

    best_accuracy = 0.0
    history_smooth = {"train_loss": [], "accuracy": []}

    for epoch in range(train_cfg.epochs):
        train_loss = train_one_epoch(
            resnet_smooth, train_loader, optimizer, criterion_smooth, device
        )
        val_loss, accuracy = evaluate(
            resnet_smooth, test_loader, criterion_smooth, device
        )
        history_smooth["train_loss"].append(train_loss)
        history_smooth["accuracy"].append(accuracy)
        print(
            f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(resnet_smooth.state_dict(), "best_resnet.pth")
            print(f"Best model saved: {accuracy:.2f}%")

    plot_training_curves(history_smooth, "ResNet_LabelSmoothing", "plots/ResNet_LabelSmoothing_curves.png")

    # Reload the best model to avoid reference issues
    best_resnet = ResNet(option=2, pretrained=False).to(device)
    best_resnet.load_state_dict(torch.load("best_resnet.pth"))
    best_resnet.eval()

    print(f"ResNet FLOPs:")
    count_flops(resnet_smooth, device)

    # ─── 4.4 SimpleCNN — with Distillation ───
    print("--- 3. SimpleCNN (with distillation) ---")
    student_cnn = SimpleCNN().to(device)
    optimizer = optim.Adam(student_cnn.parameters(), lr=train_cfg.learning_rate)

    history_distill = {"train_loss": [], "accuracy": []}
    for epoch in range(train_cfg.epochs):
        train_loss = train_with_distillation(
            student_cnn, best_resnet,
            train_loader, optimizer, distill_cfg, device
        )
        val_loss, accuracy = evaluate(
            student_cnn, test_loader,
            nn.CrossEntropyLoss(), device
        )
        history_distill["train_loss"].append(train_loss)
        history_distill["accuracy"].append(accuracy)
        print(
            f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    plot_training_curves(history_distill, "SimpleCNN_Distillation", "plots/SimpleCNN_Distillation_curves.png")

    # ─── 4.5 MobileNet — with Soft Labels ───
    print("--- 4. MobileNet (with soft labels) ---")
    mobilenet = MobileNet().to(device)
    optimizer = optim.Adam(mobilenet.parameters(), lr=train_cfg.learning_rate)

    history_mobile = {"train_loss": [], "accuracy": []}
    for epoch in range(train_cfg.epochs):
        train_loss = train_with_soft_labels(
            mobilenet, best_resnet,
            train_loader, optimizer, device
        )
        val_loss, accuracy = evaluate(
            mobilenet, test_loader,
            nn.CrossEntropyLoss(), device
        )
        history_mobile["train_loss"].append(train_loss)
        history_mobile["accuracy"].append(accuracy)
        print(
            f"Epoch [{epoch + 1}/{train_cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    plot_training_curves(history_mobile, "MobileNet_SoftLabels", "plots/MobileNet_SoftLabels_curves.png")

    print(f"MobileNet FLOPs:")
    count_flops(mobilenet, device)

    # ─── 5. Results & Visualizations ───
    print("=" * 50)
    print("RESULTS & VISUALIZATIONS")
    print("=" * 50)

    results = [
        ("SimpleCNN Baseline", simple_cnn),
        ("ResNet (no smoothing)", resnet),
        ("ResNet (label smoothing)", best_resnet),
        ("SimpleCNN (distillation)", student_cnn),
        ("MobileNet (soft labels)", mobilenet),
    ]

    for name, model in results:
        _, accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"{name:35s} → Accuracy: {accuracy:.2f}%")

    # ─── 5.1 Confusion matrices ───
    print("Generating confusion matrices...")
    for name, model in results:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            model, test_loader, device,
            name, f"plots/cm_{safe_name}.png"
        )

    # ─── 5.2 t-SNE embeddings ───
    print("Generating t-SNE plots...")
    for name, model in results:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_tsne(
            model, test_loader, device,
            name, f"plots/tsne_{safe_name}.png"
        )

    # ─── 5.3 FLOPs vs Accuracy comparison ───
    print("Generating FLOPs vs Accuracy comparison...")
    try:
        from ptflops import get_model_complexity_info

        comp_names = []
        comp_acc = []
        comp_flops = []

        for name, model in results:
            _, acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
            flops, _ = get_model_complexity_info(
                model, (3, 32, 32),
                as_strings=True,
                print_per_layer_stat=False
            )
            comp_names.append(name)
            comp_acc.append(acc)
            comp_flops.append(flops)

        plot_flops_accuracy_comparison(
            comp_names, comp_acc, comp_flops,
            "plots/flops_accuracy_comparison.png"
        )
    except ImportError:
        print("ptflops not installed, skipping comparison chart.")

    print("All experiments and visualizations complete!")


if __name__ == "__main__":
    main()