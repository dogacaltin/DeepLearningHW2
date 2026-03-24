import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from parameters import TrainConfig, DistillationConfig, LabelSmoothingConfig


def get_dataloaders(
    config: TrainConfig,
    resize: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset and return DataLoaders.

    Args:
        config: Training configuration
        resize: If > 0, resize images to this size (e.g. 224 for ImageNet compatibility).
                If 0, keep original 32x32 resolution.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Build transform lists
    train_transforms = []
    test_transforms = []

    # Resize if requested (for Transfer Learning Option 1)
    if resize > 0:
        train_transforms.append(transforms.Resize(resize))
        test_transforms.append(transforms.Resize(resize))

    # Data augmentation (only for training)
    train_transforms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(resize if resize > 0 else 32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transforms)

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device (cuda/cpu)

    Returns:
        Average loss value
    """
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float
) -> torch.Tensor:
    """
    Compute Knowledge Distillation loss.

    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs
        labels: Ground truth labels
        T: Temperature for softening probabilities
        alpha: Weight for the hard label (cross-entropy) loss

    Returns:
        Combined distillation loss
    """
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)

    distill = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
    student_loss = F.cross_entropy(student_logits, labels)

    return alpha * student_loss + (1 - alpha) * distill


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: DistillationConfig,
    device: str
) -> float:
    """
    Train the student model using Knowledge Distillation.

    Args:
        student: Small model to be trained
        teacher: Large pretrained teacher model
        dataloader: Training data loader
        optimizer: Optimization algorithm
        config: Distillation configuration
        device: Device (cuda/cpu)

    Returns:
        Average loss value
    """
    student.train()
    teacher.eval()  # Teacher is not trained!
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        student_logits = student(images)

        # Teacher does not compute gradients
        with torch.no_grad():
            teacher_logits = teacher(images)

        loss = distillation_loss(
            student_logits, teacher_logits,
            labels, config.temperature, config.alpha
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_with_soft_labels(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_classes: int = 10
) -> float:
    """
    Train the student using teacher's correct-class probability as soft labels.

    The teacher's predicted probability for the true class is assigned to
    the correct class, while the remaining probability is distributed
    equally among the other classes.

    Args:
        student: MobileNet model to be trained
        teacher: Teacher ResNet model
        dataloader: Training data loader
        optimizer: Optimization algorithm
        device: Device (cuda/cpu)
        num_classes: Number of classes

    Returns:
        Average loss value
    """
    student.train()
    teacher.eval()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(images)
            teacher_probs = F.softmax(teacher_logits, dim=1)

            # Get the teacher's probability for the correct class
            correct_prob = teacher_probs[range(len(labels)), labels]

            # Build soft labels:
            # Other classes get equal share of remaining probability
            soft_labels = torch.full(
                (len(labels), num_classes),
                fill_value=0.0,
                device=device
            )
            other_prob = (1 - correct_prob) / (num_classes - 1)

            soft_labels += other_prob.unsqueeze(1)
            soft_labels[range(len(labels)), labels] = correct_prob

        student_logits = student(images)
        student_probs = F.log_softmax(student_logits, dim=1)

        loss = F.kl_div(student_probs, soft_labels, reduction="batchmean")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)