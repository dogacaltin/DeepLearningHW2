import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """General training hyperparameters."""

    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 64
    device: str = "cuda"


@dataclass
class DistillationConfig:
    """Knowledge Distillation parameters."""

    temperature: float = 3.0
    alpha: float = 0.2


@dataclass
class LabelSmoothingConfig:
    """Label Smoothing parameters."""

    epsilon: float = 0.2


@dataclass
class TransferLearningConfig:
    """Transfer Learning parameters."""

    option: int = 1          # 1 or 2
    freeze: bool = True      # True for Option 1


def get_configs() -> tuple[TrainConfig, DistillationConfig,
                           LabelSmoothingConfig, TransferLearningConfig]:
    """
    Parse command-line arguments and return config objects.

    Returns:
        tuple: (TrainConfig, DistillationConfig,
                LabelSmoothingConfig, TransferLearningConfig)
    """
    parser = argparse.ArgumentParser(
        description="CS515 HW1b - Transfer Learning & Knowledge Distillation"
    )

    # TrainConfig
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")

    # DistillationConfig
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.2)

    # LabelSmoothingConfig
    parser.add_argument("--epsilon", type=float, default=0.2)

    # TransferLearningConfig
    parser.add_argument("--option", type=int, default=1, choices=[1, 2])
    parser.add_argument("--freeze", type=bool, default=True)

    args = parser.parse_args()

    train_config = TrainConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    distillation_config = DistillationConfig(
        temperature=args.temperature,
        alpha=args.alpha
    )

    label_smoothing_config = LabelSmoothingConfig(
        epsilon=args.epsilon
    )

    transfer_config = TransferLearningConfig(
        option=args.option,
        freeze=args.freeze
    )

    return train_config, distillation_config, label_smoothing_config, transfer_config