import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN architecture — baseline model for CIFAR-10."""

    def __init__(self) -> None:
        """Initialize layers."""
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # After conv1+pool: 32x15x15, after conv2+pool: 64x6x6 = 2304
        self.fc = nn.Linear(in_features=2304, out_features=10)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, 32, 32)

        Returns:
            Class logits (batch, 10)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x