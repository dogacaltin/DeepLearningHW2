import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    """
    ResNet-18 architecture — used for Transfer Learning and as Teacher model.

    Args:
        option: 1 or 2 (Transfer Learning option)
        pretrained: Whether to use pretrained ImageNet weights
    """

    def __init__(self, option: int = 1, pretrained: bool = True) -> None:
        super(ResNet, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)

        if option == 1:
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False
            # Replace FC layer for CIFAR-10
            self.model.fc = nn.Linear(in_features=512, out_features=10)
            # Unfreeze FC layer
            self.model.fc.weight.requires_grad = True
            self.model.fc.bias.requires_grad = True

        elif option == 2:
            # Adapt conv1 for CIFAR-10 (smaller kernel, no aggressive downsampling)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            # Replace FC layer for CIFAR-10
            self.model.fc = nn.Linear(in_features=512, out_features=10)
            # No freezing — all layers are trainable

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, 32, 32)

        Returns:
            Class logits (batch, 10)
        """
        return self.model(x)