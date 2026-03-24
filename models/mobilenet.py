import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    """
    MobileNetV2 architecture — Student model for Knowledge Distillation.

    Args:
        pretrained: Whether to use pretrained ImageNet weights
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(MobileNet, self).__init__()

        self.model = models.mobilenet_v2(pretrained=pretrained)

        # Replace classifier for CIFAR-10
        self.model.classifier[1] = nn.Linear(
            in_features=1280,
            out_features=10
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, 32, 32)

        Returns:
            Class logits (batch, 10)
        """
        return self.model(x)