import torch

import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    CNN with one convolutional layer for CIFAR10
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(8192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process in order
        """
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
