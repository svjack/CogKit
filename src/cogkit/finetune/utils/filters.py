import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFilter(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

        self.mean_value = 1.0 / (kernel_size * kernel_size)
        weight = torch.full((1, 1, kernel_size, kernel_size), 1.0)

        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        batch_size, height, width = x.shape

        if height % self.kernel_size != 0 or width % self.kernel_size != 0:
            raise ValueError(
                f"Input dimensions (height={height}, width={width}) must be divisible by kernel_size={self.kernel_size}"
            )

        x = x.unsqueeze(1)  # add channel dimension

        filtered = F.conv2d(x, self.weight, stride=self.stride, padding=0, bias=None)

        filtered = filtered * self.mean_value

        filtered = filtered.squeeze(1)

        return filtered
