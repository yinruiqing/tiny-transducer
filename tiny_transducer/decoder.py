import torch
from torch import nn
from torch.nn import Conv1d
import torch.nn.functional as F


class DecoderTinyTransducer(nn.Module):
    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int,
            output_dim: int,
            kernel_size: int,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.conv = Conv1d(in_channels=hidden_state_dim, out_channels=output_dim, kernel_size=kernel_size)

    def forward(self, x):
        out = self.embedding(x)
        out = out.transpose(1, 2)
        out = F.pad(out, (self.kernel_size - 1, 0))
        out = self.conv(out)
        out = out.transpose(1, 2)
        return out


if __name__ == '__main__':
    x = torch.ones(4, 100).long()
    encoder = DecoderTinyTransducer(10, 256, 256, 4)
    print(encoder(x).shape)
