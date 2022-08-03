import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import LSTM

from tiny_transducer.fsmn import DFSMN


class EncoderTinyTransducer(nn.Module):
    def __init__(
            self,
            input_size,
            memory_size=8,
            hidden_size=512,
            output_size=256,
            projection_size=320,
            n_layers=6):
        super().__init__()
        self.conv1 = Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.LayerNorm(output_size)
        self.conv2 = Conv1d(in_channels=output_size, out_channels=output_size, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(output_size)
        self.lstm = LSTM(input_size=output_size, hidden_size=output_size, batch_first=True)
        self.norm3 = nn.LayerNorm(output_size)
        self.dfsmn = DFSMN(memory_size=memory_size, input_size=output_size, hidden_size=hidden_size,
                           output_size=output_size,
                           projection_size=projection_size, n_layers=n_layers)

    def forward(self, x):
        out = self.norm1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        out = self.norm2(self.conv2(out.transpose(1, 2)).transpose(1, 2))
        out, _ = self.lstm(out)
        out = self.norm3(out)
        out = self.dfsmn(out)
        return out


if __name__ == '__main__':
    x = torch.randn(4, 100, 256)
    encoder = EncoderTinyTransducer(input_size=256)
    print(encoder(x).shape)
