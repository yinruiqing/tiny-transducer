import torch
from torch import nn
import torch.nn.functional as F


class CVFSMNv2(nn.Module):
    def __init__(self, memory_size, input_size, output_size, projection_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self.projection_size = projection_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.projection_size))
        self._W2 = nn.Parameter(torch.Tensor(self.projection_size, self.output_size))
        self._bias1 = nn.Parameter(torch.Tensor(self.projection_size))
        self._bias2 = nn.Parameter(torch.Tensor(self.output_size))
        self.memory_weights = nn.Parameter(torch.Tensor(self.projection_size, 1, self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias1)
        nn.init.ones_(self._bias2)
        nn.init.xavier_uniform_(self.memory_weights)

    def forward(self, input_data, skip_conn):

        p = torch.matmul(input_data, self._W1) + self._bias1
        p_T = F.pad(p.transpose(1, 2), (self.memory_size - 1, 0))
        h_hat = F.conv1d(p_T, self.memory_weights, groups=self.projection_size).transpose(1, 2)
        if skip_conn is not None:
            p_hat = h_hat + skip_conn
        else:
            p_hat = h_hat
        h = torch.matmul(p, self._W2) + self._bias2
        return h, p_hat


class FSMNBlock(nn.Module):
    def __init__(self, memory_size, input_size, output_size, projection_size):
        super().__init__()
        self.fsmn = CVFSMNv2(memory_size, input_size, output_size, projection_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_conn=None):
        out, memory_weights = self.fsmn(x, skip_conn)
        out = self.norm(out)
        return out, memory_weights


class DFSMN(nn.Module):
    def __init__(self, memory_size=8, input_size=256, hidden_size=512, output_size=256, projection_size=320,
                 n_layers=6):
        super().__init__()
        self.layer1 = FSMNBlock(memory_size, input_size, hidden_size, projection_size)
        self.fsmn_layers = self._make_layers(n_layers - 1, memory_size, hidden_size, output_size, projection_size)

    def forward(self, x):
        skip_conn = None
        x, skip_conn = self.layer1(x, skip_conn)
        for layer in self.fsmn_layers:
            x, memory_weights = layer(x, skip_conn)
        return x

    def _make_layers(self, num_block, memory_size, hidden_size, output_size, projection_size):
        layers = []
        for _ in range(num_block - 1):
            layers.append(FSMNBlock(memory_size, hidden_size, hidden_size, projection_size))
        layers.append(FSMNBlock(memory_size, hidden_size, output_size, projection_size))
        return nn.ModuleList(layers)
