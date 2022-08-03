import torch
import torch.nn as nn
from torch import Tensor

from tiny_transducer.decoder import DecoderTinyTransducer
from tiny_transducer.encoder import EncoderTinyTransducer


class TinyTransducer(nn.Module):

    def __init__(
            self,
            input_size,
            memory_size: int =8,
            hidden_size: int =512,
            output_size: int =256,
            projection_size: int =320,
            n_layers: int =6,
            num_classes: int = 300,
            decoder_hidden_state_dim: int = 256,
            decoder_kernel_size: int = 4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = EncoderTinyTransducer(
            input_size=input_size,
            memory_size=memory_size,
            hidden_size=hidden_size,
            output_size=output_size,
            projection_size=projection_size,
            n_layers=n_layers,
        )
        self.decoder = DecoderTinyTransducer(
            num_classes=num_classes,
            hidden_state_dim=decoder_hidden_state_dim,
            output_dim=output_size,
            kernel_size=decoder_kernel_size
        )
        self.fc = nn.Linear(decoder_hidden_state_dim, num_classes, bias=False)

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs

    def forward(
            self,
            inputs: Tensor,
            targets: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, _ = self.encoder(inputs)
        decoder_outputs, _ = self.decoder(targets)
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs
