import math
import torch

from torch import nn, Tensor
from .TransformerLayerCustom import TransformerEncoderCustom, TransformerEncoderLayerCustom, \
    TransformerDecoderLayerCustom, TransformerDecoderCustom


class TransformerModelEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5, need_weights=False):
        super().__init__()
        self.model_type = 'TransformerEncoder'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayerCustom(d_model, nhead, d_hid, dropout, need_weights=need_weights)
        self.encoder = TransformerEncoderCustom(encoder_layers, nlayers, need_weights=need_weights)
        self.d_model = d_model

        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output, attention_weights = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output, attention_weights


class TransformerModelDecoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5, need_weights=False):
        super().__init__()
        self.model_type = 'TransformerDecoder'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayerCustom(d_model, nhead, d_hid, dropout, need_weights=need_weights)
        self.decoder = TransformerDecoderCustom(decoder_layers, nlayers, need_weights=need_weights)
        self.d_model = d_model

        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_input, encoder_output, tgt_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        decoder_input = self.pos_encoder(decoder_input)
        output, att_weights = self.decoder(tgt=decoder_input, memory=encoder_output, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=None,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output, att_weights


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
