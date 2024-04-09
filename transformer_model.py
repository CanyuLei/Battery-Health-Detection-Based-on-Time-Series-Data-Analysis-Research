import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        return self.dropout(inputs + self.pe[:, : inputs.size(1)])


class Model(nn.Module):
    def __init__(self, inputs_size, outputs_size):
        super(Model, self).__init__()
        self.dim_up = nn.Linear(inputs_size, 128)  # 升维层
        self.positional_encoding = PositionalEncoding(128, 0.2)  # 位置编码层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)  # transformer encoder 单层
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)  # transformer encoder 多层
        self.predict = nn.Linear(128, outputs_size)  # 预测层
        self.activation = nn.PReLU()  # 激活层
        self.dropout = nn.Dropout(0.2)  # dropout层

    def transformer_encoder_forward(self, inputs):  # torch.Size([B, 32, 128])
        outputs = inputs.permute(1, 0, 2)  # torch.Size([32, B, 128])
        outputs = self.transformer_encoder(outputs)  # torch.Size([32, B, 128])
        outputs = outputs.permute(1, 0, 2)  # torch.Size([B, 32, 128])
        outputs = outputs.mean(dim=1)  # torch.Size([B, 128])
        return outputs

    def forward(self, inputs):  # torch.Size([B, 32, 2])
        outputs = self.dim_up(inputs)  # torch.Size([B, 32, 128])
        outputs = self.positional_encoding(outputs)  # torch.Size([B, 32, 128])

        outputs = self.transformer_encoder_forward(outputs)  # torch.Size([B, 128])
        outputs = self.activation(outputs)  # torch.Size([B, 128])

        outputs = self.dropout(outputs)  # torch.Size([B, 128])
        outputs = self.predict(outputs)  # torch.Size([B, 1])
        return outputs
