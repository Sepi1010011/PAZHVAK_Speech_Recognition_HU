# For DL, Classifier,... Model (Sklearn, pytorch, keras, Tensorflow,...)
import torch
import torch.nn as nn
import torch.nn.functional as F

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict



# Write the LSTM light model function
class LSTMLight(nn.Module):
    def __init__(self, input_size, num_classes=3642, bidirectional=True, use_attention=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        hidden1 = 256
        hidden2 = 128
        bi_factor = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0
        )
        self.layernorm1 = nn.LayerNorm(hidden1 * bi_factor)

        self.lstm2 = nn.LSTM(
            input_size=hidden1 * bi_factor,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0
        )
        self.layernorm2 = nn.LayerNorm(hidden2 * bi_factor)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden2 * bi_factor, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            self.attention_softmax = nn.Softmax(dim=1)

        self.td_dense = nn.Sequential(
            nn.Linear(hidden2 * bi_factor, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # biases
                continue
            if 'weight' in name and 'lstm' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight' in name and 'attention' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x, _ = self.lstm1(x)
        x = self.layernorm1(x)

        x, (hn, cn) = self.lstm2(x)
        x = self.layernorm2(x)

        if self.use_attention:
            energy = self.attention(x)  # (batch, seq_len, 1)
            weights = self.attention_softmax(energy)
            x = torch.sum(x * weights, dim=1)  # (batch, hidden_size)
        else:
            if self.bidirectional:
                x = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                x = hn[-1]

        x = self.td_dense(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)



# Write the LSTM heavy model function
class LSTMHeavy(nn.Module):
    def __init__(self, input_size, num_classes=3642, bidirectional=True, use_attention=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if bidirectional else 0
        )
        self.layernorm1 = nn.LayerNorm(512 * (2 if bidirectional else 1))
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(
            input_size=512 * (2 if bidirectional else 1),
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if bidirectional else 0
        )
        self.layernorm2 = nn.LayerNorm(256 * (2 if bidirectional else 1))
        self.dropout2 = nn.Dropout(0.5)

        self.lstm3 = nn.LSTM(
            input_size=256 * (2 if bidirectional else 1),
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if bidirectional else 0
        )
        self.layernorm3 = nn.LayerNorm(128 * (2 if bidirectional else 1))
        self.dropout3 = nn.Dropout(0.5)

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(128 * (2 if bidirectional else 1), 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )

        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(128 * (2 if bidirectional else 1), 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # Skip biases and 1D parameters
                continue
            if 'weight' in name and 'lstm' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight' in name and 'attention' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.layernorm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.layernorm2(x)
        x = self.dropout2(x)

        x, (hn, cn) = self.lstm3(x)
        x = self.layernorm3(x)
        x = self.dropout3(x)

        # Attention mechanism
        if self.use_attention:
            attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
            x = torch.sum(x * attention_weights, dim=1)  # (batch_size, hidden_size)
        else:
            # Use last hidden state if no attention
            if self.bidirectional:
                x = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch_size, hidden_size*2)
            else:
                x = hn[-1]  # (batch_size, hidden_size)

        # Dense layers
        x = self.dense(x)

        return F.log_softmax(x, dim=1)
