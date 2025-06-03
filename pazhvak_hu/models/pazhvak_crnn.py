# For DL, Classifier,... Model (Sklearn, pytorch, keras, Tensorflow,...)
import torch
import torch.nn as nn
import torch.nn.functional as F

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict


# Write the CRNN model function
class CRNN(nn.Module):
    def __init__(self, input_shape, num_classes=3642, bidirectional=True, use_attention=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Convolutional Layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # Add adaptive pooling to ensure consistent feature size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 17))  # Fix width to 17

        # Calculate feature dimension (512 channels * 17 width)
        self.lstm_input_size = 512 * 17

        # LSTM Layers with dropout=0 (since we have our own dropout)
        self.lstm1 = nn.LSTM(self.lstm_input_size, 512, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.norm1 = nn.LayerNorm(512 * (2 if bidirectional else 1))
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(512 * (2 if bidirectional else 1), 256, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.norm2 = nn.LayerNorm(256 * (2 if bidirectional else 1))
        self.dropout2 = nn.Dropout(0.5)

        self.lstm3 = nn.LSTM(256 * (2 if bidirectional else 1), 128, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.norm3 = nn.LayerNorm(128 * (2 if bidirectional else 1))
        self.dropout3 = nn.Dropout(0.5)

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(128 * (2 if bidirectional else 1), 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Softmax(dim=1)
            )

        # Fully Connected Layers
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

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if 'lstm' in name and 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'attention' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # Handle input dimensions
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim if missing
        elif x.dim() == 5:
            x = x.squeeze(1)
        
        x = self.conv(x)  # ConvNet
        x = self.adaptive_pool(x)  # Ensure consistent feature size
        b, c, h, w = x.size()

        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, h, c, w)
        x = x.view(b, h, -1)  # (batch, h, c*w)

        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = self.dropout2(x)

        x, (hn, _) = self.lstm3(x)
        x = self.norm3(x)
        x = self.dropout3(x)

        if self.use_attention:
            weights = self.attention(x)  # (batch, seq_len, 1)
            x = torch.sum(x * weights, dim=1)  # Weighted sum
        else:
            # Use last hidden state
            if self.bidirectional:
                x = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                x = hn[-1]

        x = self.dense(x)
        return F.log_softmax(x, dim=1)

