# For DL, Classifier,... Model (Sklearn, pytorch, keras, Tensorflow,...)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict



# Write the CNN light model function
class CNNLight(nn.Module):
    def __init__(self, input_channels, num_classes=3642):
        super().__init__()

        # Convolutional blocks with GroupNorm instead of BatchNorm
        self.conv_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 32),  # Using GroupNorm instead of BatchNorm
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1)
        )

        self.conv_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure proper input dimensions
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Input normalization
        x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)

        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
    
    
# Write the CNN heavy model function
class CNNHeavy(nn.Module):
    def __init__(self, input_channels, num_classes=3642):
        super().__init__()

        # Expanded convolutional blocks with more layers and channels
        self.conv_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2)
        )

        self.conv_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Calculate the actual output channels (256) and feature size
        self.fc_input_size = 256 * 7 * 7  # 256 channels * 7x7 spatial dimensions

        # Larger fully connected layers - FIXED input size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),  # Use calculated size
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Normalize per-channel
        if x.size(1) > 1:  # Only normalize if multiple channels
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
        else:  # Single channel
            x = (x - x.mean()) / (x.std() + 1e-6)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)

        return F.log_softmax(x, dim=1)