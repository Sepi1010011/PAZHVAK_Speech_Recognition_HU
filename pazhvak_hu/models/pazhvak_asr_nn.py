# For DL, Classifier,... Model (Sklearn, pytorch, keras, Tensorflow,...)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, Wav2Vec2Model

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict



# Downloading Model
asr_model_name = "m3hrdadfi/wav2vec2-large-xlsr-persian"


# Write the pipeline function here
class ASRClassifierPipeline(nn.Module):
    def __init__(self, asr_model_path: str, num_classes: int = 3):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(asr_model_path)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_path)
        hidden_size = self.asr_model.config.hidden_size # 1024

        # CNN input = (batch, channel, height, width)
        self.cnn_feature = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),

            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6)) # (batch, channel, 6, 6)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, num_classes)
        )

    def forward(self, audio_input, sampling_rate=16000):
        # audio_input: 1D numpy array or tensor
        if isinstance(audio_input, torch.Tensor):
            audio_input = audio_input.numpy()

        inputs = self.processor(audio_input, sample_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.asr_model.device)

        with torch.no_grad():
            features = self.asr_model(input_values).last_hidden_state

        # CNN expects [B, C, H, W] → convert [B, T, F] → [B, 1, T, F] , T: time_steps, F: features
        x = features.unsqueeze(1)
        x = self.cnn_feature(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)

        return probs
