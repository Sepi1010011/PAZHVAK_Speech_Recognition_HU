# For Data Processing
import numpy as np

# For Speech Processing
import torchaudio
import librosa

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, Wav2Vec2Model

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict

# If you need other libraries to be imported write it here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ##############################################################
# ___________________________Features___________________________
# ##############################################################
class FeatureExtractor(ABC):
    def extract(self, audio_input: Union[str, np.ndarray], sr: int = 16000) -> np.ndarray:
        if isinstance(audio_input, str):
            y, _ = librosa.load(audio_input, sr=sr)
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
        else:
            raise ValueError("Unsupported audio input type. Must be a file path or numpy array.")
        return self._extract_from_array(y, sr)

    @abstractmethod
    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        pass


class MFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, sr: int = 16000, n_mfcc: int = 40):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        return mfcc.T


class MelSpectrogramExtractor(FeatureExtractor):
    def __init__(self, sr: int = 16000, n_mels: int = 128):
        self.sr = sr
        self.n_mels = n_mels

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db


class SpectralContrastExtractor(FeatureExtractor):
    def __init__(self, sr: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return contrast.T


class ZeroCrossingRateExtractor(FeatureExtractor):
    def __init__(self, sr: int = 16000, frame_length: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.frame_length, hop_length=self.hop_length
        )
        return zcr.T
    
    
class Wav2VecFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name="m3hrdadfi/wav2vec2-large-xlsr-persian", device=None):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.sr = 16000  # Target sample rate

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        inputs = self.processor(y, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

        return hidden_states.cpu().numpy()



class Wav2VecBaseTorchaudioFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = self.bundle.get_model()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = self.bundle.sample_rate

    def _extract_from_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)

        # Convert to mono if needed and to tensor shape: (1, time)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        waveform = torch.tensor(y).unsqueeze(0)

        with torch.inference_mode():
            features = self.model.extract_features(waveform)[0][0]  # First layer

        return features.numpy()

