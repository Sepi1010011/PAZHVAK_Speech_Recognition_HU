# For Data Processing
import numpy as np
import os


#  For Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# For Speech Processing
import librosa
import soundfile as sf
import noisereduce as nr


# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict


class AudioPreprocessor:
    """Main class for audio preprocessing pipeline"""
    
    def __init__(self, target_sample_rate: int = 16000, target_db: int = -20, trim_top_db: int = 30):
        self.target_sample_rate = target_sample_rate
        self.target_db = target_db
        self.trim_top_db = trim_top_db
        
    def process(self, input_path: Union[str, np.ndarray], save_wav: bool = False, 
                save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Main processing method that applies the full pipeline"""
        try:
            # Validate input
            self._validate_input(input_path)
            
            # Load/resample audio
            y, sr = self._load_audio(input_path)
            
            # Apply processing steps
            y = self._apply_processing_pipeline(y, sr)
            
            # Optional save
            if save_wav and save_path:
                self._save_audio(y, sr, save_path)
                
            return y
            
        except Exception as e:
            self._handle_error(input_path, e)
            return None

    def _validate_input(self, input_path: Union[str, np.ndarray]):
        """Validate input before processing"""
        if isinstance(input_path, str) and not os.path.exists(input_path):
            raise FileNotFoundError(f"File {input_path} does not exist")

    def _load_audio(self, input_path: Union[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        """Load and resample audio"""
        if isinstance(input_path, str):
            y, sr = librosa.load(input_path, sr=self.target_sample_rate, mono=True)
        else:
            y = input_path
            sr = self.target_sample_rate
            
        if y is None or len(y) == 0:
            raise ValueError("Empty audio data")
            
        return y, sr

    def _apply_processing_pipeline(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply all processing steps in sequence"""
        processors = [
            NoiseReducer(),
            SilenceTrimmer(self.trim_top_db),
            PeakNormalizer(),
            LoudnessNormalizer(self.target_db),
        ]
        
        for processor in processors:
            y = processor.process(y, sr)
            
        self._validate_output(y)
        return y

    def _validate_output(self, y: np.ndarray):
        """Validate processed audio"""
        if np.any(np.isnan(y)):
            raise ValueError("NaN values detected after preprocessing")
        if np.all(y == 0):
            raise ValueError("Silent audio after preprocessing")

    def _save_audio(self, y: np.ndarray, sr: int, save_path: str):
        """Save processed audio to file"""
        sf.write(save_path, y, sr)

    def _handle_error(self, input_path: Union[str, np.ndarray], error: Exception):
        """Handle processing errors"""
        input_name = input_path if isinstance(input_path, str) else "audio_array"
        print(f"Failed to process {input_name}: {error}")


class AudioProcessor(ABC):
    """Abstract base class for audio processing steps"""
    
    @abstractmethod
    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        pass


class NoiseReducer(AudioProcessor):
    """Reduce background noise"""
    
    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        return nr.reduce_noise(y=y, sr=sr)


class SilenceTrimmer(AudioProcessor):
    """Trim silence from audio"""
    
    def __init__(self, top_db: int = 30):
        self.top_db = top_db
        
    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        yt, _ = librosa.effects.trim(y, top_db=self.top_db)
        return yt


class PeakNormalizer(AudioProcessor):
    """Normalize audio to peak amplitude"""
    
    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        peak = np.max(np.abs(y))
        return y / peak if peak > 0 else y


class LoudnessNormalizer(AudioProcessor):
    """Normalize audio to target loudness"""
    
    def __init__(self, target_db: int = -20):
        self.target_db = target_db
        
    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        rms = np.sqrt(np.mean(y ** 2))
        target_rms = 10 ** (self.target_db / 20)
        return y * (target_rms / (rms + 1e-6))

