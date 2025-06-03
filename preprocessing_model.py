# preprocessing_model.py
import os
import numpy as np
import librosa
import soundfile as sf
from pazhvak_hu.preprocessing.audio_preprocessing import AudioPreprocessor
from pazhvak_hu.feature_extraction.feature_extraction import (
    MelSpectrogramExtractor, 
    MFCCFeatureExtractor
)

class SingleAudioProcessor:
    """Processes single audio files for prediction"""
    
    def __init__(self, target_sample_rate=16000, target_db=-20, trim_top_db=30):
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=target_sample_rate,
            target_db=target_db,
            trim_top_db=trim_top_db
        )
        self.mel_extractor = MelSpectrogramExtractor()
        self.mfcc_extractor = MFCCFeatureExtractor()
    
    def process_audio(self, file_path: str, feature_type: str = "mfcc") -> np.ndarray:
        """
        Process a single audio file and extract features
        
        Args:
            file_path: Path to audio file
            feature_type: Type of features to extract ('mfcc' or 'mel')
            
        Returns:
            Extracted features as numpy array
        """
        # Preprocess audio
        audio = self.preprocessor.process(file_path)

        # Extract features
        if feature_type.lower() == "mel":
            features = self.mel_extractor.extract(audio)
        elif feature_type.lower() == "mfcc":
            features = self.mfcc_extractor.extract(audio)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        return (features, audio)

