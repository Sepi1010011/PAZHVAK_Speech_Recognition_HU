# For Data Processing
import os
import shutil
from tqdm import tqdm

# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict

# If you need other libraries to be imported write it here
from pazhvak_hu.dataset.pazhvak_dataset import PazhvakDataset
from audio_preprocessing import AudioPreprocessor
from pazhvak_hu.feature_extraction.feature_extraction import FeatureExtractor
from pazhvak_hu.feature_extraction.base_feat_extraction import FeatureSaver

TARGET_SAMPLE_RATE = 16000


class AudioProcessingPipeline:
    def __init__(self,
                 dataset: PazhvakDataset,
                 feature_extractor: FeatureExtractor,
                 save_wav_after_preprocess: bool = False,
                 save_dir_name: str = "processed",
                 feature_name: str = "features"):

        self.dataset = dataset
        self.dataset_root = dataset.dataset_path
        self.feature_extractor = feature_extractor
        self.save_wav = save_wav_after_preprocess
        self.wav_output_root = os.path.join(self.dataset_root, save_dir_name)
        self.feature_output_root = os.path.join(self.dataset_root, feature_name)
        os.makedirs(self.wav_output_root, exist_ok=True)
        self.saver = FeatureSaver(self.feature_output_root)
        self.preprocessor = AudioPreprocessor()

    def _get_audio_files(self):
        audio_files = []
        for root, _, files in os.walk(self.dataset_root):
            if root.startswith(self.wav_output_root) or root.startswith(self.feature_output_root):
                continue
            for file in files:
                if file.lower().endswith((".wav", ".flac", ".mp3")):
                    audio_files.append((root, file))
        return audio_files

    def run(self):
        audio_files = self._get_audio_files()

        for root, file in tqdm(audio_files, desc="Processing audio files"):
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, self.dataset_root)
            new_wav_path = os.path.join(self.wav_output_root, rel_path)
            os.makedirs(os.path.dirname(new_wav_path), exist_ok=True)

            # Preprocess
            preprocessed_audio = self.preprocessor.process(
                input_path=file_path,
                save_wav=self.save_wav,
                save_path=new_wav_path if self.save_wav else None
            )

            if preprocessed_audio is None:
                print(f"[✘] audio file is None {file_path}")
                continue

            # Feature Extraction
            try:
                features = self.feature_extractor.extract(preprocessed_audio)
                feature_filename = os.path.splitext(rel_path)[0] + ".npy"
                self.saver.save(features, feature_filename)
            except Exception as e:
                print(f"[✘] Failed to extract features from {file_path}: {e}")

        # Copy Excel file if needed
        self._copy_excel_file()

    def _copy_excel_file(self):
        excel_file = "P1993818924117826_1247741184149835.xlsx"
        src_path = os.path.join(self.dataset_root, excel_file)
        
        if os.path.exists(src_path):
            if self.save_wav:
                shutil.copy(src_path, os.path.join(self.wav_output_root, excel_file))
            shutil.copy(src_path, os.path.join(self.feature_output_root, excel_file))