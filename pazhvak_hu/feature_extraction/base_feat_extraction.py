# For Data Processing
import numpy as np
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
from feature_extraction import FeatureExtractor


# ##############################################################
# _________________________Handling_____________________________
# ##############################################################

class FeatureSaver:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir

    def save(self, features: np.ndarray, rel_path: str):
        full_path = os.path.join(self.base_output_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, features)


class HandFeatureExtraction:
    def __init__(self, dataset: 'PazhvakDataset', extractor: FeatureExtractor, feature_name: str):
        self.dataset = dataset
        self.extractor = extractor
        self.feature_name = feature_name
        self.output_dir = os.path.join(dataset.dataset_path, "features", self.feature_name)
        self.saver = FeatureSaver(self.output_dir)

    def feature_extraction(self):
        # First count total files for progress bar
        total_files = 0
        for range_folder in os.listdir(self.dataset.dataset_path):
            range_path = os.path.join(self.dataset.dataset_path, range_folder)
            if not os.path.isdir(range_path):
                continue
            for speaker_id in os.listdir(range_path):
                speaker_path = os.path.join(range_path, speaker_id)
                if not os.path.isdir(speaker_path):
                    continue
                for audio_file in os.listdir(speaker_path):
                    if audio_file.lower().endswith(('.wav', '.mp3', '.flac')):
                        total_files += 1

        # Main processing with tqdm
        with tqdm(total=total_files, desc=f"Extracting {self.feature_name}", unit="file") as pbar:
            for range_folder in os.listdir(self.dataset.dataset_path):
                range_path = os.path.join(self.dataset.dataset_path, range_folder)
                if not os.path.isdir(range_path):
                    pbar.write(f"Skipping non-folder: {range_folder}")
                    continue

                for speaker_id in os.listdir(range_path):
                    speaker_path = os.path.join(range_path, speaker_id)
                    if not os.path.isdir(speaker_path):
                        pbar.write(f"Skipping non-folder: {speaker_id}")
                        continue

                    for audio_file in os.listdir(speaker_path):
                        if not audio_file.lower().endswith(('.wav', '.mp3', '.flac')):
                            continue

                        audio_path = os.path.join(speaker_path, audio_file)
                        try:
                            features = self.extractor.extract(audio_path)
                            feature_sub_dir = os.path.join(self.output_dir, range_folder, speaker_id)
                            os.makedirs(feature_sub_dir, exist_ok=True)
                            filename = os.path.splitext(audio_file)[0] + ".npy"
                            self.saver.save(features, os.path.join(range_folder, speaker_id, filename))
                            pbar.update(1)
                        except Exception as e:
                            pbar.write(f"[✘] Failed to extract {audio_file}: {e}")
                            pbar.update(1)
        shutil.copy("/content/balanced_pazhvak/P1993818924117826_1247741184149835.xlsx", os.path.join(self.output_dir, "P1993818924117826_1247741184149835.xlsx"))