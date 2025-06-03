# For Data Processing
import numpy as np
import pandas as pd
import os
import shutil
import random
from tqdm import tqdm

# For Speech Processing
import librosa
import soundfile as sf


# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict

# If you need other libraries to be imported write it here
from pazhvak_hu.dataset.pazhvak_dataset import PazhvakDataset


# Writing prerprocessing function for dataset
class DatasetBalancer:
    def __init__(self, dataset: 'PazhvakDataset'):
        self.dataset = dataset
        self.metadata = self._clean_metadata(dataset.metadata)

    def _clean_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata.dropna(subset=['Folder Number', 'Persian Word'])

    def _collect_all_word_samples(self) -> dict:
        """Collect all audio files per Persian word from all folders."""
        word_samples = defaultdict(list)

        for _, row in self.metadata.iterrows():
            folder_num = int(row['Folder Number'])
            word = row['Persian Word']
            folder_range = self.dataset._find_word_range(folder_num)
            folder_path = os.path.join(self.dataset.dataset_path, folder_range, str(folder_num))
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith(('.wav', '.mp3', '.flac')):
                        word_samples[word].append((folder_path, f))
        return word_samples

    def _augment_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply one of several audio augmentation techniques for up sampling."""
        choice = random.choice(['noise', 'pitch', 'stretch'])
        if choice == 'noise':
            noise = np.random.normal(0, 0.005, y.shape)
            return y + noise
        elif choice == 'pitch':
            return librosa.effects.pitch_shift(y, sr, n_steps=random.choice([-2, -1, 1, 2]))
        elif choice == 'stretch':
            rate = random.uniform(0.8, 1.2)
            y_stretched = librosa.effects.time_stretch(y, rate)
            return y_stretched[:len(y)]  # Trim/pad to original length
        return y

    def balance_dataset(self, output_dataset_path: str) -> pd.DataFrame:
        os.makedirs(output_dataset_path, exist_ok=True)
        word_samples = self._collect_all_word_samples()

        avg_count = int(np.mean([len(v) for v in word_samples.values()]))
        print("The average voice count is: ",avg_count)
        new_metadata = []

        folder_counter = 1

        for word, samples in tqdm(word_samples.items(), desc="Balancing dataset"):
            total_needed = avg_count

            if len(samples) >= total_needed:
                selected = random.sample(samples, total_needed)
            else:
                selected = samples.copy()
                while len(selected) < total_needed:
                    src_folder, src_file = random.choice(samples)
                    src_path = os.path.join(src_folder, src_file)
                    try:
                        y, sr = librosa.load(src_path, sr=None)
                        y_aug = self._augment_audio(y, sr)
                        aug_id = f"{folder_counter}_{len(selected) + 1}"
                        tmp_filename = f"tmp_aug_{aug_id}.wav"
                        tmp_path = os.path.join(output_dataset_path, tmp_filename)
                        sf.write(tmp_path, y_aug, sr)
                        selected.append((output_dataset_path, tmp_filename))

                    except:
                        continue

            # Assign new folder and copy files with new naming
            folder_range = self._find_new_folder_range(folder_counter)
            folder_path = os.path.join(output_dataset_path, folder_range, str(folder_counter))
            os.makedirs(folder_path, exist_ok=True)

            for idx, (src_folder, filename) in enumerate(selected):
                new_filename = f"FA_16000_{idx + 1:04d}.wav"
                dst_path = os.path.join(folder_path, new_filename)

                if filename.startswith("tmp_aug_"):  # handle augmented audio
                    shutil.move(os.path.join(src_folder, filename), dst_path)
                else:
                    src_path = os.path.join(src_folder, filename)
                    shutil.copy2(src_path, dst_path)

            new_metadata.append((folder_counter, word))
            folder_counter += 1

        # Save new label.xlsx
        label_df = pd.DataFrame(new_metadata, columns=["Folder Number", "Persian Word"])
        label_path = os.path.join(output_dataset_path, "P1993818924117826_1247741184149835.xlsx")
        label_df.to_excel(label_path, index=False)

        return label_df

    def _find_new_folder_range(self, folder_num: int) -> str:
        """Helper to generate folder range path like '1-400', '401-800', etc."""
        range_size = 400
        start = ((folder_num - 1) // range_size) * range_size + 1
        end = start + range_size - 1
        return f"{start}-{end}"
