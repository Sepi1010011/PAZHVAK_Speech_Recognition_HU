import numpy as np
import pandas as pd
import os
import shutil
import random
from zipfile import ZipFile
from sklearn.utils import shuffle
from tqdm import tqdm
from pathlib import Path
from sklearn.utils import resample
from pathlib import Path

import librosa
import IPython.display as ipd
import IPython.display as display

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict

from dataset_analyzer import AudioAnalyzer, DatasetStatistics, WordAnalyzer
from sklearn.model_selection import train_test_split

class PazhvakDataset:
    """A class to handle Persian audio dataset with metadata and folder structure."""

    def __init__(self, dataset_path: str, metadata_file: str):
        """
        Args:
            dataset_path (str): Root path of the dataset
            metadata_file (str): Path to metadata file (Excel/CSV) relative to dataset_path
        """
        self.dataset_path = dataset_path
        self.metadata_path = os.path.join(dataset_path, metadata_file)
        self.metadata_file = metadata_file

        # Load metadata
        self.metadata = self._load_metadata()
        self.word_range_folders = self._extract_word_ranges(metadata_file)

        self.audio_analyzer = AudioAnalyzer
        self.word_analyzer = WordAnalyzer(self)
        self.stats = DatasetStatistics(self)

        # Initialize splits
        self.train_df: pd.DataFrame = None
        self.val_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata file (supports .csv, .xlsx)"""
        if self.metadata_path.endswith('.csv'):
            return pd.read_csv(self.metadata_path)

        elif self.metadata_path.endswith('.xlsx'):
            return pd.read_excel(self.metadata_path)

        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

    def _extract_word_ranges(self, metadata_file: str) -> list:
        """Extract folder ranges from dataset path"""
        list_of_folders = os.listdir(self.dataset_path)
        list_of_folders.remove(metadata_file)
        return list_of_folders

    def __len__(self) -> int:
        """Returns total number of audio folders"""
        return len(self.word_range_folders)

    def get_all_samples(self) -> pd.DataFrame:
        """Gather all audio samples from the dataset into a DataFrame"""
        all_rows = []

        print("Gathering all audio files and labels...")
        for idx, row in self.metadata.iterrows():
            try:
                folder_number = row['Folder Number']
                label = row['Persian Word'] if 'Persian Word' in row else row['Persian Word']  # Handle typo
                range_folder = self._find_word_range(int(folder_number))

                full_path = os.path.join(self.dataset_path, range_folder, str(folder_number))
                if not os.path.exists(full_path):
                    continue

                for fname in os.listdir(full_path):
                    if fname.endswith((".npy", ".wav", ".mp3", ".flac")):
                        all_rows.append({
                            'file_path': os.path.join(full_path, fname),
                            'file_name': fname,
                            'label': label,
                            'folder_number': folder_number,
                            'original_index': idx  # Keep track of original metadata index
                        })
            except Exception as e:
                print(f"Error in row {idx}: {e}")

        return pd.DataFrame(all_rows)

    def get_audio_sample(self, index: int = None, num_samples: int = 1) -> list:
        """
        Get audio sample(s) by index or random

        Args:
            index (int): Specific index to get (None for random)
            num_samples (int): Number of samples to return (default: 1)

        Returns:
            list: List of tuples (audio_data, sample_rate, word, metadata_row)
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        samples = []

        for _ in range(num_samples):
            current_index = index if index is not None else random.randint(0, len(self.metadata)-1)

            try:
                row = self.metadata.iloc[current_index]
                folder_num = row['Folder Number']
                word_range = self._find_word_range(folder_num)

                audio_folder = os.path.join(
                    self.dataset_path,
                    word_range,
                    str(folder_num))

                if not os.path.exists(audio_folder):
                    continue

                audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
                if not audio_files:
                    continue

                audio_file = os.path.join(audio_folder, random.choice(audio_files))
                y, sr = librosa.load(audio_file, sr=None)

                samples.append((y, sr, row['Persian Word'], row))
            except Exception as e:
                print(f"Error loading sample {current_index}: {str(e)}")
                continue

        return samples

    def _find_word_range(self, folder_number: int) -> str:
        """Find which range folder the number belongs to"""
        for word_range in self.word_range_folders:
            start, end = map(int, word_range.split('-'))
            if start <= folder_number <= end:
                return word_range
        raise ValueError(f"Folder number {folder_number} out of range")

    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1,
                     random_state: int = 42) -> None:
        """
        Split dataset into train/val/test sets while maintaining class balance

        Args:
            test_ratio (float): Proportion for test set
            val_ratio (float): Proportion for validation set
            random_state (int): Random seed for reproducibility
        """
        all_df = self.get_all_samples()

        print(f"Splitting dataset (total samples: {len(all_df)})...")
        # First split: Train+Val vs Test
        train_val_df, test_df = train_test_split(
            all_df,
            test_size=test_size,
            stratify=all_df['label'],
            random_state=random_state
        )

        # Calculate adjusted validation size based on remaining train+val portion
        val_adjusted = val_size / (1.0 - test_size)

        # Second split: Train vs Val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_adjusted,
            stratify=train_val_df['label'],
            random_state=random_state
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

    def export_splits(self, output_path: str) -> None:
        """
        Export train/val/test splits to disk with proper folder structure

        Args:
            output_path (str): Path to save the splits
        """
        if not all([self.train_df is not None, self.val_df is not None, self.test_df is not None]):
            raise ValueError("Splits not initialized. Call split_dataset() first")

        def save_split(df: pd.DataFrame, split: str):
            split_path = os.path.join(output_path, split, "voices")
            os.makedirs(split_path, exist_ok=True)
            entries = []

            for _, row in df.iterrows():
                new_name = f"{row['label'].replace(' ', '_')}_{row['folder_number']}_{row['file_name']}"
                dst_path = os.path.join(split_path, new_name)

                shutil.copyfile(row['file_path'], dst_path)

                entries.append({
                    "file_name": new_name,
                    "label": row['label'],
                    "folder_number": row['folder_number']
                })

            pd.DataFrame(entries).to_excel(
                os.path.join(output_path, split, "labels.xlsx"),
                index=False
            )
            print(f"[{split}] -> {len(entries)} samples written to {split}/voices and labels.xlsx")

        save_split(self.train_df, "train")
        save_split(self.val_df, "val")
        save_split(self.test_df, "test")

    def play_random_sample(self, split: str = None, num_samples: int = 1) -> None:
        """
        Play random audio samples from specified split

        Args:
            split (str): One of 'train', 'val', 'test', or None for entire dataset
            num_samples (int): Number of samples to play (default: 1)
        """
        if split not in [None, 'train', 'val', 'test']:
            raise ValueError("split must be one of: None, 'train', 'val', 'test'")

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        if split and not getattr(self, f"{split}_indices"):
            raise ValueError(f"{split} split not initialized. Call split_dataset() first")

        for i in range(num_samples):
            index = None
            if split == 'train':
                index = random.choice(self.train_indices)
            elif split == 'val':
                index = random.choice(self.val_indices)
            elif split == 'test':
                index = random.choice(self.test_indices)
            else:
                index = random.randint(0, len(self.metadata)-1)

            try:
                y, sr, word, metadata = self.get_audio_sample(index)[0]  # Get first (and only) sample
                print(f"\nSample #{i+1}/{num_samples}")
                print(f"Word: {word}")
                print(f"Folder: {metadata['Folder Number']}")
                print(f"Audio length: {len(y)/sr:.2f} seconds")
                display(ipd.Audio(y, rate=sr))
            except Exception as e:
                print(f"Error playing sample: {str(e)}")
                continue