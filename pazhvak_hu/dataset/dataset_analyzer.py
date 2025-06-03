# For Data Processing
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

#  For Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# For object oriented in python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from collections import defaultdict

import librosa

from pazhvak_dataset import PazhvakDataset


class IAudioAnalyzer(ABC):
    @abstractmethod
    def analyze(self, signal: np.ndarray, sample_rate: int) -> dict:
        pass

    @abstractmethod
    def plot(self, signal: np.ndarray, sample_rate: int):
        pass

#-------------------------------------------------------------
#<<<<<<<<<<<<<<<<<<<Audio Analysis>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-------------------------------------------------------------

@dataclass
class AudioSample:
    signal: np.ndarray
    sample_rate: int
    metadata: dict = None

class AudioAnalyzer:
    def __init__(self, audio_sample: AudioSample):
        self.audio = audio_sample

    def raw_waveform_plot(self):
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y=self.audio.signal, sr=self.audio.sample_rate)
        plt.title("Raw Waveform")
        plt.show()

    def fft_spectrum_plot(self):
        fft = np.fft.fft(self.audio.signal)
        freq = np.fft.fftfreq(len(self.audio.signal), 1/self.audio.sample_rate)
        plt.figure(figsize=(12, 4))
        plt.plot(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
        plt.title("FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

    def stft_spectrogram_plot(self, scale: str = "log"):
        D = librosa.stft(self.audio.signal, n_fft=2048, hop_length=512)
        s_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(s_db, sr=self.audio.sample_rate,
                                hop_length=512, x_axis='time', y_axis=scale)
        plt.colorbar(format='%+2.0f dB')
        plt.title("STFT Spectrogram")
        plt.show()

    def mel_spectrogram_plot(self, n_mels: int = 80):
        S = librosa.feature.melspectrogram(y=self.audio.signal, sr=self.audio.sample_rate, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(S_db, sr=self.audio.sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.show()
        return S_db

    def mfcc_plot(self, n_mfcc: int = 13):
        mfccs = librosa.feature.mfcc(y=self.audio.signal, sr=self.audio.sample_rate, n_mfcc=n_mfcc)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfccs, sr=self.audio.sample_rate, x_axis='time')
        plt.colorbar()
        plt.title("MFCC Features")
        plt.show()


#-------------------------------------------------------------
#<<<<<<<<<<<<<<<<<<<<Word Analysis>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-------------------------------------------------------------

class WordAnalyzer:
    def __init__(self, dataset: 'PazhvakDataset'):
        self.dataset = dataset

    def word_count(self) -> pd.DataFrame:
        word_stats = []

        for word, group in self.dataset.metadata.groupby('Persian Word'):
            voice_count = 0
            folder_count = len(group)

            for _, row in group.iterrows():
                folder_num = row['Folder Number']
                word_range = self.dataset._find_word_range(folder_num)
                folder_path = os.path.join(self.dataset.dataset_path, word_range, str(folder_num))

                if os.path.exists(folder_path):
                    voice_count += len([f for f in os.listdir(folder_path)
                                     if f.endswith(('.wav', '.mp3', '.flac'))])

            word_stats.append({
                'Persian Word': word,
                'Voice Count': voice_count,
                'Folder Count': folder_count
            })

        return pd.DataFrame(word_stats)

    def most_common_words(self, top_n: int = 10) -> pd.DataFrame:
        word_stats = self.word_count()
        return word_stats.sort_values('Voice Count', ascending=False).head(top_n)

    def least_common_words(self, top_n: int = 10) -> pd.DataFrame:
        word_stats = self.word_count()
        return word_stats.sort_values('Voice Count').head(top_n)

    def plot_voice_distribution(self):
        word_stats = self.word_count()
        voice_counts = word_stats['Voice Count'].value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        voice_counts.plot(kind='bar')
        plt.title("Distribution of Voice Counts per Word")
        plt.xlabel("Number of Voices")
        plt.ylabel("Number of Words")
        plt.grid(True)
        plt.show()
        
#-------------------------------------------------------------
#<<<<<<<<<<<<<<<<<<<Dataset Statistics>>>>>>>>>>>>>>>>>>>>>>>>
#-------------------------------------------------------------

class DatasetStatistics:
    def __init__(self, dataset: 'PazhvakDataset'):
        self.dataset = dataset

    def analyze_durations(self) -> dict:
        durations = []
        total_files = 0

        for _, row in tqdm(self.dataset.metadata.iterrows(), total=len(self.dataset.metadata), desc="Analyzing durations"):
            folder_num = row['Folder Number']
            word_range = self.dataset._find_word_range(folder_num)
            folder_path = os.path.join(self.dataset.dataset_path, word_range, str(folder_num))

            if os.path.exists(folder_path):
                audio_files = [f for f in os.listdir(folder_path)
                             if f.endswith(('.wav', '.mp3', '.flac'))]
                total_files += len(audio_files)

                for file in audio_files:
                    try:
                        y, sr = librosa.load(os.path.join(folder_path, file), sr=None)
                        durations.append(len(y)/sr)
                    except:
                        continue

        if not durations:
            return {}

        avg_dur = sum(durations)/len(durations)
        max_dur = max(durations)
        min_dur = min(durations)

        above_avg = sum(1 for d in durations if d > avg_dur)
        below_avg = len(durations) - above_avg

        return {
            'avg': avg_dur,
            'max': max_dur,
            'min': min_dur,
            'above_avg': above_avg,
            'below_avg': below_avg,
            'total_files': total_files
        }

    def plot_duration_distribution(self, sample_size: int = 500):
        durations = []
        sampled_metadata = self.dataset.metadata.sample(min(sample_size, len(self.dataset.metadata)))

        for _, row in tqdm(sampled_metadata.iterrows(), total=len(sampled_metadata), desc="Sampling durations"):
            folder_num = row['Folder Number']
            word_range = self.dataset._find_word_range(folder_num)
            folder_path = os.path.join(self.dataset.dataset_path, word_range, str(folder_num))

            if os.path.exists(folder_path):
                audio_files = [f for f in os.listdir(folder_path)
                             if f.endswith(('.wav', '.mp3', '.flac'))]
                for file in audio_files[:2]:  # Max 2 files per folder
                    try:
                        y, sr = librosa.load(os.path.join(folder_path, file), sr=None)
                        durations.append(len(y)/sr)
                    except:
                        continue

        plt.figure(figsize=(12, 6))
        plt.hist(durations, bins=50, edgecolor='black')
        plt.title("Distribution of Audio Durations")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Number of Voices")
        plt.grid(True)
        plt.show()