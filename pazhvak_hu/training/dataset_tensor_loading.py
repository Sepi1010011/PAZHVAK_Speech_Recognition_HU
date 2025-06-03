# For Data Processing
import numpy as np


# For DL, Classifier,... Model (Sklearn, pytorch, keras, Tensorflow,...)
import torch
from torch.utils.data import Dataset


def collate_fn_mfcc(batch):
    """
    Collate function for MFCC features with variable time lengths.
    Pads each sample to the maximum time length in the batch.

    Output: [batch_size, max_time, n_mfcc]
    """
    features, labels = zip(*batch)

    # Ensure all features are torch tensors
    features = [torch.tensor(f, dtype=torch.float32) if not isinstance(f, torch.Tensor) else f for f in features]

    # Determine max time length (T)
    max_len = max(f.shape[0] for f in features)

    padded_features = []
    for f in features:
        pad_len = max_len - f.shape[0]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, 0, 0, pad_len))  # Pad time dim
        padded_features.append(f)

    features_tensor = torch.stack(padded_features, dim=0)  # [B, T, n_mfcc]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_tensor, labels_tensor


def collate_fn_mel(batch):
    """
    Collate function for Mel spectrograms with variable time steps.
    Assumes input features are shaped (1, n_mels, time_steps).
    Pads the time dimension to the length of the longest sample in the batch.
    """
    features, labels = zip(*batch)

    # Determine maximum time length
    max_len = max(feat.shape[-1] for feat in features)

    padded_features = []
    for feat in features:
        # Ensure shape is (1, n_mels, time_steps)
        if feat.ndim == 2:
            feat = feat[np.newaxis, :, :]  # (n_mels, time) -> (1, n_mels, time)
        elif feat.shape[0] != 1:
            feat = feat.transpose(2, 0, 1)  # fix if needed

        pad_len = max_len - feat.shape[-1]
        if pad_len > 0:
            # Pad last dim (time_steps)
            feat = torch.nn.functional.pad(feat, (0, pad_len), mode='constant', value=0)
        else:
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            else:
                feat = feat.clone().detach()

        padded_features.append(feat)

    features_tensor = torch.stack(padded_features)  # (B, 1, n_mels, max_time)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return features_tensor, labels_tensor


def collate_fn_crnn(batch):
    """
    Pads variable-length MFCCs and returns tensors in [B, 1, T, n_mfcc] shape for CRNN.
    """
    features, labels = zip(*batch)

    # Ensure tensors
    features = [torch.tensor(f, dtype=torch.float32) if not isinstance(f, torch.Tensor) else f for f in features]

    # Pad time dimension to max length
    max_len = max(f.shape[0] for f in features)
    padded = []
    for f in features:
        pad_len = max_len - f.shape[0]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, 0, 0, pad_len))  # pad time dim
        padded.append(f)

    features_tensor = torch.stack(padded, dim=0)  # [B, T, n_mfcc]
    features_tensor = features_tensor.unsqueeze(1)  # [B, 1, T, n_mfcc]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_tensor, labels_tensor


class PazhvakTorchDataset(Dataset):
    def __init__(self, dataset, split="train", transform=None, label_encoder=None):
        """
        PyTorch Dataset class for loading pre-split voice data

        Args:
            dataset (PazhvakDataset): The main dataset object with splits
            split (str): One of "train", "val", or "test"
            transform: Optional transforms to be applied
            label_encoder: Optional label encoder for converting text labels to numbers
        """
        assert split in ["train", "val", "test"], "Invalid split."
        assert hasattr(dataset, f"{split}_df"), "Dataset splits not initialized. Call split_dataset() first."

        # Get the appropriate split DataFrame
        split_df = getattr(dataset, f"{split}_df")

        self.samples = split_df['file_path'].tolist()
        self.labels = split_df['label'].tolist()
        self.transform = transform
        self.label_encoder = label_encoder

        if self.label_encoder is not None:
            self.encoded_labels = self.label_encoder.transform(self.labels)
        else:
            self.encoded_labels = self.labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the numpy feature file
        feature = np.load(self.samples[idx])
        label = self.encoded_labels[idx]

        # Handle different feature shapes
        # if len(feature.shape) == 2:  # If 2D (e.g., mel spectrogram)
        #     feature = np.expand_dims(feature, axis=0)  # Add channel dimension
        # elif feature.shape[0] > 3:  # Handle possible channel-first misalignment
        #     feature = feature.transpose(1, 2, 0)[:, :, :1]  # Convert to channel-last and take first channel

        # Normalize and ensure float32
        feature = (feature - feature.mean()) / (feature.std() + 1e-8)
        feature = feature.astype(np.float32)

        # Apply transforms if specified
        if self.transform:
            feature = self.transform(feature)

        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(feature)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor

    def get_label_names(self):
        """Get the original text labels (unencoded)"""
        return self.labels
    

