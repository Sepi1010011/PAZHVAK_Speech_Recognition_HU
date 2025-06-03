import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from pazhvak_hu.dataset.pazhvak_dataset import PazhvakDataset

from dataset_tensor_loading import PazhvakTorchDataset, collate_fn_mfcc, collate_fn_mel


class PazhvakDataLoader:
    """
    class for creating PyTorch DataLoaders for Pazhvak dataset with different feature types.
    
    Args:
        dataset_path (str): Path to the dataset directory
        metadata_file (str): Name of the metadata file (default: "P1993818924117826_1247741184149835.xlsx")
        batch_size (int): Batch size for DataLoader (default: 32)
        num_workers (int): Number of workers for DataLoader (default: 2)
    """
    
    def __init__(self, dataset_path: str, metadata_file: str = "P1993818924117826_1247741184149835.xlsx",
                 batch_size: int = 32, num_workers: int = 2):
        self.dataset_path = dataset_path
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_encoder = None
        self.num_classes = None
        
        # Initialize dataset and splits
        self.dataset = PazhvakDataset(dataset_path=dataset_path, metadata_file=metadata_file)
        self._prepare_datasets()
    
    def _prepare_datasets(self):
        """Prepare the dataset splits and label encoder"""
        self.dataset.split_dataset(test_size=0.2, val_size=0.1, random_state=42)
        
        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataset.train_df['label'])
        self.num_classes = len(self.label_encoder.classes_)
    
    def get_data_loaders(self, feature_type: str = "mfcc", shuffle_train: bool = True) -> tuple:
        """
        Get DataLoaders for train, validation, and test sets.
        
        Args:
            feature_type (str): Type of features to use ("mfcc", "mel", etc.)
            shuffle_train (bool): Whether to shuffle training data (default: True)
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Get the appropriate collate function
        collate_fn = self._get_collate_fn(feature_type)
        
        # Create datasets
        train_dataset = PazhvakTorchDataset(
            self.dataset, 
            split="train", 
            label_encoder=self.label_encoder
        )
        
        val_dataset = PazhvakTorchDataset(
            self.dataset, 
            split="val", 
            label_encoder=self.label_encoder
        )
        
        test_dataset = PazhvakTorchDataset(
            self.dataset, 
            split="test", 
            label_encoder=self.label_encoder
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_collate_fn(self, feature_type: str):
        """Get the appropriate collate function based on feature type"""
        if feature_type.lower() == "mfcc":
            return collate_fn_mfcc
        elif feature_type.lower() == "mel":
            return collate_fn_mel
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}. Available options: 'mfcc', 'mel'")
    
    def get_label_encoder(self) -> LabelEncoder:
        """Get the fitted label encoder"""
        return self.label_encoder
    
    def get_num_classes(self) -> int:
        """Get the number of classes"""
        return self.num_classes


# Example usage
if __name__ == "__main__":
    # Initialize with dataset path
    data_loader_factory = PazhvakDataLoader(
        dataset_path="/path/to/your/dataset",
        batch_size=64,
        num_workers=4
    )
    
    # Get MFCC data loaders
    mfcc_train, mfcc_val, mfcc_test = data_loader_factory.get_data_loaders(feature_type="mfcc")
    
    # Get Mel data loaders
    mel_train, mel_val, mel_test = data_loader_factory.get_data_loaders(feature_type="mel")
    
    # Get label information
    label_encoder = data_loader_factory.get_label_encoder()
    num_classes = data_loader_factory.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    print(f"Sample batch - MFCC: {next(iter(mfcc_train))}")
    print(f"Sample batch - Mel: {next(iter(mel_train))}")