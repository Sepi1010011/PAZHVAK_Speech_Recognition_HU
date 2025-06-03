from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Tuple, Callable
from datetime import datetime

from dataset_load import PazhvakDataLoader
from pazhvak_hu.models.pazhvak_lstm import LSTMLight, LSTMHeavy
from callbacks_training import count_parameters, save_checkpoint, create_tensorboard_logger
from callbacks_training import EarlyStopping

class LSTMTrainer:
    """Handles training and evaluation of LSTM models for Pazhvak speech recognition"""
    
    def __init__(self, 
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 num_classes: int,
                 device: torch.device = None,
                 log_dir: str = "logs"):
        """
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_classes: Number of output classes
            device: Torch device (default: cuda if available else cpu)
            log_dir: Directory for TensorBoard logs
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def train_model(self,
                   model_fn: Callable,
                   model_name: str,
                   input_shape: Tuple[int, ...],
                   epochs: int = 50,
                   bidirectional: bool = False,
                   use_attention: bool = False,
                   batch_size: int = 32) -> Tuple[nn.Module, Dict]:
        """
        Train and evaluate an LSTM model
        
        Args:
            model_fn: Function that returns the model
            model_name: Name for logging/saving
            input_shape: Shape of input features
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Tuple of (trained model, training history)
        """
        # Initialize model and components
        feature_dim = input_shape[-1]
        model = model_fn(feature_dim, self.num_classes, bidirectional=bidirectional, use_attention=use_attention).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        loss_fn = nn.CrossEntropyLoss()
        early_stopper = EarlyStopping(patience=10)
        writer = create_tensorboard_logger(model_name)
                
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        best_val_loss = float('inf')
        
        print(f"\n🔍 Training {model_name} Model")
        print("=" * 50)
        print(f"Feature dimension: {feature_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, optimizer, loss_fn, epoch, epochs, writer
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, loss_fn, epoch, epochs, writer
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, f"{model_name}_best.pth")
                print("💾 Saved new best model")
            
            # Early stopping
            early_stopper(val_loss, model)
            if early_stopper.should_stop:
                print("✅ Early stopping triggered")
                break
        
        # Final evaluation
        test_acc, test_loss = self.evaluate_model(model)
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
        
        print(f"\n🎯 Final Test Results for {model_name}:")
        print(f"Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
        
        writer.close()
        return model, history
    
    def _train_epoch(self, model, optimizer, loss_fn, epoch, total_epochs, writer):
        """Single training epoch"""
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        for batch_x, batch_y in loop:
            batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            loop.set_postfix({
                'loss': loss.item(),
                'acc': correct/total if total > 0 else 0
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total if total > 0 else 0
        
        print(f"📊 Epoch {epoch+1}: Train Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, model, loss_fn, epoch, total_epochs, writer):
        """Single validation epoch"""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
            for batch_x, batch_y in loop:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                loop.set_postfix({
                    'val_loss': loss.item(),
                    'val_acc': correct/total if total > 0 else 0
                })
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = correct / total if total > 0 else 0
        
        print(f"📊 Epoch {epoch+1}: Val Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")
        writer.add_scalar("Loss/val", epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def evaluate_model(self, model) -> Tuple[float, float]:
        """Evaluate model on test set"""
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)
                outputs = model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        return correct/total if total > 0 else 0, test_loss/len(self.test_loader)
    
    def _prepare_batch(self, batch_x, batch_y):
        """Prepare batch by ensuring correct shape and moving to device"""
        # Handle different input shapes
        if batch_x.dim() == 4:  # [batch, channels, seq_len, features]
            batch_x = batch_x.squeeze(1)  # Remove channel dim if present
        
        # Ensure features are last dimension
        feature_dim = self.num_classes if hasattr(self, 'num_classes') else batch_x.size(-1)
        if batch_x.size(-1) != feature_dim:
            batch_x = batch_x.permute(0, 2, 1)  # [batch, seq_len, features]
        
        return batch_x.to(self.device), batch_y.to(self.device)
    

# making model function calling
def build_lstm_light(input_shape, num_classes, bidirectional, use_attention):
    return LSTMLight(input_shape, num_classes, bidirectional, use_attention)

def build_lstm_heavy(input_shape, num_classes, bidirectional, use_attention):
    return LSTMHeavy(input_shape, num_classes, bidirectional, use_attention)


if __name__ == "__main__":
    data_loader_factory = PazhvakDataLoader(
        dataset_path="/path/to/your/dataset",
        batch_size=64,
        num_workers=4
    )
    
    train_loader, val_loader, test_loader = data_loader_factory.get_data_loaders(feature_type="mfcc")
    
    # Get label information
    label_encoder = data_loader_factory.get_label_encoder()
    num_classes = data_loader_factory.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    print(f"Sample batch - MFCC: {next(iter(train_loader))}")
    
    
    trainer = LSTMTrainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_classes=num_classes
    )
    
    # Train models
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]

    models = {
        "lstm_light": build_lstm_light,
        "lstm_heavy": build_lstm_heavy
    }

    results = {}
    for name, model_fn in models.items():
        print(f"\n🔧 Training model: {name}")
        model, history = trainer.train_model(
            model_fn=model_fn,
            model_name=name,
            input_shape=input_shape,
            epochs=40
        )
        results[name] = model

