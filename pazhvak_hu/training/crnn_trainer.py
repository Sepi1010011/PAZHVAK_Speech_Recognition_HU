from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pathlib import Path
from typing import Dict, Tuple, Callable
from datetime import datetime

from dataset_load import PazhvakDataLoader
from pazhvak_hu.models.pazhvak_crnn import CRNN 
from callbacks_training import count_parameters, save_checkpoint, create_tensorboard_logger
from callbacks_training import EarlyStopping


class CRNNTrainer:
    """Handles training and evaluation of CRNN models for Pazhvak speech recognition"""
    
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
                   model_name: str,
                   input_shape: Tuple[int, ...],
                   bidirectional: bool = False,
                   use_attention: bool = False,
                   epochs: int = 50,
                   batch_size: int = 32) -> Tuple[nn.Module, Dict]:
        """
        Train and evaluate a CRNN model
        
        Args:
            model_name: Name for logging/saving
            input_shape: Shape of input features (channels, height, width)
            bidirectional: Whether to use bidirectional RNN
            use_attention: Whether to use attention mechanism
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Tuple of (trained model, training history)
        """
        model = CRNN(input_shape, self.num_classes, bidirectional, use_attention).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        print(f"Input shape: {input_shape}")
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
                save_checkpoint(model, optimizer, epoch, f"{model_name}_best.pth")
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
        # Ensure proper shape: [batch, channels, height, width]
        if batch_x.dim() == 3:  # Missing channel dim
            batch_x = batch_x.unsqueeze(1)
        
        return batch_x.to(self.device), batch_y.to(self.device)
    


if __name__ == "__main__":
    data_loader_factory = PazhvakDataLoader(
        dataset_path="/path/to/your/dataset",
        batch_size=64,
        num_workers=4
    )
    
    train_loader, val_loader, test_loader = data_loader_factory.get_data_loaders(feature_type="mel")  # Typically mel for CRNN
    
    # Get label information
    label_encoder = data_loader_factory.get_label_encoder()
    num_classes = data_loader_factory.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    print(f"Sample batch shape: {next(iter(train_loader))[0].shape}")
    
    trainer = CRNNTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes
    )
    
    # Train model
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]  # Should be (channels, height, width)

    print(f"\n🔧 Training model: CRNN")
    model, history = trainer.train_model(
        model_name="CRNN",
        input_shape=input_shape,
        epochs=40
    )