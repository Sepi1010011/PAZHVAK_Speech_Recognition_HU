# For Data Processing

import os
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter



# Callbacks (TensorBoard, EarlyStopping)

def create_tensorboard_logger(model_name):
    log_dir = os.path.join("logs", f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)


def save_checkpoint(model, optimizer, epoch, path="model_best.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
