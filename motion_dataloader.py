import torch
from torch.utils.data import DataLoader
import numpy as np

def collate_motion_batch(batch):
    """
    Custom collate function for motion data batches.
    
    Args:
        batch: List of dictionaries with keys ["current", "prior", "mask"]
    
    Returns:
        Dictionary with batched tensors
    """
    # Extract individual components
    current_list = [item["current"] for item in batch]
    prior_list = [item["prior"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    
    # Stack into batch tensors
    current_batch = torch.stack(current_list, dim=0)  # (batch_size, T_current, njoints, nfeats)
    prior_batch = torch.stack(prior_list, dim=0)      # (batch_size, T_prior, njoints, nfeats)
    mask_batch = torch.stack(mask_list, dim=0)        # (batch_size, T_current)
    
    return {
        "current": current_batch,
        "prior": prior_batch,
        "mask": mask_batch
    }

def create_motion_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for motion data with proper collation.
    
    Args:
        dataset: Motion dataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_motion_batch,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )

class MotionDataModule:
    """
    Data module that handles train/val/test splits and dataloaders.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4, 
                 window_size=8, stride=2, seq_len=60):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.stride = stride
        self.seq_len = seq_len
        
        # Import here to avoid circular imports
        from footwork_dataset import MotionDataset
        
        # Create datasets
        self.train_dataset = MotionDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            window_size=window_size,
            stride=stride,
            split="train"
        )
        
        self.val_dataset = MotionDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            window_size=window_size,
            stride=stride,
            split="val"
        )
        
        self.test_dataset = MotionDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            window_size=window_size,
            stride=stride,
            split="test"
        )
    
    def train_dataloader(self):
        return create_motion_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return create_motion_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return create_motion_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_data_stats(self):
        """Get statistics about the dataset."""
        return {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "window_size": self.window_size,
            "stride": self.stride,
            "njoints": 52,
            "nfeats": 3
        }