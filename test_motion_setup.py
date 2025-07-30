import torch
import numpy as np
import os
import tempfile
from footwork_dataset import Footwork2Framework, MotionDataset
from motion_transformer import MotionTransformerEncoder
from motion_dataloader import MotionDataModule, collate_motion_batch

def create_sample_data(data_dir, num_files=5, seq_length=60):
    """Create sample .npz files for testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_files):
        # Create random motion data
        T = np.random.randint(seq_length//2, seq_length*2)  # Variable sequence lengths
        
        # SMPL-X pose parameters
        global_orient = np.random.randn(T, 3) * 0.1  # Small rotations
        body_pose = np.random.randn(T, 63) * 0.1     # 21 body joints * 3
        left_hand_pose = np.random.randn(T, 45) * 0.05   # 15 hand joints * 3  
        right_hand_pose = np.random.randn(T, 45) * 0.05  # 15 hand joints * 3
        
        # Save as .npz file
        np.savez(
            os.path.join(data_dir, f'motion_{i:03d}.npz'),
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose
        )
    
    print(f"Created {num_files} sample motion files in {data_dir}")

def test_dataset():
    """Test the dataset loading."""
    print("Testing dataset loading...")
    
    # Create temporary data
    with tempfile.TemporaryDirectory() as temp_dir:
        create_sample_data(temp_dir, num_files=3)
        
        # Test basic dataset
        dataset = Footwork2Framework(
            data_dir=temp_dir,
            window_size=8,
            stride=2
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Current shape: {sample['current'].shape}")
        print(f"Prior shape: {sample['prior'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        
        # Test motion dataset wrapper
        motion_dataset = MotionDataset(
            data_dir=temp_dir,
            window_size=8,
            stride=2,
            split="train"
        )
        
        print(f"Motion dataset length: {len(motion_dataset)}")
        
        # Test batch collation
        batch = [motion_dataset[i] for i in range(min(4, len(motion_dataset)))]
        collated = collate_motion_batch(batch)
        
        print(f"Batch current shape: {collated['current'].shape}")
        print(f"Batch prior shape: {collated['prior'].shape}")
        print(f"Batch mask shape: {collated['mask'].shape}")
        
        print("✓ Dataset test passed!")

def test_transformer():
    """Test the transformer encoder."""
    print("\nTesting transformer encoder...")
    
    # Create dummy batch
    batch_size = 4
    T_current = 8
    T_prior = 4
    njoints = 52
    nfeats = 3
    
    batch = {
        'current': torch.randn(batch_size, T_current, njoints, nfeats),
        'prior': torch.randn(batch_size, T_prior, njoints, nfeats),
        'mask': torch.ones(batch_size, T_current, dtype=bool)
    }
    
    print(f"Input shapes:")
    print(f"  Current: {batch['current'].shape}")
    print(f"  Prior: {batch['prior'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    
    # Create transformer
    transformer = MotionTransformerEncoder(
        latent_dim=128,
        num_layers=2,
        num_heads=4,
        ff_size=256,
        njoints=njoints,
        nfeats=nfeats
    )
    
    # Forward pass
    with torch.no_grad():
        output = transformer(batch)
    
    print(f"Output shapes:")
    print(f"  Mu: {output['mu'].shape}")
    print(f"  Logvar: {output['logvar'].shape}")
    
    print("✓ Transformer test passed!")

def test_data_module():
    """Test the data module."""
    print("\nTesting data module...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_sample_data(temp_dir, num_files=10)
        
        # Create data module
        data_module = MotionDataModule(
            data_dir=temp_dir,
            batch_size=4,
            num_workers=0,  # Avoid multiprocessing issues in test
            window_size=8,
            stride=2
        )
        
        # Get statistics
        stats = data_module.get_data_stats()
        print("Data statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test one batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"Batch shapes:")
            print(f"  Current: {batch['current'].shape}")
            print(f"  Prior: {batch['prior'].shape}")
            print(f"  Mask: {batch['mask'].shape}")
        
        print("✓ Data module test passed!")

def test_end_to_end():
    """Test end-to-end training step."""
    print("\nTesting end-to-end training step...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_sample_data(temp_dir, num_files=10)
        
        # Create data module
        data_module = MotionDataModule(
            data_dir=temp_dir,
            batch_size=4,
            num_workers=0,
            window_size=8,
            stride=2
        )
        
        # Create model components
        transformer = MotionTransformerEncoder(
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            ff_size=128,
            njoints=52,
            nfeats=3
        )
        
        # Simple decoder
        decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 52 * 3)
        )
        
        # Test training step
        train_loader = data_module.train_dataloader()
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            
            # Forward pass through encoder
            encoder_output = transformer(batch)
            mu, logvar = encoder_output['mu'], encoder_output['logvar']
            
            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Decode
            reconstructed = decoder(z)
            reconstructed = reconstructed.view(batch['current'].shape[0], 1, 52, 3)
            
            # Loss computation
            target = batch['current'][:, -1:, :, :]  # Last frame as target
            recon_loss = torch.nn.MSELoss()(reconstructed, target)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + 0.1 * kl_loss
            
            print(f"Loss values:")
            print(f"  Reconstruction: {recon_loss.item():.4f}")
            print(f"  KL divergence: {kl_loss.item():.4f}")
            print(f"  Total: {total_loss.item():.4f}")
            
            # Backward pass
            total_loss.backward()
            
            print("✓ End-to-end test passed!")
        else:
            print("⚠ No training batches available")

if __name__ == "__main__":
    print("Running motion generation setup tests...\n")
    
    # Disable debug prints for cleaner test output
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        test_dataset()
        test_transformer()
        test_data_module()
        test_end_to_end()
        
        print("\n🎉 All tests passed! Your motion generation setup is working correctly.")
        print("\nNext steps:")
        print("1. Prepare your actual motion data in the expected .npz format")
        print("2. Run: python train_motion_vae.py --data_dir /path/to/your/data")
        print("3. Monitor training with tensorboard: tensorboard --logdir ./logs")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()