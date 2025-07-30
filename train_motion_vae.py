import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm

from footwork_dataset import MotionDataset
from motion_transformer import MotionTransformerEncoder
from motion_dataloader import MotionDataModule

class MotionVAE(nn.Module):
    """
    Complete Motion VAE with transformer encoder and decoder.
    """
    def __init__(self, latent_dim=256, njoints=52, nfeats=3, **transformer_kwargs):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        
        # Encoder (transformer)
        self.encoder = MotionTransformerEncoder(
            latent_dim=latent_dim,
            njoints=njoints,
            nfeats=nfeats,
            **transformer_kwargs
        )
        
        # Decoder (simple MLP for now - can be replaced with transformer decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, njoints * nfeats)
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, batch):
        # Encode
        encoder_output = self.encoder(batch)
        mu, logvar = encoder_output["mu"], encoder_output["logvar"]
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        batch_size = z.shape[0]
        reconstructed = self.decoder(z)  # (batch_size, njoints * nfeats)
        reconstructed = reconstructed.view(batch_size, 1, self.njoints, self.nfeats)
        
        return {
            "reconstructed": reconstructed,
            "mu": mu,
            "logvar": logvar,
            "z": z
        }

def vae_loss(reconstructed, target, mu, logvar, beta=1.0):
    """
    VAE loss combining reconstruction loss and KL divergence.
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(reconstructed, target)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.numel()  # Normalize by number of elements
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss
    }

def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        
        # Use the last frame of current sequence as reconstruction target
        target = batch["current"][:, -1:, :, :]  # (batch_size, 1, njoints, nfeats)
        
        # Compute loss
        loss_dict = vae_loss(
            output["reconstructed"], 
            target, 
            output["mu"], 
            output["logvar"], 
            beta=beta
        )
        
        # Backward pass
        loss_dict["total_loss"].backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict["total_loss"].item()
        total_recon_loss += loss_dict["recon_loss"].item()
        total_kl_loss += loss_dict["kl_loss"].item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss_dict["total_loss"].item():.4f}',
            'Recon': f'{loss_dict["recon_loss"].item():.4f}',
            'KL': f'{loss_dict["kl_loss"].item():.4f}'
        })
    
    return {
        "total_loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader)
    }

def validate_epoch(model, dataloader, device, beta=1.0):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            output = model(batch)
            
            # Use the last frame of current sequence as reconstruction target
            target = batch["current"][:, -1:, :, :]
            
            # Compute loss
            loss_dict = vae_loss(
                output["reconstructed"], 
                target, 
                output["mu"], 
                output["logvar"], 
                beta=beta
            )
            
            total_loss += loss_dict["total_loss"].item()
            total_recon_loss += loss_dict["recon_loss"].item()
            total_kl_loss += loss_dict["kl_loss"].item()
    
    return {
        "total_loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader)
    }

def main():
    parser = argparse.ArgumentParser(description='Train Motion VAE')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to motion data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for VAE loss')
    parser.add_argument('--window_size', type=int, default=8, help='Motion window size')
    parser.add_argument('--stride', type=int, default=2, help='Sliding window stride')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data module
    data_module = MotionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Print data statistics
    stats = data_module.get_data_stats()
    print("Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    model = MotionVAE(
        latent_dim=args.latent_dim,
        njoints=52,
        nfeats=3,
        num_layers=6,
        num_heads=8,
        ff_size=1024,
        dropout=0.1
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.beta)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, args.beta)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['total_loss'], epoch)
        writer.add_scalar('Recon_Loss/Train', train_metrics['recon_loss'], epoch)
        writer.add_scalar('Recon_Loss/Val', val_metrics['recon_loss'], epoch)
        writer.add_scalar('KL_Loss/Train', train_metrics['kl_loss'], epoch)
        writer.add_scalar('KL_Loss/Val', val_metrics['kl_loss'], epoch)
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f})")
        print(f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f})")
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()