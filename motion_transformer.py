import torch
import torch.nn as nn
import numpy as np

class MotionTransformerEncoder(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=8, 
                 dropout=0.1, ablation=None, njoints=52, nfeats=3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.njoints = njoints
        self.nfeats = nfeats
        
        # Input dimension: njoints * nfeats (52 * 3 = 156 for rotation vectors)
        input_dim = njoints * nfeats
        
        # Skeleton embedding: maps flattened motion to latent space
        self.skelEmbedding = nn.Linear(input_dim, latent_dim)
        
        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (seq_len, batch, features)
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers for VAE
        if ablation == "average_encoder":
            self.mu_layer = nn.Linear(latent_dim, latent_dim)
            self.sigma_layer = nn.Linear(latent_dim, latent_dim)
        else:
            # For motion priors, we extract mu and logvar from specific positions
            # We'll use learned projection layers
            self.mu_layer = nn.Linear(latent_dim, latent_dim)
            self.logvar_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, batch):
        current, prior, mask = batch["current"], batch["prior"], batch["mask"]
        
        # Debug prints
        print(f"[DEBUG] Input shapes - current: {current.shape}, prior: {prior.shape}, mask: {mask.shape}")
        
        # Handle batch dimensions
        if current.dim() == 3:
            current = current.unsqueeze(0)  # (1, T, njoints, nfeats)
        if prior.dim() == 3:
            prior = prior.unsqueeze(0)      # (1, T_prior, njoints, nfeats)
        
        # Extract dimensions
        bs, T_current, njoints_current, nfeats_current = current.shape
        bs_prior, T_prior, njoints_prior, nfeats_prior = prior.shape
        
        # Validate dimensions
        assert njoints_current == njoints_prior == self.njoints, \
            f"Joint mismatch: current={njoints_current}, prior={njoints_prior}, expected={self.njoints}"
        assert nfeats_current == nfeats_prior == self.nfeats, \
            f"Feature mismatch: current={nfeats_current}, prior={nfeats_prior}, expected={self.nfeats}"
        
        # Handle batch size mismatch
        if bs != bs_prior:
            if bs_prior == 1:
                prior = prior.expand(bs, T_prior, njoints_prior, nfeats_prior)
            elif bs == 1:
                current = current.expand(bs_prior, T_current, njoints_current, nfeats_current)
                bs = bs_prior
            else:
                raise ValueError(f"Incompatible batch sizes: current={bs}, prior={bs_prior}")
        
        print(f"[DEBUG] After dimension handling - bs={bs}, T_current={T_current}, T_prior={T_prior}")
        
        # Flatten motion data: (bs, T, njoints * nfeats)
        x = current.reshape(bs, T_current, self.njoints * self.nfeats)
        p = prior.reshape(bs, T_prior, self.njoints * self.nfeats)
        
        print(f"[DEBUG] After flatten - x: {x.shape}, p: {p.shape}")
        
        # Apply skeleton embedding
        x = self.skelEmbedding(x)          # (bs, T_current, latent_dim)
        prior_emb = self.skelEmbedding(p)  # (bs, T_prior, latent_dim)
        
        print(f"[DEBUG] After embedding - x: {x.shape}, prior_emb: {prior_emb.shape}")
        
        # Handle mask
        mask = self._process_mask(mask, bs, T_current)
        
        if self.ablation == "average_encoder":
            # Simple averaging approach
            x = self.sequence_pos_encoder(x)
            final = self.seqTransEncoder(x.transpose(0, 1), src_key_padding_mask=~mask)
            z = final.mean(dim=0)  # Average over sequence dimension
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
            
        else:
            # Concatenate prior + current along sequence dimension
            xseq = torch.cat([prior_emb, x], dim=1)  # (bs, T_prior + T_current, latent_dim)
            
            # Create combined mask
            prior_mask = torch.ones((bs, T_prior), dtype=bool, device=mask.device)
            maskseq = torch.cat([prior_mask, mask], dim=1)
            
            print(f"[DEBUG] Combined sequence - xseq: {xseq.shape}, maskseq: {maskseq.shape}")
            
            # Add positional encoding
            xseq = self.sequence_pos_encoder(xseq)
            
            # Transformer expects (seq_len, batch, features)
            xseq = xseq.transpose(0, 1)  # (T_prior + T_current, bs, latent_dim)
            
            # Apply transformer
            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            
            print(f"[DEBUG] Transformer output - final: {final.shape}")
            
            # Extract features from prior context for VAE parameters
            # Use the last frame of prior context and first frame of current
            if T_prior >= 1:
                # Use last prior frame for mu
                mu_features = final[T_prior - 1]  # Last prior frame
                mu = self.mu_layer(mu_features)
                
                if T_prior >= 2:
                    # Use second-to-last prior frame for logvar
                    logvar_features = final[T_prior - 2]
                else:
                    # Use first current frame for logvar
                    logvar_features = final[T_prior]
                logvar = self.logvar_layer(logvar_features)
            else:
                raise ValueError(f"Prior sequence length (T_prior={T_prior}) must be at least 1")
        
        print(f"[DEBUG] Final outputs - mu: {mu.shape}, logvar: {logvar.shape}")
        
        return {"mu": mu, "logvar": logvar}
    
    def _process_mask(self, mask, bs, T_current):
        """Process and validate the mask tensor."""
        print(f"[DEBUG] Processing mask - original shape: {mask.shape}")
        
        # Handle different mask dimensions
        if mask.dim() == 1:
            if len(mask) == T_current:
                mask = mask.unsqueeze(0).expand(bs, -1)
            else:
                # Create default mask
                mask = torch.ones((bs, T_current), dtype=bool, device=mask.device)
                
        elif mask.dim() == 2:
            mask_bs, mask_seq = mask.shape
            
            if mask_bs == T_current and mask_seq == bs:
                # Transpose case
                mask = mask.transpose(0, 1)
            elif mask_bs == bs and mask_seq != T_current:
                # Adjust sequence length
                if mask_seq < T_current:
                    # Pad with True
                    pad = torch.ones((bs, T_current - mask_seq), dtype=bool, device=mask.device)
                    mask = torch.cat([mask, pad], dim=1)
                else:
                    # Truncate
                    mask = mask[:, :T_current]
            elif mask_bs != bs:
                # Create new mask
                mask = torch.ones((bs, T_current), dtype=bool, device=mask.device)
                
        else:
            # Multi-dimensional mask - reduce to 2D
            while mask.dim() > 2:
                mask = mask.any(dim=-1)
            mask = self._process_mask(mask, bs, T_current)
        
        # Final validation
        if mask.shape != (bs, T_current):
            print(f"[WARNING] Mask shape {mask.shape} doesn't match expected ({bs}, {T_current})")
            mask = torch.ones((bs, T_current), dtype=bool, device=mask.device)
        
        print(f"[DEBUG] Final mask shape: {mask.shape}")
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)