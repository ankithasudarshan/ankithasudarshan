# Motion Generation with Motion Priors

This repository contains a corrected implementation of motion generation using transformer encoders with motion priors instead of text priors. The system is designed for SMPL-X motion sequences and uses a VAE architecture.

## Key Corrections Made

### 1. Dataset Issues Fixed
- **Data Loading**: Fixed `.npz` file loading with proper key access
- **Concatenation**: Corrected `np.cat` to `np.concatenate`
- **Reshaping**: Fixed dimension calculations (156 total parameters → 52 joints × 3 features)
- **Sliding Windows**: Implemented proper windowing with padding
- **Motion Priors**: Added logic to generate motion priors from previous frames

### 2. Transformer Issues Fixed
- **Dimension Handling**: Proper batch dimension management
- **Mask Processing**: Robust mask handling for different input formats
- **Prior Integration**: Correct concatenation of motion priors with current sequences
- **VAE Parameters**: Fixed extraction of μ and σ from transformer outputs

### 3. Data Loading Issues Fixed
- **Batch Collation**: Custom collate function for motion data
- **Train/Val/Test Splits**: Proper dataset splitting
- **DataLoader Integration**: Seamless integration with PyTorch DataLoader

## File Structure

```
├── footwork_dataset.py          # Corrected dataset implementation
├── motion_transformer.py        # Transformer encoder for motion
├── motion_dataloader.py         # Data loading utilities
├── train_motion_vae.py         # Complete training script
├── test_motion_setup.py        # Test script to verify setup
└── README_motion_generation.md # This file
```

## Data Format

Your motion data should be stored as `.npz` files with the following structure:

```python
{
    'global_orient': np.array(shape=(T, 3)),      # Global orientation
    'body_pose': np.array(shape=(T, 63)),         # Body pose (21 joints × 3)
    'left_hand_pose': np.array(shape=(T, 45)),    # Left hand (15 joints × 3)
    'right_hand_pose': np.array(shape=(T, 45))    # Right hand (15 joints × 3)
}
```

Total: 156 parameters per frame → reshaped to (52 joints, 3 features)

## Usage

### 1. Test Your Setup

First, verify everything works with the test script:

```bash
python test_motion_setup.py
```

This will create sample data and test all components.

### 2. Prepare Your Data

Organize your `.npz` motion files in a directory:

```
/path/to/your/data/
├── motion_001.npz
├── motion_002.npz
├── motion_003.npz
└── ...
```

### 3. Train the Model

```bash
python train_motion_vae.py \
    --data_dir /path/to/your/data \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --latent_dim 256 \
    --window_size 8 \
    --stride 2
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs
```

## Key Parameters

- **window_size**: Length of motion sequences (default: 8 frames)
- **stride**: Stride for sliding windows (default: 2 frames)
- **latent_dim**: Dimensionality of the latent space (default: 256)
- **beta**: Weight for KL divergence in VAE loss (default: 1.0)

## Architecture Details

### Motion Prior Generation
- Uses previous 4 frames as motion prior context
- Falls back to random frames from same sequence if not enough history
- Provides temporal context for motion generation

### Transformer Encoder
- Concatenates motion priors + current sequence
- Applies positional encoding
- Uses self-attention to model motion dependencies
- Extracts VAE parameters (μ, σ) from prior context

### VAE Training
- Reconstruction loss: MSE between predicted and target frames
- KL divergence: Regularizes latent space
- Target: Last frame of current motion window

## Example Output

```
Data Statistics:
  train_samples: 1250
  val_samples: 156
  test_samples: 156
  window_size: 8
  stride: 2
  njoints: 52
  nfeats: 3

Model created with 2,847,616 parameters

Epoch 1/100
Training: 100%|██████████| 40/40 [00:15<00:00,  2.58it/s, Loss=0.1234, Recon=0.1156, KL=0.0078]
Validation: 100%|██████████| 5/5 [00:01<00:00,  4.12it/s]
Train Loss: 0.1234 (Recon: 0.1156, KL: 0.0078)
Val Loss: 0.1189 (Recon: 0.1123, KL: 0.0066)
New best model saved with val loss: 0.1189
```

## Customization

### Different Motion Representations
To use different motion representations, modify the dataset:

```python
# In footwork_dataset.py, modify _build_poses()
def _build_poses(self):
    # Your custom motion loading logic here
    pass
```

### Custom Transformer Architecture
Modify the transformer parameters:

```python
model = MotionVAE(
    latent_dim=512,          # Larger latent space
    num_layers=12,           # Deeper transformer
    num_heads=16,            # More attention heads
    ff_size=2048,            # Larger feedforward
    dropout=0.1
)
```

### Different Prior Strategies
Modify prior generation in the dataset:

```python
# In footwork_dataset.py, __getitem__()
# Custom prior generation logic
if start_idx >= prior_length:
    # Your custom prior strategy
    pass
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch Errors**
   - Check your `.npz` files have the correct keys
   - Verify motion data shapes match expected format

2. **Memory Issues**
   - Reduce batch_size
   - Reduce window_size or num_workers

3. **Training Instability**
   - Lower learning rate
   - Adjust beta parameter for VAE loss
   - Add gradient clipping

### Debug Mode
Enable debug prints by commenting out the logging line in test script:

```python
# logging.getLogger().setLevel(logging.WARNING)  # Comment this out
```

## Extensions

This codebase can be extended for:

- **Motion Prediction**: Predict future frames given motion priors
- **Motion Completion**: Fill in missing motion segments
- **Style Transfer**: Transfer motion style between different characters
- **Motion Interpolation**: Smooth transitions between motion clips

## Dependencies

```
torch>=1.9.0
numpy>=1.21.0
tqdm>=4.62.0
tensorboard>=2.7.0
```

Install with:
```bash
pip install torch numpy tqdm tensorboard
```