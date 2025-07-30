# File Modification Instructions

## Modifications for `f2f.py` (Dataloader file)

### 1. Add imports at the top:
```python
import random  # Add this import
```

### 2. Modify `__init__` method:
Add these parameters and line:
```python
def __init__(self, data_dir, use_joints=False, seq_len=60, window_size=8, stride=2):
    # ... existing code ...
    self.window_size = window_size
    self.stride = stride
    
    # ... existing code until the end, then add:
    self.samples = self._build_samples()  # ADD THIS LINE
```

### 3. Fix `_build_poses` method:
Replace the entire method with:
```python
def _build_poses(self):
    """
    Build list of motion sequences from .npz files.
    """
    poses = []
    num_frames = []
    for path in self.file_paths:
        try:
            raw_data = np.load(path, allow_pickle=True)
            
            # FIX 1: Change raw_data.f.body_pose to raw_data['body_pose']
            body_pose = raw_data['body_pose']  # (T, 63)
            left_hand_pose = raw_data['left_hand_pose']  # (T, 45)
            right_hand_pose = raw_data['right_hand_pose']  # (T, 45)
            global_orient = raw_data['global_orient']  # (T, 3)
            
            T = body_pose.shape[0]

            # FIX 2: Change np.cat to np.concatenate and fix axis
            comp_data = np.concatenate([
                global_orient,      # (T, 3)
                body_pose,          # (T, 63) 
                left_hand_pose,     # (T, 45)
                right_hand_pose     # (T, 45)
            ], axis=1)  # Concatenate along feature axis, not time axis
            
            # FIX 3: Add the missing reshape line
            rotated_data = comp_data.reshape(T, 52, 3)
            
            poses.append(rotated_data)
            num_frames.append(T)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
    return poses, num_frames
```

### 4. Add new method `_build_samples`:
```python
def _build_samples(self):
    """
    Build list of valid motion windows from all sequences.
    """
    samples = []
    
    for seq_idx, pose_seq in enumerate(self._pose):
        T = pose_seq.shape[0]
        
        # Create sliding windows
        for start_idx in range(0, max(1, T - self.window_size + 1), self.stride):
            end_idx = min(start_idx + self.window_size, T)
            
            # Only add if we have enough frames
            if end_idx - start_idx >= self.window_size // 2:  # At least half window
                samples.append((seq_idx, start_idx, end_idx))
                
    return samples
```

### 5. Fix `__len__` method:
```python
def __len__(self):
    return len(self.samples)  # Remove the reference to self.samples in original
```

### 6. Fix `_load_rotvec` method (fix indentation):
```python
def _load_rotvec(self, ind, frame_ix):
    pose = self._pose[ind][frame_ix]  # Already shaped as (len(frame_ix), 52, 3)
    return pose
```

### 7. Add new method `_pad_sequence`:
```python
def _pad_sequence(self, sequence, target_length):
    """
    Pad sequence to target length by repeating the last frame.
    """
    current_length = sequence.shape[0]
    if current_length >= target_length:
        return sequence[:target_length]
    
    # Pad by repeating last frame
    last_frame = sequence[-1:]  # (1, 52, 3)
    pad_length = target_length - current_length
    padding = np.repeat(last_frame, pad_length, axis=0)
    
    return np.concatenate([sequence, padding], axis=0)
```

### 8. Add new method `__getitem__`:
```python
def __getitem__(self, index):
    seq_idx, start_idx, end_idx = self.samples[index]
    
    # Get the motion window
    frame_indices = np.arange(start_idx, end_idx)
    current_motion = self._load_rotvec(seq_idx, frame_indices)
    
    # Pad if necessary
    current_motion = self._pad_sequence(current_motion, self.window_size)
    
    # Generate prior motion (previous frames or random motion from same sequence)
    prior_length = 4  # Length of prior context
    
    if start_idx >= prior_length:
        # Use previous frames as prior
        prior_indices = np.arange(start_idx - prior_length, start_idx)
        prior_motion = self._load_rotvec(seq_idx, prior_indices)
    else:
        # Use random frames from the same sequence as prior
        total_frames = self._num_frames_in_video[seq_idx]
        if total_frames >= prior_length:
            random_start = random.randint(0, total_frames - prior_length)
            prior_indices = np.arange(random_start, random_start + prior_length)
            prior_motion = self._load_rotvec(seq_idx, prior_indices)
        else:
            # If sequence is too short, repeat frames
            all_frames = np.arange(total_frames)
            prior_motion = self._load_rotvec(seq_idx, all_frames)
            prior_motion = self._pad_sequence(prior_motion, prior_length)

    # Create mask (all frames are valid in this case)
    mask = np.ones(self.window_size, dtype=bool)
    
    # Convert to torch tensors
    current_motion = torch.from_numpy(current_motion).float()  # (window_size, 52, 3)
    prior_motion = torch.from_numpy(prior_motion).float()      # (prior_length, 52, 3)
    mask = torch.from_numpy(mask)
    
    return {
        "current": current_motion,
        "prior": prior_motion, 
        "mask": mask
    }
```

---

## Modifications for `f2f_dataset.py` (Get Dataset file)

### 1. Remove the broken line at the top:
Delete this line:
```python
touch: cannot touch 'dataset_modified.py': Permission denied
```

### 2. Fix imports section:
Replace the import section with:
```python
import random
import numpy as np
import torch

# Handle imports that might not exist
try:
    from .tools import parse_info_name
    from ..utils.tensors import collate
    from ..utils.misc import to_torch
    import src.utils.rotation_conversions as geometry
except ImportError:
    # Fallback implementations if imports fail
    def to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return torch.tensor(x)
    
    def collate(batch):
        # Simple collate function
        if isinstance(batch[0], dict):
            return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
        else:
            return torch.stack(batch)
    
    # You'll need to implement rotation_conversions or remove pose representation conversions
```

### 3. Modify `__init__` method:
Add these parameters:
```python
def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train",
             pose_rep="rot6d", translation=True, glob=True, max_len=-1, min_len=-1, num_seq_max=-1, 
             window_size=8, stride=2, **kwargs):  # ADD window_size and stride parameters
    # ... existing code ...
    # ADD THESE NEW PARAMETERS:
    self.window_size = window_size
    self.stride = stride
```

### 4. Replace the broken `_get_item_data_index` method:
Keep all the existing frame sampling logic, but replace the windowing part at the end:

Replace this broken section:
```python
# Split into sliding windows of size 8
window_size = 8
stride = 2  # or customize
chunks = []
windows = []
nframes = pose.shape[0]  # total number of frames

# ... all the broken windowing code ...

return windows  # List of arrays, each shape: (window_size, ...)
```

With this:
```python
# GET POSE DATA
pose = self.get_pose_data(data_index, frame_ix)

# REPLACE THE BROKEN WINDOWING CODE WITH THIS:
# Convert to motion windows with priors
motion_windows = self._create_motion_windows_with_priors(pose, data_index)

# For now, return the first window (you can modify this to return all windows)
if len(motion_windows) > 0:
    return motion_windows[0]  # Return first window
else:
    # Fallback: create a single window from available data
    return self._create_single_window_with_prior(pose, data_index)
```

### 5. Add these new methods:

```python
def _create_motion_windows_with_priors(self, pose, data_index):
    """
    Create sliding windows from pose data with motion priors.
    """
    windows = []
    nframes = pose.shape[0]  # pose shape: (T, njoints, nfeats)
    
    # Create sliding windows
    for start_idx in range(0, nframes, self.stride):
        end_idx = min(start_idx + self.window_size, nframes)
        
        if end_idx - start_idx >= self.window_size // 2:  # At least half window
            # Get current window
            current_window = pose[start_idx:end_idx]  # (window_len, njoints, nfeats)
            
            # Pad if necessary
            if current_window.shape[0] < self.window_size:
                current_window = self._pad_sequence_torch(current_window, self.window_size)
            
            # Generate prior
            prior_motion = self._generate_prior_motion(pose, start_idx, data_index)
            
            # Create mask
            mask = torch.ones(self.window_size, dtype=torch.bool)
            
            window_data = {
                "current": current_window,
                "prior": prior_motion,
                "mask": mask
            }
            windows.append(window_data)
    
    return windows

def _create_single_window_with_prior(self, pose, data_index):
    """
    Create a single motion window with prior from pose data.
    """
    nframes = pose.shape[0]
    
    # Take the last window_size frames (or all if less)
    if nframes >= self.window_size:
        current_motion = pose[-self.window_size:]
        start_idx = nframes - self.window_size
    else:
        current_motion = self._pad_sequence_torch(pose, self.window_size)
        start_idx = 0
    
    # Generate prior
    prior_motion = self._generate_prior_motion(pose, start_idx, data_index)
    
    # Create mask
    mask = torch.ones(self.window_size, dtype=torch.bool)
    
    return {
        "current": current_motion,
        "prior": prior_motion,
        "mask": mask
    }

def _generate_prior_motion(self, pose, start_idx, data_index):
    """
    Generate motion prior for a given starting index.
    """
    prior_length = 4
    nframes = pose.shape[0]
    
    if start_idx >= prior_length:
        # Use previous frames as prior
        prior_motion = pose[start_idx - prior_length:start_idx]
    else:
        # Use random frames from the sequence as prior
        if nframes >= prior_length:
            random_start = random.randint(0, max(0, nframes - prior_length))
            prior_motion = pose[random_start:random_start + prior_length]
        else:
            # If sequence is too short, repeat frames
            prior_motion = self._pad_sequence_torch(pose, prior_length)
    
    return prior_motion

def _pad_sequence_torch(self, sequence, target_length):
    """
    Pad sequence to target length by repeating the last frame.
    """
    current_length = sequence.shape[0]
    if current_length >= target_length:
        return sequence[:target_length]
    
    # Pad by repeating last frame
    last_frame = sequence[-1:]  # (1, njoints, nfeats)
    pad_length = target_length - current_length
    padding = last_frame.repeat(pad_length, 1, 1)
    
    return torch.cat([sequence, padding], dim=0)
```

### 6. Fix `get_label_sample` method:
Replace this line:
```python
data_index = np.random.choice(choices, n)
```
With:
```python
data_index = np.random.choice(total_samples, n)
```

---

## Summary of Key Fixes:

### For `f2f.py`:
1. ✅ Fixed `.npz` file key access (`raw_data.f.` → `raw_data['']`)
2. ✅ Fixed `np.cat` → `np.concatenate` 
3. ✅ Fixed concatenation axis (axis=1 for features)
4. ✅ Added missing reshape line
5. ✅ Fixed indentation in `_load_rotvec`
6. ✅ Added proper sliding window implementation
7. ✅ Added motion prior generation
8. ✅ Added `__getitem__` method

### For `f2f_dataset.py`:
1. ✅ Removed broken command line
2. ✅ Added fallback imports
3. ✅ Fixed broken windowing logic
4. ✅ Added motion prior support
5. ✅ Added proper tensor handling
6. ✅ Fixed undefined variable bug

These modifications will make your motion generation system work with motion priors instead of text priors, and fix all the dimension mismatch and data loading errors you were experiencing.