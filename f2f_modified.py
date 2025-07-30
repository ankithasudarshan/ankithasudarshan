import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class Footwork2Framework(Dataset):
    def __init__(self, data_dir, use_joints=False, seq_len=60, window_size=8, stride=2):
        """
        Dataset for stitched SMPL-X motion sequences (from GVHMR + HAMER) 
        for VAE pretraining.

        Args:
            data_dir (str): Path to directory containing .npz files.
            use_joints (bool): If True, loads 3D joint positions; else loads pose_aa.
            seq_len (int): Length of each motion sequence.
            window_size (int): Size of sliding windows for motion chunks.
            stride (int): Stride for sliding windows.
        """
        self.data_dir = data_dir
        self.use_joints = use_joints
        self.seq_len = seq_len
        self.window_size = window_size
        self.stride = stride

        self.file_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        self._pose, self._num_frames_in_video = self._build_poses()
        self.samples = self._build_samples()

    def _build_poses(self):
        """
        Build list of motion sequences from .npz files.
        """
        poses = []
        num_frames = []
        for path in self.file_paths:
            try:
                raw_data = np.load(path, allow_pickle=True)
                
                # Fixed: Change raw_data.f.body_pose to raw_data['body_pose']
                body_pose = raw_data['body_pose']  # (T, 63)
                left_hand_pose = raw_data['left_hand_pose']  # (T, 45)
                right_hand_pose = raw_data['right_hand_pose']  # (T, 45)
                global_orient = raw_data['global_orient']  # (T, 3)
                
                T = body_pose.shape[0]

                # Fixed: Change np.cat to np.concatenate and fix axis
                comp_data = np.concatenate([
                    global_orient,      # (T, 3)
                    body_pose,          # (T, 63) 
                    left_hand_pose,     # (T, 45)
                    right_hand_pose     # (T, 45)
                ], axis=1)  # Concatenate along feature axis, not time axis
                
                # Fixed: Add the missing reshape line
                rotated_data = comp_data.reshape(T, 52, 3)
                
                poses.append(rotated_data)
                num_frames.append(T)
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")
        return poses, num_frames

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

    def __len__(self):
        return len(self.samples)

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix]  # Already shaped as (len(frame_ix), 52, 3)
        return pose

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