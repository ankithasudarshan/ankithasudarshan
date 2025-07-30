# REMOVE THIS LINE: touch: cannot touch 'dataset_modified.py': Permission denied
import random

import numpy as np
import torch

# COMMENT OUT OR REPLACE these imports if they don't exist in your project:
# from .tools import parse_info_name
# from ..utils.tensors import collate
# from ..utils.misc import to_torch
# import src.utils.rotation_conversions as geometry

# REPLACE WITH THESE IMPORTS:
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

POSE_REPS = ["xyz", "rotvec", "rotmat", "rotquat", "rot6d"]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train",
                 pose_rep="rot6d", translation=True, glob=True, max_len=-1, min_len=-1, num_seq_max=-1, 
                 window_size=8, stride=2, **kwargs):  # ADD window_size and stride parameters
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max
        # ADD THESE NEW PARAMETERS:
        self.window_size = window_size
        self.stride = stride

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()

        # to remove shuffling
        self._original_train = None
        self._original_test = None

        # These should be implemented by subclasses:
        # self._actions
        # self._train/self._test  
        # self._num_frames_in_video[data_index]
        # self._action_to_label[action]
        # self._label_to_action[label]
        # self._load_pose(data_index, frame_ix) -> should call self._load_rotvec
        # self._actions[ind]
        # self._action_classes[action]

    # Keep existing methods unchanged...
    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_pose_data(self, data_index, frame_ix):
        pose = self._load(data_index, frame_ix)
        return pose

    def get_label(self, ind):
        action = self.get_action(ind)
        return self.action_to_label(action)

    def parse_action(self, path, return_int=True):
        info = parse_info_name(path)["A"]
        if return_int:
            return int(info)
        else:
            return info

    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        
        # MODIFY THIS: Return motion data with priors
        motion_data = self._get_item_data_index(data_index)
        return motion_data

    def _load(self, ind, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(ind, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(ind, frame_ix)
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(ind, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
                    
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]

        # KEEP EXISTING FRAME SAMPLING LOGIC (no changes needed here)
        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            if num_frames > nframes:
                fair = False
                if fair:
                    choices = np.random.choice(range(nframes), num_frames, replace=True)
                    frame_ix = sorted(choices)
                else:
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes), padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes), num_frames, replace=False)
                frame_ix = sorted(choices)
            else:
                raise ValueError("Sampling not recognized.")

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

    # ADD THIS NEW METHOD: Create motion windows with priors
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

    # ADD THIS NEW METHOD: Create single window with prior
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

    # ADD THIS NEW METHOD: Generate prior motion
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

    # ADD THIS NEW METHOD: Pad sequences (torch version)
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

    # KEEP ALL EXISTING METHODS UNCHANGED:
    def get_label_sample(self, label, n=1, return_labels=False, return_index=False):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test
        total_samples = len(self._train)

        if n == 1:
            data_index = index[np.random.choice(total_samples)]
            x = self._get_item_data_index(data_index)
        else:
            data_index = np.random.choice(total_samples, n)  # FIX: was using undefined 'choices'
            x = [self._get_item_data_index(index[di]) for di in data_index]
            
        if return_labels:
            if return_index:
                return x, label, data_index
            return x, label
        else:
            if return_index:
                return x, data_index
            return x

    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=False, return_index=False) for label in labels]
        batch = collate(samples)
        return batch

    # KEEP REST OF METHODS UNCHANGED...
    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test