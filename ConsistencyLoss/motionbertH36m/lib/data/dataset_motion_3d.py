from pathlib import Path
from typing import Literal

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data


class MotionDataset(Dataset):
    def __init__(self, data_root: Path, subset_list: list[str], data_split: Literal["test", "train"]): # data_split: train/test
        np.random.seed(0)
        self.data_root = data_root
        self.subset_list = subset_list
        self.data_split = data_split
        self.file_list = []
        for subset in self.subset_list:
            data_path = self.data_root / subset / self.data_split
            self.file_list += sorted(data_path.glob('*.pkl'))
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 


class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(Path(args["data_root"]), subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index) -> tuple[torch.FloatTensor, torch.FloatTensor, int] | tuple[torch.FloatTensor, torch.FloatTensor, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]


        if self.data_split == "test":
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[..., :2] = motion_3d[..., :2]
                motion_2d[..., 2] = 1
            return (
                torch.FloatTensor(motion_2d),
                torch.FloatTensor(motion_3d),
                motion_file["meta"]["performer"],
                motion_file["meta"]["resolution"],
                motion_file["meta"]["factor"],
                motion_file["meta"]["gt"],
                motion_file["meta"]["action"],
            )

        assert self.data_split == "train"

        if self.synthetic or self.gt_2d:
            motion_3d = self.aug.augment3D(motion_3d)
            motion_2d = np.empty(motion_3d.shape, dtype=np.float32)
            motion_2d[..., :2] = motion_3d[..., :2]
            motion_2d[..., 2] = 1                        # No 2D detection, use GT xy and c=1.
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), motion_file["meta"]["performer"]

        if motion_file["data_input"] is None:
            raise ValueError('Training illegal.')

        motion_2d = motion_file["data_input"]
        if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
            motion_2d = flip_data(motion_2d)
            motion_3d = flip_data(motion_3d)

        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), motion_file["meta"]["performer"]


class SkiPoseDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split, train_views = [0, 1, 2, 3, 4, 5]):
        self.data_path = Path(args["data_root"] + f"/{data_split}/{data_split}_sequences.pkl")
        self.data = read_pkl(self.data_path)
        self.data_split = data_split
        self.train_views = train_views
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.FloatTensor, torch.FloatTensor, int] | tuple[torch.FloatTensor, torch.FloatTensor, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        'Generates one sample of data'
        # Select sample
        data = self.data[index]
        motion_3d = data["gt_3D"]
        performer_id = data["subj"][0] + 2 # Add 2 to avoid 1, which is a special performer id in H36m
        if self.data_split == "test":
            motion_2d = data["pred_2D"].copy()
            motion_2d[..., :2] /= 256
            if self.gt_2d:
                motion_2d = data["gt_2D"]  # (C, S, 17, 2)
                # Add extra dimension
                motion_2d = np.concatenate([motion_2d, np.ones((motion_2d.shape[0], motion_2d.shape[1], motion_2d.shape[2], 1))], axis=-1)
                print("Using GT 2D")
            return (
                torch.FloatTensor(motion_2d),
                torch.FloatTensor(motion_3d),
                performer_id,
                256 + np.zeros((6, 2)),  # Camera resolution
                256./30,  # Camera factor 
                torch.FloatTensor(motion_3d)*1000, # Convert from meters to millimeters
                0,  # Action is just set to 0
                data["camera_position"],
                data["camera_R_cam_2_world"],
            )

        assert self.data_split == "train"
        pred_2d = data["pred_2D"].copy()
        pred_2d[..., :2] /= 256
        if self.synthetic or self.gt_2d:
            motion_3d = self.aug.augment3D(motion_3d)
            motion_2d = data["gt_2D"].copy()
            
            # Remove views not in train_views
            motion_2d = motion_2d[self.train_views]
            motion_3d = motion_3d[self.train_views]
            pred_2d = pred_2d[self.train_views]
            frame_existance = frame_existance[self.train_views]
            
            # Add extra dimension
            motion_2d = np.concatenate([motion_2d, np.ones((motion_2d.shape[0], motion_2d.shape[1], motion_2d.shape[2], 1))], axis=-1)                  # No 2D detection, use GT xy and c=1.
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), performer_id, torch.FloatTensor(data["frame_existance"]), torch.FloatTensor(pred_2d)

        frame_existance = torch.FloatTensor(data["frame_existance"])
        motion_2d = data["pred_2D"].copy() 
        motion_2d[..., :2] /= 256       
        if self.flip and random.random() > 0.5: # Training augmentation - random flipping
            motion_2d = flip_data(motion_2d)
            motion_3d = flip_data(motion_3d)
            
        # Remove views not in train_views
        motion_2d = motion_2d[self.train_views]
        motion_3d = motion_3d[self.train_views]
        pred_2d = pred_2d[self.train_views]
        frame_existance = frame_existance[self.train_views]
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), performer_id, torch.FloatTensor(frame_existance), torch.FloatTensor(pred_2d)