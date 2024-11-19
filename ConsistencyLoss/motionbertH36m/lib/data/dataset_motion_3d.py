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
