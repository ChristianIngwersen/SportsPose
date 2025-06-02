# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)
from typing import Literal

import numpy as np
from numpy import ndarray
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)
    
class DataReaderH36M(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

    def resolution(self, set: Literal["train", "test"]):
        camera_names = self.dt_dataset[set]["camera_name"]
        resolution = np.empty((2, len(camera_names)), dtype=np.int32)
        resolution[:] = 1000
        resolution[..., (camera_names == '60457274') | (camera_names == '54138969')] = 1002
        return resolution

    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32).T  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32).T  # [N, 17, 2]
        # map to [-1, 1]
        resolution = self.resolution("train")
        trainset /= resolution[0] / 2
        trainset[0] -= 1
        trainset[1] -= resolution[1] / resolution[0]

        resolution = self.resolution("test")
        testset /= resolution[0] / 2
        testset[0] -= 1
        testset[1] -= resolution[1] / resolution[0]

        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32).T
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32).T
                if len(train_confidence.shape)==2: # (17, 1559752)
                    train_confidence = train_confidence[None]
                    test_confidence = test_confidence[None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(1, *trainset.shape[1:])
                test_confidence = np.ones(1, *testset.shape[1:])
            trainset = np.concatenate((trainset, train_confidence), axis=0)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=0)  # [N, 17, 3]
        return trainset.T, testset.T

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32).T  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32).T    # [N, 17, 3]
        # map to [-1, 1]
        resolution = self.resolution("train")
        train_labels /= resolution[0] / 2
        train_labels[0] -= 1
        train_labels[1] -= resolution[1] / resolution[0]

        resolution = self.resolution("test")
        test_labels /= resolution[0] / 2
        test_labels[0] -= 1
        test_labels[1] -= resolution[1] / resolution[0]

        return train_labels.T, test_labels.T

    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        self.test_hw = self.resolution("test")
        return self.test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()
        gt_data = np.asarray(self.dt_dataset['test']['joints_2.5d_image'][::self.sample_stride])
        test_factor = np.asarray(self.dt_dataset['test']['2.5d_factor'][::self.sample_stride])[split_id_test]

        train_clip_source = np.asarray(self.dt_dataset['train']['source'][::self.sample_stride])[list(map(lambda x: x[0], split_id_train))]
        test_clip_source = np.asarray(self.dt_dataset['test']['source'][::self.sample_stride])[list(map(lambda x: x[0], split_id_test))]
        test_action = np.asarray(self.dt_dataset['test']['action'][::self.sample_stride])[list(map(lambda x: x[0], split_id_test))]

        train_data, test_data = (unflatten_batch_and_view(train_data[split_id_train], train_clip_source),
                                 unflatten_batch_and_view(test_data[split_id_test], test_clip_source))
        gt_test = unflatten_batch_and_view(gt_data[split_id_test], test_clip_source)

        # train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        train_labels, test_labels = (unflatten_batch_and_view(train_labels[split_id_train], train_clip_source),
                                    unflatten_batch_and_view(test_labels[split_id_test], test_clip_source))            # (N, 27, 17, 3)

        resolution_train, resolution_test = (unflatten_batch_and_view(self.resolution("train").T[split_id_train], train_clip_source),
                                            unflatten_batch_and_view(self.resolution("test").T[split_id_test], test_clip_source))

        factor_test = unflatten_batch_and_view(test_factor, test_clip_source)
        action_test = unflatten_batch_and_view(test_action, test_clip_source)

        remove_sequence = lambda res: list(map(lambda x: list(map(lambda y: y[..., 0, :], x)), res))
        resolution_train, resolution_test = remove_sequence(resolution_train), remove_sequence(resolution_test)
        remove_view = lambda res: list(map(lambda x: list(map(lambda y: y[0], x)), res))
        action_test = remove_view(action_test)

        return train_data, test_data, train_labels, test_labels, resolution_train, resolution_test, factor_test, action_test, gt_test


def unflatten_batch_and_view(array: ndarray, source: list) -> list[list[ndarray]]:
    """Given a list of sources of the form 's_{SUBJECT ID}_act_{ACTION ID}(.+)?_cam_{CAMERA ID}' with shape (B,) and an array
    of corresponding data with shape (B, ...), this function will reshape the array into (V, S, B', ...) where V is the
    number of cameras, S is the number of subjects, and B' is the number of samples for each subject and camera.

    :param array: The array of data to reshape.
    :param source: The list of sources to use for reshaping the array.
    :return: An array/list with shape (S, A, V, B', ...)
    """
    splits = np.asarray([s.split('_')[1::2] for s in source], dtype=int).T
    subject, activity, camera = splits[[0, 1, -1]]
    subject_indices = [np.where(subject == i)[0] for i in np.unique(subject)]
    array = [array[subject] for subject in subject_indices]
    activity_indices = [[np.where(activity[index] == i)[0] for i in np.unique(activity)] for index in subject_indices]
    array = [[a[index] for index in indices] for a, indices in zip(array, activity_indices)]

    # camera_indices = [[np.where(camera[index] == k)[0] for k in np.unique(camera)] for index in subject_indices]
    # array = [a[index] for index, a in zip(camera_indices, array)]
    camera_indices = [[[np.where(c[index] == k)[0] for k in np.unique(camera)] for index in indices] for c, indices in zip([camera[subject] for subject in subject_indices], activity_indices)]
    array = [[a2[c2] for a2, c2 in zip(a, c)] for a, c in zip(array, camera_indices)]
    return array
