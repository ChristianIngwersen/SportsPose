import numpy as np
import h5py
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
from lib.utils.basis_transforms import transform_func, max_separated_xy_heatmap, rescale_pose, h36m_from_coco
import onnxruntime as ort
import torch

# Load the model
onnx_path = "/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504/end2end.onnx"
ort_session = ort.InferenceSession(onnx_path)
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
bones_h36m = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
joints_coco2_h36m = [0, 12, 13, 14, 6, 7, 8, 1, 2, 3, 4, 5, 9, 10, 11]


def infer_2D_pose(images, ort_session):
    # joint indice description


    pose_input_size = (384, 288)
    frames = np.stack([
        transform_func(frame.astype(np.float32) / 255.0, pose_input_size)
        for frame in images
    ])
    
    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: frames}
    x_heatmaps, y_heatmaps = ort_session.run(None, ort_inputs)
    
    # Convert output from heatmap to joints
    if len(x_heatmaps.shape) == 4:
        # Max separated xy heatmaps expects the shape to be (B, J, W) and (B, J, H) or (J, W) and (J, H)
        x_heatmaps, y_heatmaps = x_heatmaps[:, 1, ...], y_heatmaps[:, 1, ...]
    joints, pose_conf = max_separated_xy_heatmap(x_heatmaps, y_heatmaps)
    joints /= (x_heatmaps.shape[-1], y_heatmaps.shape[-1])
    
    joints = rescale_pose(joints, np.array([0, 0, 256, 256]), np.array(pose_input_size))  # Convert to original video resolution
    
    # Convert from COCO to H36M
    joints = np.array(h36m_from_coco(torch.tensor(joints)))
    return joints, pose_conf

def get_sequence(h5_label_file, mode, index):
    seq   = int(h5_label_file['seq'][index])
    indexing = np.array(h5_label_file['seq']) == h5_label_file['seq'][index]
    cams = h5_label_file['cam'][indexing]
    
    cam_ids = np.unique(h5_label_file['cam'][indexing])
    # Split the data into sequences based on the sequence number and camera number
    
    correct_cams = [cams == cam_id for cam_id in cam_ids]
    frames = [h5_label_file['frame'][indexing][correct_cam] for correct_cam in correct_cams]
    subjs = [h5_label_file['subj'][indexing][correct_cam][0] for correct_cam in correct_cams]
    pose_3D = [h5_label_file['3D'][indexing][correct_cam].reshape([-1, 17, 3]) for correct_cam in correct_cams] # (C, S, 17, 3)
    pose_2D =[h5_label_file['2D'][indexing][correct_cam].reshape([-1, 17, 2]) for correct_cam in correct_cams]  # (C, S, 17, 2)
    pose_2D_px = [256*pose_2Ds for pose_2Ds in pose_2D]
    camera_intrinsic = [h5_label_file['cam_intrinsic'][indexing][correct_cam] for correct_cam in correct_cams]
    camera_position = [h5_label_file['cam_position'][indexing][correct_cam] for correct_cam in correct_cams]
    camera_R_cam_2_world = [h5_label_file['R_cam_2_world'][indexing][correct_cam] for correct_cam in correct_cams]
        
    # Now stack and load the images for each camera
    image_path = []
    for i in range(len(cam_ids)):
        image_path.append(['./{}/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(mode, seq, int(cam_ids[i]), int(frame)) for frame in frames[i]])
        
    images = [[imageio.imread(image_name) for image_name in image_cam] for image_cam in image_path]
    # convert the the frames to format (C, S, H, W, 3)
    images = [np.stack(image_cam). transpose(0, 3, 1, 2) for image_cam in images]  # (C, S, 3, H, W)
    
    # Infer the 2D pose
    results = [infer_2D_pose(frames, ort_session) for frames in images]
    joints, pose_conf = zip(*results)
    # Add confidence to the joints in the last dimension
    joints = [np.concatenate([joints[i], np.expand_dims(pose_conf[i], axis=-1)], axis=-1) for i in range(len(joints))]
    for joint in joints:
        joint[:, 9, :] = joint[:, 10, :]  # Set head-top to head
    

    ### Pad the sequences to the same length
    first_frame = int(min([frame.min() for frame in frames]))
    last_frame = first_frame + 151  #int(max([frame.max() for frame in frames]))
    
    frame_existance = np.zeros((len(cam_ids), last_frame - first_frame + 1), dtype=bool)
    for i in range(len(cam_ids)):
        frame_existance[i, frames[i].astype("int") - first_frame] = True
    
    
    def pad_array(input_list, frame_existance):
        input_array = np.zeros((len(cam_ids), last_frame - first_frame + 1) + input_list[0].shape[1:])
        for i in range(len(cam_ids)):
            input_array[i][frame_existance[i]] = input_list[i] 
            ### This is pretty messy as we both have non-existing frames in the beginning and in the middle
            # Check if there is a hole in the beginning
            first_existing_frame = np.argmax(frame_existance[i])
            if first_existing_frame > 0:
                input_array[i, :first_existing_frame] = input_list[i][:first_existing_frame]
            # Check for holes in the middle
            for j in range(first_existing_frame, len(frame_existance[i])):
                if frame_existance[i][j] == False:
                    input_array[i, j] = input_array[i, j-1]
        return np.array(input_array)

    gt_3D = pad_array(pose_3D, frame_existance)
    gt_2D = pad_array(pose_2D, frame_existance)
    pose_2D_px = pad_array(pose_2D_px, frame_existance)
    camera_intrinsic = pad_array(camera_intrinsic, frame_existance)
    camera_position = pad_array(camera_position, frame_existance)
    camera_R_cam_2_world = pad_array(camera_R_cam_2_world, frame_existance)
    pred_2D = pad_array(joints, frame_existance)
    # Set confidence to 0 for missing frames
    pred_2D[~frame_existance][..., 2] = 0
    pred_conf = pad_array(pose_conf, frame_existance)

    sequence_data = {"subj": subjs,
                     "gt_3D": gt_3D,
                     "gt_2D": gt_2D,
                     "pose_2D_px": pose_2D_px,
                     "camera_intrinsic": camera_intrinsic,
                     "camera_position": camera_position,
                     "camera_R_cam_2_world": camera_R_cam_2_world,
                     "pred_2D": pred_2D,
                     "pred_conf": pred_conf,
                     "image_path": image_path,
                     "frame_existance": frame_existance}
    
    return sequence_data
        
def prepare_data(mode):
    # load label data
    h5_label_file = h5py.File(f'./{mode}/labels.h5', 'r')
    print('Available labels:',list(h5_label_file.keys()))
    
    sequences = []    
    unique_sequences = np.unique(h5_label_file['seq'])
    for unique_seq in tqdm(unique_sequences):
        index = np.argmax(np.array(h5_label_file['seq']) == unique_seq)
        sequences.append(get_sequence(h5_label_file, mode, index))
    
    # Save the sequences with pickle
    with open(f"{mode}_sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

if __name__ == "__main__":
    prepare_data("train")
    prepare_data("test")
    