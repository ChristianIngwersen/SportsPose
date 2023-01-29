import pandas as pd
import glob
import os
import numpy as np
import pickle


def get_total_capture_sequence_dist(data_path):
    """
    Get the TotalCapture vicon data and transform the global position to mm from inches.
    Then compute the pairwise euclidean distance between each point in the rows in the array.
    :param data_path: path to the total capture data
    :return: total capture data
    """

    joints = [
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Head",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
    ]

    csv_files = glob.glob(os.path.join(data_path, "**", "**", "*gbl_pos.txt"))
    sequence_movements = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep="\t")

        # For all the joints, get the x, y, z coordinates
        joint_movements = []
        for joint in joints:
            # Extract the x, y, z coordinates
            df[joint] = df[joint].apply(lambda x: x.split(" "))
            df[joint] = df[joint].apply(lambda x: [float(i) for i in x])

            # Convert to numpy array
            sequence = np.asarray(df[joint].tolist())

            # Compute pairwise euclidean distance between each point in the rows in the array
            sequence_dist = np.linalg.norm(sequence[1:, :] - sequence[:-1, :], axis=1)

            # Convert to cm from inches
            sequence_dist = sequence_dist * 2.54

            joint_movements.append(sequence_dist)

        # Get mean joint movements
        joint_movements = np.asarray(joint_movements).mean(axis=0)
        sequence_movements.append(joint_movements)

    return sequence_movements


def compute_frames_under_cm_threshold(sequence_movements, threshold=10):
    """
    Compute the number of frames under a certain threshold
    :param sequence_movements: sequence movements
    :param threshold: threshold
    :return: number of frames under the threshold
    """
    # Loop through each sequence and combine frames untill their combined movement is larger than the threshold
    frames = 0
    base_frames = 0
    for sequence_movement in sequence_movements:
        tmp = 0
        for movement in sequence_movement:
            if movement + tmp < threshold:
                base_frames += 1
                tmp += movement
            else:
                frames += 1
                base_frames += 1
                tmp = 0
    print(base_frames, frames)

    return base_frames, frames


def get_totalcapture_table_metrics():
    sequence_movement = get_total_capture_sequence_dist("data/totalcapture")
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)


def get_3dpw_sequence_dist(data_path):
    """
    Get the 3DPW data
    Then compute the pairwise euclidean distance between each point in the rows in the array.
    :param data_path: path to the 3DPW capture data
    :return: 3DPW data
    """
    pickle_files = glob.glob(os.path.join(data_path, "**", "*.pkl"))
    sequence_movements = []
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        joints = data["jointPositions"][0].reshape(-1, 24, 3)

        # Compute pairwise euclidean distance between each point in the rows in the array
        sequence_dist = np.linalg.norm(
            joints[1:, :, :] - joints[:-1, :, :], axis=2
        ).mean(axis=1)

        # Convert to cm from m
        sequence_dist = sequence_dist * 100

        sequence_movements.append(sequence_dist)

    return sequence_movements


def get_3dpw_table_metrics():
    sequence_movement = get_3dpw_sequence_dist("data/3dpw")
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)


if __name__ == "__main__":
    get_3dpw_table_metrics()
