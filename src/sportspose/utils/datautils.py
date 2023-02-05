import pandas as pd
import glob
import os
import numpy as np
import pickle
import scipy.io


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
    print(f"{frames}/{base_frames}: {(frames/base_frames)*100:.2f}")

    return base_frames, frames


def get_totalcapture_table_metrics():
    print("TotalCapture")
    sequence_movement = get_total_capture_sequence_dist("data/totalcapture")
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)
    print()


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
        joints = joints[data["campose_valid"][0].astype(np.bool_), ...]

        # Compute pairwise euclidean distance between each point in the rows in the array
        sequence_dist = np.linalg.norm(
            joints[1:, :, :] - joints[:-1, :, :], axis=2
        ).mean(axis=1)

        # Convert to cm from m
        sequence_dist = sequence_dist * 100

        sequence_movements.append(sequence_dist)

    return sequence_movements


def get_3dpw_table_metrics():
    print("3DPW")
    sequence_movement = get_3dpw_sequence_dist("data/3dpw")
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)
    print()


def get_humaneva1_sequence_dist(data_path):
    """
    Get the HumanEva1 data
    Then compute the pairwise euclidean distance between each point in the rows in the array.
    :param data_path: path to the HumanEva1 capture data
    :return: sequence dist
    """
    # Get all mat files
    mat_files = glob.glob(os.path.join(data_path, "S*", "Mocap_Data", "*.mat"))
    sequence_movements = []
    for mat_file in mat_files:
        # Load mat file
        mat = scipy.io.loadmat(mat_file)

        # Get the joint positions
        joints = mat["Markers"]

        # Compute pairwise euclidean distance between each point in the rows in the array
        sequence_dist = np.linalg.norm(
            joints[1:, :, :] - joints[:-1, :, :], axis=2
        ).mean(axis=1)

        # Convert to cm from mm
        sequence_dist = sequence_dist / 10

        sequence_movements.append(sequence_dist)

    return sequence_movements


def get_humaneva1_table_metrics():
    print("HumanEva1")
    sequence_movement = get_humaneva1_sequence_dist("/home/cin/humaneva1")
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)
    print()


def get_humaneva2_sequence_dist(data_path):
    """
    Get the HumanEva2 data
    Then compute the pairwise euclidean distance between each point in the rows in the array.
    :param data_path: path to the HumanEva2 capture data
    :return: sequence dist
    """
    # Get all mat files
    mat_files = glob.glob(os.path.join(data_path, "S*", "Mocap_Data", "C*.mat"))
    sequence_movements = []
    for mat_file in mat_files:
        # Load mat file
        mat = scipy.io.loadmat(mat_file)

        # Get the joint positions
        joints = mat["Markers"]

        # Compute pairwise euclidean distance between each point in the rows in the array
        sequence_dist = np.linalg.norm(
            joints[1:, :, :] - joints[:-1, :, :], axis=2
        ).mean(axis=1)

        # Convert to cm from mm
        sequence_dist = sequence_dist / 10

        sequence_movements.append(sequence_dist)

    return sequence_movements


def get_sportspose_sequence_dist(data_path):
    """
    Get the SportsPose data
    Then compute the pairwise euclidean distance between each point in the rows in the array.
    :param data_path: path to the SportsPose capture data
    :return: sequence dist
    """
    # Get all npy files
    npy_files = glob.glob(os.path.join(data_path, "**", "**", "**", "*.npy"))
    sequence_movements = []

    for npy_file in npy_files:
        # Check if filename contains "golf"
        if "golf" in npy_file:
            continue
        # Load file
        joints = np.load(npy_file)

        # Select only wrist joints
        # joints = joints[:, [9, 10], :]

        # Compute pairwise euclidean distance between each point in the rows in the array
        sequence_dist = np.linalg.norm(
            joints[1:, :, :] - joints[:-1, :, :], axis=2
        ).mean(axis=1)

        # Convert to cm from m
        sequence_dist = sequence_dist * 100

        sequence_movements.append(sequence_dist)

    return sequence_movements


if __name__ == "__main__":
    # data_path = "/Users/cin/Projects/SportsPose/data/humaneva2/HumanEva_II"
    # x = get_humaneva2_sequence_dist(data_path)

    print("SportsPose")
    data_path = "/Users/cin/Projects/SportsPose/data/sportspose"
    sequence_movement = get_sportspose_sequence_dist(data_path)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=5)
    frames = compute_frames_under_cm_threshold(sequence_movement, threshold=10)
    print("")

    get_3dpw_table_metrics()
    # get_humaneva1_table_metrics()
    get_totalcapture_table_metrics()
