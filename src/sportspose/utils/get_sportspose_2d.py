import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import cv2


def rotate_keypoints(keypoints: np.ndarray, image_height: int) -> np.ndarray:
    """
    Rotate a numpy array of 2D keypoints by 90 degrees clockwise.

    Args:
        keypoints: A numpy array of shape (batch_size, num_keypoints, 2)
                   representing the 2D keypoints.
        image_height: The height of the original image in pixels.

    Returns:
        A numpy array of shape (batch_size, num_keypoints, 2)
        representing the rotated keypoints.

    """
    rotated_keypoints = np.empty_like(keypoints)
    rotated_keypoints[:, :, 0] = image_height - keypoints[:, :, 1]
    rotated_keypoints[:, :, 1] = keypoints[:, :, 0]
    return rotated_keypoints


if __name__ == "__main__":
    data_path = "/Users/cin/Projects/SportsPose/data/sportspose"
    debug = False
    data3d = {}
    data2d = {}
    subject_count = 1
    subject_mapping = {}

    # Sessions
    session_paths = [
        x for x in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(x)
    ]
    for session in session_paths:
        # Get calibration files
        calibration_files = glob.glob(os.path.join(session, "*_calib.pkl"))
        video_files = sorted(glob.glob(os.path.join(session, "*.avi")))
        with open(calibration_files[0], "rb") as f:
            calibration = pickle.load(f)

        # Get all npy files
        npy_files = glob.glob(os.path.join(session, "**", "**", "*.npy"))

        for npy_file in npy_files:
            # Check if filename contains "golf"
            # if "golf" in npy_file:
            #    continue

            # Subject
            subject = npy_file.split("/")[-3]
            if subject not in subject_mapping:
                subject_mapping[subject] = f"S{subject_count}"
                subject_count += 1
                data3d[subject_mapping[subject]] = {}
                data2d[subject_mapping[subject]] = {}

            # Activity
            activity = npy_file.split("/")[-2]
            if activity not in data3d[subject_mapping[subject]]:
                data3d[subject_mapping[subject]][activity] = []
                data2d[subject_mapping[subject]][activity] = []

            # Load data
            joint3d = np.load(npy_file)

            with open(npy_file.replace(".npy", "_timing.pkl"), "rb") as f:
                timing = pickle.load(f)

            # Get the 2D coordinates
            cam_id = 6
            projection_matrix = calibration["calibration"][cam_id]["P"]

            # Make 3d joints of shape (N_frames, N_joints, 3) homogeneous i.e. (N_frames, N_joints, 3)
            joint3d = np.concatenate(
                [joint3d, np.ones((joint3d.shape[0], joint3d.shape[1], 1))], axis=2
            )

            # Project full list of 3d joints to 2d
            joint2d = np.matmul(joint3d, projection_matrix.T)

            # Normalize by the last coordinate
            joint2d = joint2d[:, :, :2] / joint2d[:, :, 2:]

            # Read rotation info from calib
            times_to_rotate = calibration["numtimesrot90clockwise"][cam_id]

            # Get original image dimensions
            cap = cv2.VideoCapture(video_files[cam_id])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Rotate coordinates of joints clockwise by 90 degrees using information about original image dimensions
            for _ in range(times_to_rotate):
                joint2d = rotate_keypoints(joint2d, height)

            # Add to data
            data3d[subject_mapping[subject]][activity].append(joint3d)
            data2d[subject_mapping[subject]][activity].append(joint2d)

            if debug:
                # for the first frame plot it for sanity
                test_im = np.zeros((height, width, 3))
                for _ in range(times_to_rotate):
                    test_im = cv2.rotate(test_im, cv2.ROTATE_90_CLOCKWISE)

                # write joints with opencv
                for joint in joint2d[0]:
                    cv2.circle(
                        test_im, (int(joint[0]), int(joint[1])), 10, (255, 255, 255), -1
                    )

                plt.imshow(test_im)

                plt.show()
    print()
