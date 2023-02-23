import os
import glob
import pickle
import numpy as np
import torchvision
import subprocess
import torch
import json


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
    data_path = r"D:\MarkerlessSportsData\processed\MarkerlessEndBachelor_withVideoPaths"
    video_base_path = r"D:\MarkerlessSportsData"
    out_dir = r"D:\SportsPose"
    debug = False
    data3d = {}
    data2d = {}
    subject_count = 1
    subject_mapping = {}

    # Sessions
    session_paths = [
        x for x in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(x)
    ]

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    model.eval()

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
            subject = npy_file.split(os.sep)[-3]
            if subject not in subject_mapping:
                subject_mapping[subject] = f"S{subject_count}"
                subject_count += 1
                data3d[subject_mapping[subject]] = {}
                data2d[subject_mapping[subject]] = {}

            # Activity
            activity = npy_file.split(os.sep)[-2]
            if activity not in data3d[subject_mapping[subject]]:
                data3d[subject_mapping[subject]][activity] = []
                data2d[subject_mapping[subject]][activity] = []

            # Load data
            joint3d = np.load(npy_file)
            activity_data = {}
            activity_data["joints3d"] = joint3d.tolist()
            activity_data["joints2d"] = {}

            # Make 3d joints of shape (N_frames, N_joints, 3) homogeneous i.e. (N_frames, N_joints, 3)
            joint3d = np.concatenate(
                [joint3d, np.ones((joint3d.shape[0], joint3d.shape[1], 1))], axis=2
            )

            with open(npy_file.replace(".npy", "_timing.pkl"), "rb") as f:
                timing = pickle.load(f)

            # Get the video file
            video_files = sorted(
                glob.glob(os.path.join(video_base_path, *timing["video_path"], "*.avi"))
            )

            new_calib = {}

            for cam_id, video_file in enumerate(video_files):
                # Crop the frames around the person
                # Get the first frame
                frame = torchvision.io.read_video(video_file, start_pts=0, end_pts=1)
                frame = frame[0][0]
        
                # Convert to float and channels first
                frame = frame.float() / frame.max()
                frame = frame.permute(2, 0, 1).unsqueeze(0)

                # Get the prediction
                with torch.no_grad():
                    prediction = model(frame)
                boxes = prediction[0]["boxes"]
                scores = prediction[0]["scores"]
                labels = prediction[0]["labels"]

                # Get calib
                current_calib = calibration["calibration"][cam_id]

                times_to_rotate = calibration["numtimesrot90clockwise"][cam_id]
                width = frame.shape[3]
                height = frame.shape[2]
                cam_matrix = current_calib["A"]
                rot = current_calib["R"]
                trans = current_calib["T"]
                essential_mat = np.zeros((3, 4))
                essential_mat[:3, :3] = rot
                essential_mat[:3, 3] = trans

                # Get all boxes where the label is 1 (person) and select the one with the highest score
                indexer = labels == 1
                boxes = boxes[indexer]
                scores = scores[indexer]
                cropped = False
                if not len(scores) == 0:
                    if scores.max() > 0.5:
                        cropped = True
                        box = boxes[scores.argmax()]

                        xmin, ymin, xmax, ymax = box.tolist()

                        # expand the box
                        width = xmax - xmin
                        height = ymax - ymin
                        xmin -= height * 0.5
                        xmax += height * 0.5
                        ymin -= height * 0.2
                        ymax += height * 0.2

                        # Get the dimensions of the video
                        video_width = frame.shape[3]
                        video_height = frame.shape[2]

                        # Adjust the crop region if it is larger than the video
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax > video_width:
                            xmax = video_width
                        if ymax > video_height:
                            ymax = video_height

                        # Calculate the dimensions of the crop region
                        width = xmax - xmin
                        height = ymax - ymin

                        # Update the calibration matrix - remeber to check from which camera it comes from
                        cam_matrix[0, 2] -= xmin
                        cam_matrix[1, 2] -= ymin
                
                cam = os.path.split(os.path.splitext(video_file)[0])[-1]
                new_calib[os.path.split(os.path.splitext(video_file)[0])[-1]] = {}
                new_calib[os.path.split(os.path.splitext(video_file)[0])[-1]]["P"] = cam_matrix @ essential_mat
                new_calib[os.path.split(os.path.splitext(video_file)[0])[-1]]["R"] = rot
                new_calib[os.path.split(os.path.splitext(video_file)[0])[-1]]["T"] = trans
                new_calib[os.path.split(os.path.splitext(video_file)[0])[-1]]["A"] = cam_matrix
                new_calib[cam]["numtimesrot90clockwise"] = times_to_rotate


                # Get 2D joints:

                # Project full list of 3d joints to 2d
                joint2d = np.matmul(joint3d, new_calib[cam]["P"].T)

                # Normalize by the last coordinate
                joint2d = joint2d[:, :, :2] / joint2d[:, :, 2:] 

                # Rotate coordinates of joints clockwise by 90 degrees using information about original image dimensions
                for _ in range(times_to_rotate):
                    joint2d = rotate_keypoints(joint2d, height)
                activity_data["joints2d"][cam] = joint2d.tolist()
                
                # Build the ffmpeg command
                # Subject folder
                if not os.path.exists(os.path.join(out_dir, subject_mapping[subject])):
                    os.mkdir(os.path.join(out_dir, subject_mapping[subject]))

                # Activity folder
                if not os.path.exists(os.path.join(out_dir, subject_mapping[subject], activity)):
                    os.mkdir(os.path.join(out_dir, subject_mapping[subject], activity))

                # Frame folder
                if not os.path.exists(os.path.join(out_dir, subject_mapping[subject], activity, os.path.split(os.path.split(video_file)[0])[-1])):
                    os.mkdir(os.path.join(out_dir, subject_mapping[subject], activity, os.path.split(os.path.split(video_file)[0])[-1]))
                
                # Cam folder
                final_dir = os.path.join(out_dir, subject_mapping[subject], activity, os.path.split(os.path.split(video_file)[0])[-1], os.path.split(os.path.splitext(video_file)[0])[-1])
                if not os.path.exists(final_dir):
                    os.mkdir(final_dir)
                 
                output = os.path.join(final_dir, "tmp.mp4")

                if cropped:
                    command = f'ffmpeg -y -i {video_file} -filter:v "crop={width}:{height}:{xmin}:{ymin}" {output}'
                else: 
                    command = f"ffmpeg -y -i {video_file} {output}"

                # Run the command
                _ = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Rotate cmd
                for i in range(times_to_rotate):
                    invid = output
                    output = os.path.join(final_dir, f"tmp{i}.mp4")
                    command = ["ffmpeg", "-y", "-i", invid, "-vf", "transpose=1", output]
                    
                    _ = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    os.remove(invid)

                # Extract frames
                _ = subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            output,
                            f"{final_dir}/%04d.png",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                # Remove tmp
                os.remove(output)

                # Serialize calib
                for k, v in new_calib[cam].items():
                    if isinstance(v, np.ndarray):
                        new_calib[cam][k] = v.tolist()

            activity_data["calib"] = new_calib
            json_out = os.path.join(out_dir, subject_mapping[subject], activity, os.path.split(os.path.split(video_file)[0])[-1], "data.json")

            # save file
            with open(json_out, "w") as outfile:
                json.dump(activity_data, outfile)
