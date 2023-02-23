import os
import glob
import pickle
import numpy as np
import torchvision


if __name__ == "__main__":
    data_path = "/Users/cin/Projects/SportsPose/data/sportspose"
    video_base_path = "/Users/cin/Projects/SportsPose/data/sportspose_videos"
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

            # Get the video file
            video_files = sorted(
                os.path.join(video_base_path, *timing["video_path"], "*.avi")
            )

            for video_file in video_files:
                # Crop the frames around the person
                # Get the first frame
                frame = torchvision.io.read_video(video_file, start_pts=0, end_pts=1)
                frame = frame[0][0]

                # Get the prediction
                prediction = model(frame)
                boxes = prediction[0]["boxes"]
                scores = prediction[0]["scores"]
                labels = prediction[0]["labels"]

                # Get all boxes where the label is 1 (person) and select the one with the highest score
                indexer = labels == 1
                boxes = boxes[indexer]
                scores = scores[indexer]
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

                # Update the calibration matrix
                calibration["K"][0, 2] -= xmin
                calibration["K"][1, 2] -= ymin

                # Rebuild projection matrix and handle translation
                projection = np.matmul(calibration["K"], calibration["R"])
                projection = np.concatenate(
                    (projection, calibration["T"].reshape(3, 1)), axis=1
                )

                if False:
                    # Build the ffmpeg command
                    output = video.replace("raw", "interim")
                    command = f'ffmpeg -y -i {video} -filter:v "crop={width}:{height}:{xmin}:{ymin}" {output}'

                    # Run the command
                    subprocess.run(command, shell=True)
