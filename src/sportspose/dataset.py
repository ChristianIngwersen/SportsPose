import os
import pickle
import re
import warnings

import cv2
import numpy as np
import torch.utils.data
from PIL import Image

from sportspose.camera import Camera
from sportspose.utils import SPORTSPOSE_CAMERA_INDEX_RIGHT, VIEW_TO_SERIAL, chunks

import decord


class Measurement:
    """
    Class for storing and receiving measurement data from different types of datasets.
    Can currently read from SportsPose datasets.

    Attributes:
        joints_3d (dict): Dictionary containing 3D joint data.
        metadata (dict): Dictionary containing metadata.
        video (dict): Dictionary containing video information.
        calibration (dict): Dictionary containing calibration information.

    """

    def __init__(self, data_dict={}, meta_data={}, video_info={}):
        """
        Initialize the Measurement class.

        Args:
            data_dict (dict): Dictionary containing 3D joint data.
            meta_data (dict): Dictionary containing metadata.
            video_info (dict): Dictionary containing video information.
        """
        self.joints_3d = {
            "valid_frames": None,
            "file_path": None,
            "data_points": None,
        }
        self.metadata = {}
        self.calibration = {}
        self.video = {"views": {}}

        self.joints_3d.update(data_dict)
        self.metadata.update(meta_data)
        self.video.update(video_info)
        self.name_dict = {
            "joints_3d": self.joints_3d,
            "metadata": self.metadata,
            "video": self.video,
        }

    def from_sportspose(self, joints_3d, metadata, video):
        """
        Initialize the Measurement class from a SportsPose dataset.

        Args:
            joints_3d (dict): Dictionary containing 3D joint data.
            metadata (dict): Dictionary containing metadata.
            video (dict): Dictionary containing video information.

        Returns:
            Measurement: Measurement object.
        """
        # joints
        self.joints_3d.update(joints_3d)

        # Metadata
        self.metadata.update(metadata)

        # Video
        self.video.update(video)

        # Fill out additional variables
        view_calibs = {}
        for view in self.video["views"]:
            calib = self.video[view]["calibration"]
            view_calibs[view] = calib

            # set up camera object
            cam = Camera()
            cam.intrinsic.f = calib["f"]
            cam.intrinsic.c = calib["c"]
            cam.intrinsic.k = calib["k"]
            cam.extrinsic.T = calib["T"]
            cam.extrinsic.R = calib["R"]
            cam.extrinsic.tpr = calib["tpr"]
            self.video[view]["camera"] = cam

        self.calibration = view_calibs
        return self

    def construct_camera(self):
        """Function for initialising cameras with calibration to dataset
        Args:
            cameras (list): List of cameras to add
        """
        # Initilize cameras for each measurement and each view
        # Get camera calibrations dict indexed by serial number
        cam_calibs = {c["serial"]: c for c in self.video["calibration"]["cameras"]}
        for view in self.video["views"]:
            cam_calib = cam_calibs[VIEW_TO_SERIAL[view]]
            self.video[view]["numtimesrot90clockwise"] = (
                int(cam_calib["viewrotation"]) // 90
            ) % 4
            # Image will be rotated to 0 degrees by dataset, so set this value to 0
            cam_calib["viewrotation"] = "0"
            cam = Camera()
            cam.from_calib_dict(cam_calib)
            # Save a camera for each view in each measurement
            self.video[view]["camera"] = cam


class SyncIndex:
    """Class for storing the indices of the synchronized data

    Attributes:
        measurement_idx (int): Index of the measurement, (usually stored in dataset.measurements)
        video_idx (int): Index of the video in the measurement
        joints_3d_idx (int): Index of the joints_3d in the measurement
    """

    def __init__(self, measurement_idx, video_idx, joints_3d_idx):
        """Init function for SyncIndex class

        Args:
            measurement_idx (int): Index of the measurement
            video_idx (int): Index of the video
            joints_3d_idx (int): Index of the joints_3d
        """
        self.measurement_idx = measurement_idx
        self.video_idx = video_idx
        self.joints_3d_idx = joints_3d_idx


def sportspose_load_function(data_dir, video_root_dir, views={"right": {}}):
    """
    Assumes single dir to sportspose outer folder.
    Within this folder the structure should be
    Day -> Person -> Activity ->

    views is dictionary of views to include. Each view has unique
        dictionary that maps from session names to what camera index was
        this view for that session. Updates current known values,
        so only needed for future sessions

    Args:
        data_dir (str): Path to the root of the dataset
        video_root_dir (str): Path to the root directory of the videos
        views (dict): Dictionary of views to load

    Returns:
        measurements (list): List of measurements
    """

    measurements = []

    # Update known indices for sessions
    if "right" in views:
        views["right"].update(SPORTSPOSE_CAMERA_INDEX_RIGHT)

    # regex to find propperly named folders/files
    re_day = r"^[inout]+doors$"
    re_person = r"^S[0-9]{2}$"
    re_activity = r"^[a-z_]+$"
    re_file = r"^[a-z_]+[0-9]{4}.npy$"

    for i, dayname in enumerate(os.listdir(data_dir)):

        path_day = os.path.join(data_dir, dayname)
        if not re.match(re_day, dayname) or not os.path.isdir(path_day):
            continue

        for j, personname in enumerate(os.listdir(path_day)):

            calibfile = os.path.join(path_day, personname, "calib.pkl")
            if not os.path.isfile(calibfile):
                warnings.warn(
                    "Warning! Cannot find calibration file {calibfile}. Skipping measurement, as no projection "
                    "can be made from 3D points."
                )
                continue  # Do not add measurement when no calibration file can be found
            else:
                with open(calibfile, "rb") as f:
                    calib = pickle.load(f)

            path_person = os.path.join(path_day, personname)
            if not re.match(re_person, personname) or not os.path.isdir(path_person):
                continue

            for o, activityname in enumerate(os.listdir(path_person)):

                path_activity = os.path.join(path_person, activityname)
                if not re.match(re_activity, activityname) or not os.path.isdir(
                    path_activity
                ):
                    continue

                for filename in os.listdir(path_activity):

                    if re.match(re_file, filename):
                        # Add this file as a measurement
                        # Gather metadata first
                        jointspath = os.path.join(path_activity, filename)
                        metadata = {
                            "dir_name": jointspath,
                            "activity": activityname,
                            "person_id": personname,
                            "dataset": "SportsPose",
                            "markers_exists": dayname == "qualisys",
                            "tag": dayname,
                        }

                        # Currently missing how to get valid_frames and framerate
                        data_joints_3d = np.load(jointspath)
                        framerate = 90

                        joint_3d = {
                            "data_points": data_joints_3d,
                            "data_points_shape": data_joints_3d.shape,
                            "file_path": jointspath,
                            "valid_frames": None,
                            "frame_rate": framerate,
                        }

                        viewlist = list(views.keys())
                        video = {"views": viewlist}

                        timingfile = os.path.join(
                            path_activity, filename.replace(".npy", "_timing.pkl")
                        )

                        if not os.path.isfile(timingfile):
                            warnings.warn(
                                "Warning! Cannot find timing file {timingfile}. Skipping measurement as no "
                                "information on camera frame timing or camera filepath can be extracted!"
                            )
                            # set corresponding values
                            abs_video_path = ""
                        else:
                            with open(timingfile, "rb") as f:
                                timing = pickle.load(f)

                            # path of video
                            rel_video_path = os.path.join(*timing["video_path"])
                            abs_video_path = os.path.join(
                                video_root_dir, rel_video_path
                            )

                            metadata["datetime"] = rel_video_path[-17:]

                            if not os.path.isdir(abs_video_path):
                                warnings.warn(
                                    f"Cannot find the given video at path {abs_video_path}! Please update video_root_dir, or move files to here!"
                                )

                        for view in viewlist:
                            if not (dayname in views[view]):
                                warnings.warn(
                                    f"Warning! Cannot find index of video for {view} view. Video path cannot be inferred."
                                )
                            else:
                                # video index
                                video_idx = views[view][dayname][personname]
                                video[view] = {}
                                video[view]["camera_index"] = video_idx

                                # video paths
                                video_view_path = os.path.join(
                                    abs_video_path, "CAM" + str(video_idx) + ".avi"
                                )
                                video[view]["path"] = [video_view_path]

                                # take timestamps from timings file
                                video[view]["timestamps"] = (
                                    timing["video_index"][video_idx] / framerate
                                )
                                video[view]["timestamps_ms"] = timing["times_ms"][
                                    video_idx
                                ]
                                video[view]["timestamps_ms"] -= video[view][
                                    "timestamps_ms"
                                ][0]

                                # calibrations
                                video[view]["calibration"] = calib["calibration"][
                                    video_idx
                                ]
                                video[view]["numtimesrot90clockwise"] = calib[
                                    "numtimesrot90clockwise"
                                ][video_idx]

                        # Add measurement to return list
                        measurements.append(
                            Measurement().from_sportspose(joint_3d, metadata, video)
                        )

    # Check that measurements not empty
    if len(measurements) == 0:
        warnings.warn(
            f"Warning! No measurement objects are returned! Please check the input paths!"
        )
    return measurements


class SportsPoseDataset(torch.utils.data.Dataset):
    """
    The class for storing image and 3D pose data in a dataset.

    The dataset stores its data in list of measurements, where each measurement is a dictionary with the following keys:
    - "video": The data related to video
    - "joints_3d": The 3D joint positions
    - "metadata": The metadata related to the measurement

    During indexing and sampling, the dataset can return atrributes of the measurements, such as "joints_3d.datapoints".
    The attributes are specified by the return_preset argument, which can be either a string or a dictionary.
    If the attributes are not explicitly stored, the way to compute them is specified in the self.key2func dictionary.
    """

    def __init__(
        self,
        data_dir,
        dataset_type="sportspose",
        return_preset="3d_pose_estimation",
        seq_size=1,
        overlap_size=0,
        ts_view="right",
        do_skip_invalid_frames=True,
        convert_3d_points=None,
        views=["right"],
        transform=None,
        swing_idxs=None,
        video_root_dir=None,
        marker_format="coco",
        blacklist={},
        whitelist={},
        sample_level="frame",
        sample_method=None,
        validation_dataset=False,
    ):
        """
            Args:
        data_dir (str): The path to the dataset directory root directory.
        dataset_type (str): The type of dataset. Currently only support for "sportspose".
        return_preset (str or dict): The preset for returning batches. It must be one of the following:
            (string) "3d_pose_estimation": Return batches of images and 2D + 3D pose annotations.
            (dict): A nested dictionary specifying which attributes to return by specifying their keys to be True
        seq_size (int): The number of consecutive frames to return in a batch.
        overlap_size (int): The number of overlapping frames between consecutive batches.
        ts_view (str): The view to use for temporal sampling. Must be one of the views in the dataset.
        do_skip_invalid_frames (bool): Whether to skip frames with invalid 3D pose annotations.
        convert_3d_points (callable): A callable to convert 3D points from the dataset format to the desired format.
        views (list): A list of views to load from the dataset.
        transform (callable): A callable to transform the data.
        swing_idxs (list): A list of indices to use for temporal sampling.
        video_root_dir (str): The path to the root directory of the videos for dataset of type "sportspose".
        marker_format (str): The format of the marker annotations. Must be one of the following:
            (string) "coco": The COCO format.
        blacklist (dict): A dictionary of lists of attributes to exclude from the dataset.
        whitelist (dict): A dictionary of lists of attributes to include in the dataset.
        sample_level (str): The level to sample from the dataset. Must be one of the following:
            (string) "frame": Sample based on the frames.
            (string) "video": Sample from the videos. In this case "sample_method" can be provided,
                                otherwise uniform over frames is used.
        sample_method (callable): A callable to sample from the dataset. If None, the default sampling method is used.
        """
        super().__init__()
        self.marker_format = marker_format
        self.sample_level = sample_level
        self.sample_method = (
            self.sample_uniform if sample_method is None else sample_method
        )
        self.validation_dataset = validation_dataset
        if dataset_type.lower() == "sportspose":

            if video_root_dir is None:
                video_root_dir = os.path.join(data_dir, "videos")

            data_dir = os.path.join(data_dir, "data")

            self.measurements = sportspose_load_function(
                data_dir, video_root_dir, views={"right": {}}
            )

        # Check if the settings for returning batches are a preset or custom setting
        if isinstance(return_preset, dict):
            self.batch_return_vals = return_preset
        else:
            presets = {
                "3d_pose_estimation": {
                    "joints_2d": True,
                    "joints_3d": {
                        "data_points": True,
                    },
                    "video": {
                        "views": True,
                        "image": True,
                        "view": {"path": True},
                    },
                }
            }
            # What data to pass along in batch
            self.batch_return_vals = presets[return_preset]

        # Dictionary to store which functions to get values from when not explicitly stored
        self.key2func = {
            "joints_3d": {
                "data_points": self.idx2joint3d,
                "data_points_full": self.idx2joint3d_full,
            },
            "joints_2d": self.idx2joints2d,  # shape: (measurement, view, joints)
            "video": {
                "image": self.idx2image,
                "calibration": self.idx2calibration,
                "image_names": self.idx2img_name,
                "view": {
                    "path": self.idx2image_path},
            },
            "metadata": {
                "file_name": self.idx2name,  # shape: (measurement, )
            },
        }

        # Check for video-joint 2d specific transforms
        self.joints_2d_exists = False
        self.video_transform = False
        if "video" in self.batch_return_vals.keys():
            if "image" in self.batch_return_vals["video"].keys():
                # Only enable video transform if there actually is a transform
                if transform is not None:
                    self.video_transform = True
        if "joints_2d" in self.batch_return_vals.keys():
            self.joints_2d_exists = True

        self.blacklist = blacklist
        self.whitelist = whitelist

        self.convert_3d_points = convert_3d_points
        self.do_skip_invalid_frames = do_skip_invalid_frames
        self.swing_idxs = swing_idxs
        self.seq_size = seq_size
        self.views = views

        # Transform
        self.transform = transform
        if self.transform is None:
            self.transform = lambda x: x

        # Create index map: idx -> (measurement_idx, range(start, end), marker_idxs)
        self.index_map = {}
        self.index = 0
        self.video_idx2frame_idx = {}
        for measurement_idx, measurement in enumerate(self.measurements):
            # Go through all measurements and images to create index map
            self.add_measurement(measurement, measurement_idx, ts_view, overlap_size)

        if self.__len__() == 0:
            raise ValueError(
                "No valid frames found in dataset. Please check your settings."
            )

    def sample_video(self, measure_idx, sample_method):
        """
        Function for selecting frames using sample_method in video specified by measure_idx.

        Args:
            measure_idx (int): Index of measurement of video to sample from
            sample_method (function): Function for sampling frames from video

        Returns:
            (int): Index of frame to sample
        """
        measurement = self.measurements[measure_idx]
        return sample_method(self, measurement)

    def add_frames(self, image_idx_range, joint_idx, measurement, measurement_idx, ts_view):
        """Helper function for adding data samples containing (measurement, images, and corresponding markers)
        Args:
            image_idx_range (range): The range of the images in the sequence
            measurement (Measurement): Measurement containing the measured values to store
            measurement_idx (int): Index of the measurement to add in self.measurement
            ts_view (str): View of frame to add

        Returns:
            None if the sequence is not added, otherwise the index of the sequence
        """
        # Edge case when seq size does not evenly divide number of elems
        if len(image_idx_range) != self.seq_size:
            return

        # Get marker indices from timestamps
        marker_idxs = range(joint_idx, joint_idx+self.seq_size)

        image_idxs = list(image_idx_range)

        # Skip entire sequence if there is an invalid frame
        if measurement.joints_3d["valid_frames"] is not None:
            try:
                valid_frames = measurement.joints_3d["valid_frames"][marker_idxs]
                if self.do_skip_invalid_frames and not np.all(valid_frames):
                    return
            except KeyError:
                pass

        # Check if datapoint contains all required views
        for view in self.views:
            if view not in measurement.video["views"]:
                # Do not add datapoint if all required views are not present
                return

        # Class for bookmarking: measurement, images, and corresponding markers
        syn_idx = SyncIndex(measurement_idx, image_idxs, marker_idxs)
        self.index_map[self.index] = syn_idx
        self.index += 1
        return self.index - 1  # Return the index in frames

    def infer_attributes(self, measure):
        """Wrapper function for adding automatic inference of some attributes during building of dataset.
        This currently includes the following:
            - joints_3d["data_points_shape"]    : The overall shape of the joints_3d data points (measurement, joints, dim)
            - video[view]["frame_from_video"]  : Whether the frame is from a video or an image (bool)
            - video[view]["n_frames"]         : Number of frames in video (int)
            - video[view]["img_dims"]        : Dimensions of image (height, width, channels)

        Args:
            measure (Measurement): Measurement to infer attributes for

        Returns:
            Nothing
        """
        # First sort the views for consistency
        measure.video["views"] = sorted(measure.video["views"])
        if self.marker_format == "coco":
            if "data_points_coco" in measure.joints_3d:
                measure.joints_3d["data_points"] = measure.joints_3d["data_points_coco"]

        measure.joints_3d["data_points_shape"] = measure.joints_3d["data_points"].shape

        for view in self.views:
            frame_path = measure.video[view]["path"]
            measure.video[view]["frame_from_video"] = not self.is_image(frame_path)
            measure.video[view]["n_frames"] = measure.video[view]["timestamps"].shape[0]

            # Get dimensions of image
            frame = frame_path[0]
            if measure.video[view]["frame_from_video"]:
                vid = cv2.VideoCapture(frame)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                # Load image to get dimensions
                new_img = Image.open(frame_path[0])
                width, height = new_img.size
                new_img.close()
            measure.video[view]["img_dims"] = torch.tensor([width, height]).int()

    def add_measurement(self, measurement, measurement_idx, ts_view, overlap_size):
        """Function for decomposing a sequential measurements into individual datapoints
        Args:
            measurement (Measurement): Measurement containing the measured values to store
            measurement_idx (int): The index of the overall measurement
            ts_view (str): String containing the view to add
            overlap_size (int): The size of the overlap of measurements between each data point

        Returns:
            Nothing, but adds the measurement to the dataset
        """
        measurement_name = measurement.metadata["dir_name"]
        # Get timestamps for video capture in seconds
        image_tss = measurement.video[ts_view]["timestamps"]

        # Infer additional attributes of measurement
        self.infer_attributes(measurement)
        # Get specific indices for swing start and stop if specified
        if self.swing_idxs:
            start, stop = self.swing_idxs.get(measurement_name, (None, None))
            if start is None:
                return  # Skip measurement if no swing detected
        else:
            start, stop = (0, len(image_tss))

        frame_idxs = {}
        seq_added = 0
        # Go through images in chunks of seq size
        frame_seqs = chunks(range(start, stop), self.seq_size, overlap_size)
        if self.filter_datapoint(measurement):
            for joint_idx, image_idx_range in enumerate(frame_seqs):
                frame_idx = self.add_frames(image_idx_range, joint_idx, measurement, measurement_idx, ts_view)
                if frame_idx is not None:
                    # If datapoint was added, also add it to the measurement list
                    frame_idxs[seq_added] = frame_idx
                    seq_added += 1

        if len(frame_idxs) > 0:
            measurement.video["n_valid_seq"] = seq_added
            # Remember the video_idx is based on the valid sequences sampled from the dataloader
            self.video_idx2frame_idx[len(self.video_idx2frame_idx)] = frame_idxs

    def filter_datapoint(self, measure):
        """Function for checking if measure is legal to add to dataset under black and whitelist.
        Note the blacklist takes precesdens over whitelist,
        so adding "person_id: S05" to whitelist, but "activity: volley" to blacklist,
        would yield true for all measurements of S05  except where he/she plays volley.

        Args:
            measure (Measurement): measurement to add as datapoint
        Returns:
            bool: If the measure is allowed in the dataset
        """

        # Set default dictionary to get keys from
        blacklist = self.blacklist
        whitelist = self.whitelist
        key2name = measure.name_dict

        def check_blacklist(key2name, blacklist):
            """Function for checking if attribute is in blacklist.
            The nestings of the dictionaries are recursively checked.

            Args:
                key2name (dict): Dictionary containing the keys to the attributes
                blacklist (dict): Dictionary containing the attributes to blacklist
            """
            # First run through blacklist
            for key, value in blacklist.items():
                if isinstance(value, dict):
                    #  Case where dict is further nested
                    res = check_blacklist(key2name[key], blacklist[key])
                    if res is False:
                        return res
                else:
                    # Case where attribute is child in dict
                    if key2name[key] in value:
                        # Blacklist takes precedence over whitelist
                        return False
            # Case where no attributes in blacklist was detected
            return True

        def check_whitelist(key2name, whitelist):
            """Function for checking if attribute is in whitelist
            The nestings of the dictionaries are recursively checked.

            Args:
                key2name (dict): Dictionary containing the keys to the attributes
                whitelist (dict): Dictionary containing the attributes to whitelist
            """
            # First run through whitelist
            for key, value in whitelist.items():
                if isinstance(value, dict):
                    #  Case where dict is further nested
                    res = check_whitelist(key2name[key], whitelist[key])
                    if res is False:
                        return res

                else:
                    # Case where attribute is child in dict
                    if key2name[key] not in value:
                        # If attribute is used in whitelist, but not mentioned in whitelist return False
                        return False
            # Case where no attributes in blacklist was detected
            return True

        outside_blacklist = check_blacklist(key2name, blacklist)
        if outside_blacklist is False:
            return outside_blacklist
        in_whitelist = check_whitelist(key2name, whitelist)
        return in_whitelist

    def idx2joint3d(self, index):
        """Get 3D joints from overall index of dataset
        Args:
            index (int): Overall index of dataset

        Returns
            joints_3d (np.array): 3D joints for the measurement
        """
        sync_index = self.index_map[index]
        joints_3d = self.get_3d_joints(sync_index, self.convert_3d_points)
        return joints_3d

    def idx2joint3d_full(self, index):
        """Get 3D joints with all available datapoints from overall index of dataset
        Args:
            index (int): Overall index of dataset

        Returns
            joints_3d (np.array): Full 3D joints for the measurement
        """
        sync_index = self.index_map[index]
        joints_3d = self.measurements[sync_index.measurement_idx].joints_3d[
            "data_points_full"
        ]
        return joints_3d[sync_index.joints_3d_idx, ...]

    def rotate_joint2d(self, measure, view, data):
        """Function for rotating 2D joints to match images.

        Args:
            measure (Measurement): Measurement to rotate joints for
            view (str): View to rotate joints for
            data (np.array): 2D joints to rotate

        Returns:
            np.array: Rotated 2D joints
        """
        width, height = measure.video[view]["img_dims"].numpy()
        center = (height / 2, width / 2)
        rot = measure.video[view]["numtimesrot90clockwise"]
        if rot == 0:
            trans = np.array([0, 0])
        elif rot == 1:
            trans = np.array([height, 0])
        elif rot == 2:
            trans = np.array([height, width])
        elif rot == 3:
            trans = np.array([0, width])
        rotation = cv2.getRotationMatrix2D(center, rot * 90, 1)[:, :2]
        return data @ rotation + trans[None, None]

    def idx2joint2d(self, index, view, rotate=True):
        """Function for returning the 2D joints of given index
        Args:
            index (int): Index of measure to return 2D joints for
            view (str): View to project 3D joints to 2D
            view (bool): If True, rotate joints to match image
        return:
            joint_2d (np.array): 2D joints
        """
        sync_index = self.index_map[index]
        measure = self.measurements[sync_index.measurement_idx]
        cam = measure.video[view]["camera"]
        joints_3d = self.idx2joint3d(index)
        joint_2d = cam.project(joints_3d)
        if rotate:
            joint_2d = self.rotate_joint2d(measure, view, joint_2d)
        return joint_2d

    def idx2joints2d(self, index):
        """Function for getting 2D joints from all views given an index
        Args:
            index (int): index in dataset to get joints from
        returns:
            view_item (Dict): Dict containing joints for each view
        """
        view_item = {}
        for view in self.views:
            view_item[view] = self.idx2joint2d(index, view)
        return view_item

    def idx2image(self, index, view=None):
        """Function for getting images from all views given an index
        Args:
            index (int): index in dataset to get images from
            view (str/None): View of images to return, if None images are returned for all views
        returns:
            view_item (Dict): Dict containing an image for each view
        """
        if view is None:
            # Case where view is not given, so all views are returned
            view_item = {}
            for view in self.views:
                view_item[view] = self.idx2image(index, view)
            return view_item

        else:
            # Case where view is given (base case)
            sync_index = self.index_map[index]
            image = self.get_images(sync_index, view)[0]
            return image

    def idx2calibration(self, index, view=None):
        """Function for getting camera calibrations from all views given an index
        Args:
            index (int): index in dataset to get camera calibrations from
            view (str/None): View of image path to return, if None paths are returned for all views
        returns:
            view_item (Dict): Dict containing camera calibrations for each view
        """
        if view is None:
            # Case where view is not given, so all views are returned
            view_item = {}
            for view in self.views:
                view_item[view] = self.idx2calibration(index, view)
            return view_item
        else:
            # Case where view is given
            sync_index = self.index_map[index]
            measure = self.measurements[sync_index.measurement_idx]
            return measure.video[view]["camera"].camera_to_dict()

    def idx2image_path(self, index, view=None):
        """Function for getting all image paths from all views given an index
        Args:
            index (int): index in dataset to get image paths from
            view (str/None): View of image path to return, if None paths are returned for all views
        returns:
            view_item (Dict): Dict containing an imagepath for each view
        """
        if view is None:
            # Case where view is not given, so all views are returned
            view_item = {}
            for view in self.views:
                view_item[view] = self.idx2image_path(index, view)
            return view_item
        else:
            # Case where view is given
            sync_index = self.index_map[index]
            measure = self.measurements[sync_index.measurement_idx]
            if measure.video[view]["frame_from_video"]:
                # Case where frame source is a video
                return list(measure.video[view]["path"])
            else:
                # Case where frame source is an image
                return list(measure.video[view]["path"][sync_index.video_idx])

    def idx2img_timestamps(self, index, view):
        """Returns the frame timings of an image in view of a datapoint given index

        Args:
            index (int): index in dataset to get image timestamps from
            view (str): View of image timestamps to return

        Returns:
            timestamps_array (np.array): Array of timestamps for frames in view
        """
        sync_index = self.index_map[index]
        timestamps_array = np.array(
            self.measurements[sync_index.measurement_idx][view]["video"]["timestamps"]
        )
        return timestamps_array

    def idx2name(self, index):
        """Returns the name of the directory a measurement can be found in given index

        Args:
            index (int): index in dataset to get name from

        Returns:
            name (str): name of directory
        """
        sync_index = self.index_map[index]
        return [self.measurements[sync_index.measurement_idx].metadata["dir_name"]]

    def idx2img_name(self, index, view=None):
        """Function for getting images names for all views or specific view given
        Args:
            index (int): index in dataset to get image name from
            view (str/None): View of image name to return, if None, image names are returned for all views

        returns:
            view_item (Dict): Dict containing an image name for each view
        """
        if view is None:
            # Case where view is not given, so all views are returned
            view_item = {}
            for view in self.views:
                view_item[view] = [self.idx2img_name(index, view)]
            return view_item

        else:
            # Case where view is given
            img_path = self.idx2image_path(index, view)[0]
            return os.path.normpath(img_path).split(os.path.sep)[-1]

    def get_idx_from_timestamp(self, marker_framerate, ts):
        """
        Get the corresponding Qualisys marker idx from timestamp. Multiplies
        the marker framerate by the timestamp and rounds to integer.
        Args:
            marker_framerate (float): Framerate of the 3D marker data
            ts (float): Timestamp to get index for

        Returns:
            int: Index of the marker data corresponding to the timestamp
        """
        return np.rint(marker_framerate * ts).astype(int)

    def is_image(self, path):
        """Check if path links to an images

        Args:
            path (str): Path to file to check if image

        Returns:
            bool: True if path links to an image, False otherwise
        """
        file_type = os.path.splitext(str(path[0]))[1]
        image_formats = [".gif", ".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png"]
        if file_type in image_formats:
            return True
        else:
            return False

    def get_image_paths(self, sync_index, view):
        """Get path of specific image, specified by sync_index.
        In case of sequence of images is indexed, returns list of paths.
        In case origin of frames are a video, a list containing a single path is returned.
        """
        measure = self.measurements[sync_index.measurement_idx]
        frames = measure.video[view]["path"]
        if measure.video[view]["frame_from_video"]:
            # Case where frames are from a video (return single path)
            return frames
        else:
            # Case where frames are images (return all paths)
            return [frames[i] for i in sync_index.video_idx]

    def get_video_frame(self, path, frames):
        """Function for returning frames specified in by frames from a video specified in path
        Args:
            path (str): Path to video
            frames (list of ints): List of frames indexes to return

        Returns:
            found_frames (list of np.arrays): List of images of shape (N, H, W, C)
        """
        vr = decord.VideoReader(path)
        found_frames = []
        for frame in frames:
            # Get frames with the accurate method (used when directly indexing video)
            found_frames.append(vr[frame].asnumpy())
        return found_frames

    def get_3d_joints(self, sync_index, convert_3d_points=None, squeeze=False):
        """Get 3d joints.

        Args:
            measurement_idx (int): Measurement idx.
            marker_idxs (int or list of ints): Marker indices.
            convert_3d_points (callable): Should return converted 3d points.

        Returns:
            (np.array): 3d joints of shape (num_joints, 3) for a single marker
                index and (num_joints, 3, num_indices) for a list of indices.
        """
        data = self.measurements[sync_index.measurement_idx].joints_3d["data_points"]
        if squeeze:
            joints = data[sync_index.joints_3d_idx, :, 0:3].squeeze()
        else:
            joints = data[sync_index.joints_3d_idx, :, 0:3]

        if convert_3d_points:
            joints = convert_3d_points(joints)
        return joints

    def get_images(self, sync_index, view, rotate=True):
        """Get images.

        Args:
            measurement_idx (int): Measurement index.
            image_idxs (int or list of ints): Image indices.
            view (string): Camera view.

        Returns:
            (np.array): Images with shape (num_images, w, h, c) in rgb format.
                When a single index is given, a single image is returned with
                shape (w, h, c).

        """
        measure = self.measurements[sync_index.measurement_idx]
        if "rotated" in view:
            base_view = view.split("_")[0]
            im_paths = self.get_image_paths(sync_index, base_view)
        else:
            im_paths = self.get_image_paths(sync_index, view)

        if measure.video[view]["frame_from_video"]:
            # Case where frames are from a video
            images = self.get_video_frame(im_paths[0], sync_index.video_idx)
        else:
            images = []
            # Case where frames are images
            for im_path in im_paths:
                im = cv2.imread(im_path)
                im_rgb = im[:, :, ::-1].copy()  # Convert from BGR to RGB
                images.append(im_rgb)

        images = np.array(images)
        # Rotate images to 0 degrees
        if rotate:
            rot = measure.video[view]["numtimesrot90clockwise"]
            images = np.rot90(
                images, -rot, axes=(1, 2)
            ).copy()  # Note np.rot90 rotates counterclockwise so - is added
        return images, im_paths

    def __len__(self):
        """Function for returning the length of the dataset"""
        if self.sample_level == "frame":
            return len(self.index_map)

        elif self.sample_level == "video":
            return len(self.video_idx2frame_idx)

        else:
            raise "Argument sample_level must be either 'frame' or 'video'. Was :" + str(
                self.sample_level
            )

    def sample_uniform(self, measurements, index):
        """Function for sampling a uniform random frame from a measurement
        Args:
            measurements (Dict of Measurement): Measurements to sample from
            index (int): Index of the measurement to sample from

        Returns:
            frame_idx (int): Index of the frame sampled
        """

        max_val = len(measurements)
        # Uniform random sample
        if self.validation_dataset:
            # Use the same random seed for validation dataset
            np.random.seed(index)
        choice = np.random.choice(max_val)
        # Return the frame index
        return measurements[choice]

    def video_idx2measurements(self, seq_idx):
        """Function returns the measurements corresponding to the video index
        Args:
            seq_idx (int): The index of the measurement chosen

        Returns:
            measurements (Dict of Measurement)
        """
        return self.video_idx2frame_idx[seq_idx]

    def __getitem__(self, index):
        """Function for returning all items specified as true in return_dict.
            Default value is iterating over self.batch_return_vals to get attributes, but in cases where
            the attribute is itself a Dict (for example a subclass of source), the function is recursively called

        Args:
            index (int): index in dataset to get the specified values from.
            return_dict (dict / None): The nested dictonary containing the attributes to return if value is true.
                if None, the default dict self.batch_return_vals is used.
            key2func (dict / None): The nested dictonary containing the functions to get attributes from.
                if None, the default dict self.key2func is used.
        Returns:
            item (dict): Dictonary containing all values for index specified as true in the return_dict.
        """

        def loop_get_item(index, return_dict, key2func, key2name):
            """Function for returning all items specified as true in return_dict.
                Default value is iterating over self.batch_return_vals to get attributes, but in cases where
                the attribute is itself a Dict (for example a subclass of source), the function is recursively called

            Args:
                index (int): index in dataset to get the specified values from.
                return_dict (dict / None): The nested dictonary containing the attributes to return if value is true.
                    if None, the default dict self.batch_return_vals is used.
                key2func (dict / None): The nested dictonary containing the functions to get attributes from.
                    if None, the default dict self.key2func is used.
            Returns:
                item (dict): Dictonary containing all values for index specified as true in the return_dict.
            """
            item = {}  # Dict containing results
            for key in return_dict.keys():
                dataset_attribute = return_dict[key]
                if dataset_attribute is True:
                    # Case where attribute should be returned
                    try:
                        item[key] = key2func[key](index)
                    except KeyError:
                        # Check if name is in attribute list
                        try:
                            item[key] = key2name[key]

                        except KeyError:
                            raise KeyError(
                                "Attribute: '"
                                + key
                                + "' to return was not found in dataset. "
                                "Check if the attribute name is spelled correctly"
                            )

                elif isinstance(dataset_attribute, dict):
                    # Case where a dictionary (sub-category) is encountered
                    # Check if key is view
                    if key == "view":
                        for view in self.views:
                            item[view] = loop_get_item(
                                index, dataset_attribute, key2func["view"], key2name[view]
                            )
                    else:
                        item[key] = loop_get_item(
                            index, dataset_attribute, key2func[key], key2name[key]
                        )

            return item


        if self.sample_level == "video":
            measurements = self.video_idx2measurements(index)
            # Convert (video_idx, frame_idx) to video idx
            index = self.sample_method(measurements, index)

        # Set default dictionary to get keys from
        return_dict = self.batch_return_vals
        key2func = self.key2func
        sync_index = self.index_map[index]
        measure = self.measurements[sync_index.measurement_idx]
        key2name = measure.name_dict
        item = loop_get_item(
            index, return_dict=return_dict, key2func=key2func, key2name=key2name
        )

        # Apply transforms to any video or 2d joints using albumentations
        if self.video_transform is True and self.joints_2d_exists is True:
            for view in self.views:
                if item["video"]["image"][view].shape[0] > 1:
                    raise NotImplementedError("Transformations for temporal video is not implemented")
                transform = self.transform(
                    image=item["video"]["image"][view][0], keypoints=item["joints_2d"][view][0]
                )
                item["video"]["image"][view] = transform["image"][None, ...]
                item["joints_2d"][view] = torch.tensor(transform["keypoints"])[None, ...]
        else:
            item = self.transform(item)
        return item

if __name__ == "__main__":
    datapath = "/work3/ckin/datasets/SportsPose/SportsPose"
    print("Testing SportsPoseDataset")
    dataset_no_im = SportsPoseDataset(
        data_dir=datapath,
        sample_level="video", 
        return_preset = {
            "joints_2d": True,
            "joints_3d": {
                "data_points": True,
            },
            "metadata": {
                "file_name": True,
            },
            "video": {"view": {"camera": False, "img_dims": True}, "image": False, "calibration": True}
        },
        seq_size=243
    )
    test_sample = dataset_no_im[0]
    print(test_sample.keys())

    # Check if 2D joints still is available with the camera mod
    print(test_sample["joints_2d"]["right"].shape)

    # Check if it works in a pytorch dataloader as intented
    dataloader = torch.utils.data.DataLoader(dataset_no_im, batch_size=3, shuffle=True)
    for batch in dataloader:
        print(batch["video"]["right"]["img_dims"].shape)
        print(batch["video"]["right"]["img_dims"])

