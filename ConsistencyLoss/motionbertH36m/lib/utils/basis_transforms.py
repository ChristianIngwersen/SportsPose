import cv2
import numpy as np
import torch
from torch import Tensor

from typing import Literal, Optional, Union, NewType, Sequence
FrameDecimal = NewType("FrameDecimal", np.ndarray)
"""Represents a numpy array of RGB frames in the range [0; 1]. Shape (C, H, W)"""
FramesDecimal = NewType("FramesDecimal", np.ndarray)
"""Represents a numpy array of RGB frames in the range [0; 1]. Shape (B, C, H, W)"""


def center_crop(image: FrameDecimal, crop_size: tuple[int, int]) -> FrameDecimal:
    """
    Center crops a numpy array.

    Args:
        image (FrameDecimal): The image to be cropped. (C, H, W)
        crop_size (tuple): The desired size of the crop. (H, W)

    Returns:
        cropped_img (FrameDecimal): The cropped image.
    """
    _, height, width = image.shape
    desired_height, desired_width = crop_size
    if height < desired_height or width < desired_width:
        # Case where crop size is bigger than image size, so image is zero padded at booth sides.
        pad_h = np.max([desired_height - height, 0])
        pad_w = np.max([desired_width - width, 0])
        image = np.pad(image, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), 'constant')
        _, height, width = image.shape
    x1 = np.round((width - desired_width) / 2.).astype(np.int32)
    y1 = np.round((height - desired_height) / 2.).astype(np.int32)
    cropped_img = image[..., y1:(y1 + desired_height), x1:(x1 + desired_width)]
    return cropped_img


def transform_func(image: FrameDecimal, size: tuple[int, int]) -> np.ndarray:
    """ Function for transforming image using only cv2 transforms and return as numpy array
    The image is resized while keeping the aspect ratio, and padded with zeros to the wanted size.
    This means no part of the image is cropped out.

    :param image: Image to be transformed (C, H, W)
    :param size: Wanted size of image (H, W)
    :return: Transformed image (C, H, W) as numpy array
    """

    _, height, width = image.shape
    height, width = resize_size((height, width), min(size))
    image = cv2.resize(image.transpose(1, 2, 0), dsize=(width, height)).transpose(2, 0, 1)
    image = center_crop(image, size)  # This will only pad the image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image -= mean[:, None, None]
    image /= std[:, None, None]

    return FrameDecimal(image)


def crop_frames(frames: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
    """ Function for cropping frames given a bounding box. Handles both batched and unbatched frames.

    No checks are done to ensure that the bounding box is within the frame - this is merely a small utility function.

    Args:
        frames (np.ndarray): Frames to be cropped (B?, C?, H, W)
        bounding_box (np.ndarray): Bounding box to crop in format (x1, y1, x2, y2)

    Returns:
        cropped_frames (np.ndarray): Frames cropped to the specified bounding box. Of shape
            (B, C, H, W, C) if frames is (B, C, H, W) and (C, H, W) if frames is (C, H, W)
    """
    return frames[..., bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]


def resize_size(image_size: tuple[int, int], new_long: int) -> tuple[int, int]:
    """Calculates the new dimensions of a resized image such as to preserve the aspect ratio given the original
    dimensions and the length of the longest edge of the resized image.

    :param image_size: The original size of the image. The biggest dimension must not be 0.
    :param new_long: The length of the longest edge of the resized image.

    :return: The new dimensions of the resized image.
     """
    height, width = image_size
    short, long = (width, height) if width <= height else (height, width)

    new_short = np.round(new_long * short / long).astype(int)

    new_h, new_w = (new_long, new_short) if height >= width else (new_short, new_long)
    return int(new_h), int(new_w)


def cxcywh_without_cropping(bb: np.ndarray, padding_ltrb: tuple[int, int, int, int]) -> np.ndarray:
    """ Calculates the center percentage coordinate system when cropping is removed.
    :bb: The bounding box in the format center x, center y, width, height, and are normalized to the range [0, 1].
    :padding_ltrb: The padding in the format left, top, right, bottom, and are normalized to the range [0, 1].
    :return: The bounding box in the format center x, center y, width, height, and are normalized to the range [0, 1].
    """

    horizontal_padding = np.sum(padding_ltrb[::2])
    vertical_padding = np.sum(padding_ltrb[1::2])

    bb[..., :2] -= padding_ltrb[:2]  # Move center by padding
    # Unpadded image is smaller so bb is proportionally larger. Scale up dimensions accordingly
    bb /= np.tile((1 - horizontal_padding, 1 - vertical_padding), reps=2)
    return bb


def pad_to_size(wanted_size: int, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Pads the input image to the given square size using zero padding while keeping the aspect ratio.
    The image is padded such that the padding is equal on all sides, with ties broken by padding the right and bottom.
    The used padding is also returned.

    :param wanted_size: The size to pad the image to (square size).
    :param img: The image to pad. Should be of shape (3, height, width).
    :return: The padded image, and the padding in the format left, top, right, bottom, and are normalized to the range [0, 1].
    """

    # Resize the image to the wanted size
    resize_shape = resize_size((img.shape[-2], img.shape[-1]), new_long=wanted_size)
    img = cv2.resize(img.transpose((2, 1, 0)), dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
    # Compute padding needed to make the image the wanted size
    image_width, image_height = img.shape[:2]

    crop_height, crop_width = wanted_size, wanted_size

    # small side
    width_pad_small = (crop_width - image_width) // 2 * (crop_width > image_width)
    height_pad_small = (crop_height - image_height) // 2 * (crop_height > image_height)

    # large side
    width_pad_large = (crop_width - image_width + 1) // 2 * (crop_width > image_width)
    height_pad_large = (crop_height - image_height + 1) // 2 * (crop_height > image_height)
    padding_ltrb = [width_pad_small, height_pad_small, width_pad_large, height_pad_large]

    height_padding = (height_pad_small, height_pad_large)
    width_padding = (width_pad_small, width_pad_large)
    channel_padding = (0, 0)
    # zero-pad the image
    img = np.pad(img, (width_padding, height_padding, channel_padding))
    img = img.transpose(2, 1, 0)
    # Convert padding from pixel to percentage coordinate system
    padding_ltrb = (padding_ltrb[0]/wanted_size,
                    padding_ltrb[1]/wanted_size,
                    padding_ltrb[2]/wanted_size,
                    padding_ltrb[3]/wanted_size)
    return img, padding_ltrb


def max_separated_xy_heatmap(heatmap_x: np.ndarray, heatmap_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from x and y dimension seperated heatmap (simcc representations).
    Args:
        heatmap_x (np.ndarray): x-axis SimCC in shape (J, W) or (B, J, W)
        heatmap_y (np.ndarray): y-axis SimCC in shape (J, H) or (B, J, H)

    Returns:
        tuple: locs (np.ndarray): locations of maximum heatmap responses in shape (J, 2) or (B, J, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape (J,) or (B, J)
    """
    N, K = heatmap_x.shape[:2]
    heatmap_x = heatmap_x.reshape(N * K, -1)
    heatmap_y = heatmap_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(heatmap_x, axis=1)
    y_locs = np.argmax(heatmap_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(heatmap_x, axis=1)
    max_val_y = np.amax(heatmap_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def rescale_pose(joints: np.ndarray, crop_bounding_box: np.ndarray, model_input_resolution: np.ndarray) -> np.ndarray:
    """Scales predicted joint positions to original image resolution for a constant crop.

    The predicted joint positions calculated from heatmap outputs from pose model are rescaled to match the location and
    resolution of the original image. A scale factor is calculated based on the size of the bounding box image segment
    input into the HRNet predict functions. A translation compensation is calculated based on aspect ratio differences
    between input image shape and pose model input shape.

    Args:
        joints (np.ndarray): Array of "selectjoints"-format with shape (N-frames, N-joints, 2)
        crop_bounding_box (np.ndarray): Array of shape (4,) in top-left, bottom-right format
        model_input_resolution (np.ndarray): Array of shape (2,) with the resolution of the input image to the model

    Returns:
        A new array of joints with each joint's positions scaled to match original image resolution
    """
    crop_resolution = (crop_bounding_box[2:] - crop_bounding_box[:2])[::-1]
    resize_scale = (crop_resolution / model_input_resolution).max()  # max scale that keeps aspect ratio
    min_scale_index = (crop_resolution / model_input_resolution).argmin()
    predict_translation = (model_input_resolution[min_scale_index] * resize_scale - crop_resolution[
        min_scale_index]) / 2
    predicted_data = joints * model_input_resolution[::-1] * resize_scale
    # Subtract translation from padding
    predicted_data[..., min_scale_index - 1] -= predict_translation
    predicted_data += crop_bounding_box[:2]
    return predicted_data


def h36m_from_coco(coco: Tensor) -> Tensor:
    """
    COCO: {0-nose 1-Leye 2-Reye 3-Lear 4-Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}

    H36M:
    0: 'root',
    1: 'rhip',  2: 'rkne', 3: 'rank',
    4: 'lhip',  5: 'lkne', 6: 'lank',
    7: 'belly',
    8: 'neck',
    9: 'nose',
    10: 'head',
    11: 'lsho', 12: 'lelb', 13: 'lwri',
    14: 'rsho', 15: 'relb', 16: 'rwri'
    """
    # TODO: Add docstring
    if isinstance(coco, torch.Tensor):
        h36m = torch.zeros_like(coco)
    else:
        T, V, C = coco.shape
        coco = coco[None, ...]
        h36m = np.zeros([1, T, 17, C])

    h36m_arms, coco_arms = [11, 12, 13, 14, 15, 16], [5, 7, 9, 6, 8, 10]  # shoulder, elbow, wrist. Right side first
    h36m_legs, coco_legs = [1, 2, 3, 4, 5, 6], [12, 14, 16, 11, 13, 15]  # hip, knee, ankle. Right side first
    h36m_nose, coco_nose = [9], [0]
    h36m[..., h36m_arms + h36m_legs + h36m_nose, :] = coco[..., coco_arms + coco_legs + coco_nose, :]

    # center hip, neck, and head ← mean left & right hip, shoulder, and ear (respectively)
    h36m[..., [0, 8, 10], :] = coco[..., [(11, 12), (5, 6), (3, 4)], :].mean(-2)
    h36m[..., 7, :] = h36m[..., [0, 8], :].mean(-2)  # belly ← mean root (center hip) and neck

    # Get direction of neck to head
    neck_to_center_head = torch.nn.functional.normalize(h36m[..., 10, :] - h36m[..., 8, :], dim=-1)
    distance_to_eyes = (h36m[..., 10, :] - coco[..., (1, 2), :].mean(-2)).norm(dim=-1, keepdim=True)
    h36m[..., 10, :] += neck_to_center_head * distance_to_eyes * 0.8  # 80% is a number pulled out of my ___

    return h36m