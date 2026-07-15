from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from detection.data import DetectionOutput


class GeoMappingInput(BaseModel):
    pass

class GeoMappingOutput(BaseModel):
    pass

class SRTInput(BaseModel):
    pass

class SRTOutput(BaseModel):
    """
    Output of the SRT processing. This extracts all the drone data.
    """
    pass

class GeoTaggingInput(BaseModel):
    """
    Data that goes into the geo tagging module. This includes the necessary parameters needed to perform the mapping computation as well as the bounding box coordinates of the object in the image.

    Frame-specific attributes are 1D arrays of length F (one entry per frame, aligned with
    ``detection_output.annotated_frames``). Camera-level attributes are scalars.

    Attributes:
    ###############################################
    # General Metadata about Camera and Detection #
    ###############################################
    - detection_output (DetectionOutput): The output of the detection module, which includes the bounding box coordinates of the detected objects in the video.
    - camera_type (str): The type of camera used to capture the video. Monocular or stereo (currently only monocular is supported).
    - image_width (int): The width of the image in pixels.
    - image_height (int): The height of the image in pixels.
    - horizontal_fov (float | None): The horizontal field of view of the camera in degrees. Optional if diagonal_fov is provided.
    - vertical_fov (float | None): The vertical field of view of the camera in degrees. Optional if diagonal_fov is provided.
    - diagonal_fov (float | None): The diagonal field of view of the camera in degrees. Optional if horizontal_fov and vertical_fov are provided.
    - aspect_ratio (float | None): The aspect ratio of the camera (width / height). Optional if image dimensions can be used instead.

    #############################
    # Frame Specific Parameters #
    #############################
    - longitude (np.ndarray): Camera longitude per frame, shape (F,).
    - latitude (np.ndarray): Camera latitude per frame, shape (F,).
    - altitude (np.ndarray): Camera altitude (MSL) per frame, shape (F,).
    - height_above_ground (np.ndarray): Camera height above ground per frame, shape (F,).
    - gimbal_pitch (np.ndarray): Gimbal pitch in degrees per frame, shape (F,).
    - gimbal_yaw (np.ndarray): Gimbal yaw in degrees per frame, shape (F,).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_output: DetectionOutput
    camera_type: str
    image_width: int
    image_height: int

    longitude: np.ndarray
    latitude: np.ndarray
    altitude: np.ndarray
    height_above_ground: np.ndarray
    gimbal_pitch: np.ndarray
    gimbal_yaw: np.ndarray

    horizontal_fov: float | None = None
    vertical_fov: float | None = None
    diagonal_fov: float | None = None
    aspect_ratio: float | None = None

class GeoTaggingOutput(BaseModel):
    """
    Output of the geo tagging module.

    Attributes:
    - gps_coords (list[np.ndarray]): One entry per frame. Each array has shape (N_f, 4, 2)
      holding (lat, lon) for the four corners of every detection in that frame, ordered
      [top-left, top-right, bottom-right, bottom-left].
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gps_coords: list[np.ndarray]

class DeduplicationInput(BaseModel):
    pass

class DeduplicationOutput(BaseModel):
    pass
