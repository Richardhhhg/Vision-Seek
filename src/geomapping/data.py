from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from detection.data import DetectionOutput


class GeoMappingInput(BaseModel):
    """
    Input data for the GeoMapping Module.
    
    Attributes:
    - detection_output (DetectionOutput): The output of the detection module, which includes the bounding box coordinates of the detected objects in the video.
    - srt_file_path (str): The path to the SRT file that contains the telemetry data for the video.
    - telemetry_file_path (str): The path to the telemetry file that contains additional telemetry data for the video.
    - video_file_path (str): The path to the video file that was processed by the detection module.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_output: DetectionOutput
    srt_file_path: str
    telemetry_file_path: str
    video_file_path: str

class GeoMappingOutput(BaseModel):
    """
    Returns all the geotagged objects in geojson format. This is the final output of the geomapping module.

    Attributes:
    - gps_coords (np.ndarray): Array of shape (N_unique, 4, 2) holding (lat, lon) for
      the four corners of every unique physical object detected in the video, ordered
      [top-left, top-right, bottom-right, bottom-left]. Where N_unique is the number of unique physical objects detected in the video.
    - geojson_path (str): Path to the saved GeoJSON FeatureCollection on disk.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gps_coords: np.ndarray
    geojson_path: str

class TelemetryInput(BaseModel):
    """
    Takes in raw inputs from videos, srt, and other telemetry data.

    Attributes:
    - srt_file_path (str): The path to the SRT file that contains the telemetry data for the video.
    - telemetry_file_path (str): The path to the telemetry file that contains additional telemetry data for the video.

    Some other things to consider (in the future):
    - SRT and telemetry data may not always have the same data. Ie. telemetry may contain gimbal data or this may be found in SRT.
    - Frame rate of video may not match the srt and telemetry data intervals. Currently assuming 30 fps with test data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    srt_file_path: str
    telemetry_file_path: str

class TelemetryOutput(BaseModel):
    """
    Converts all drone telemetry and stuff data into usable format for the rest of the codebase.

    Attributes:
    ###############################################
    # General Metadata about Camera and Detection #
    ###############################################
    - image_width (int): The width of the image in pixels.
    - image_height (int): The height of the image in pixels.
    - camera_type (str): The type of camera used to capture the video. Monocular or stereo (currently only monocular is supported).
    - horizontal_fov (float | None): The horizontal field of view of the camera in degrees. Optional if diagonal_fov is provided. In degrees
    - vertical_fov (float | None): The vertical field of view of the camera in degrees. Optional if diagonal_fov is provided. In degrees
    - diagonal_fov (float | None): The diagonal field of view of the camera in degrees. Optional if horizontal_fov and vertical_fov are provided. In degrees
    - aspect_ratio (float | None): The aspect ratio of the camera (width / height). Optional if image dimensions can be used instead.

    #############################
    # Frame Specific Parameters #
    #############################
    TODO: Should probably support altitude in the future. This would also require the altitude the drone starts at (on ground)
    - longitude (np.ndarray): Camera longitude per frame, shape (F,).
    - latitude (np.ndarray): Camera latitude per frame, shape (F,).
    - height_above_ground (np.ndarray): Camera height above ground per frame, shape (F,).
    - gimbal_pitch (np.ndarray): Gimbal pitch in degrees per frame, shape (F,).
    - gimbal_yaw (np.ndarray): Gimbal yaw in degrees per frame, shape (F,).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_width: int
    image_height: int
    camera_type: str

    longitude: np.ndarray
    latitude: np.ndarray
    height_above_ground: np.ndarray
    gimbal_pitch: np.ndarray
    gimbal_yaw: np.ndarray

    diagonal_fov: float | None = None
    vertical_fov: float | None = None
    horizontal_fov: float | None = None
    aspect_ratio: float | None = None

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
    TODO: Should probably support altitude in the future. This would also require the altitude the drone starts at (on ground)
    - longitude (np.ndarray): Camera longitude per frame, shape (F,).
    - latitude (np.ndarray): Camera latitude per frame, shape (F,).
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
    """
    GPS frames to deduplicate. 

    Attributes:
    - gps_coords (list[np.ndarray]): One entry per frame. Each array has shape (N_f, 4, 2)
      holding (lat, lon) for the four corners of every detection in that frame, ordered
      [top-left, top-right, bottom-right, bottom-left]. Where N_f is the number of detections in frame f.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gps_coords: list[np.ndarray]

class DeduplicationOutput(BaseModel):
    """
    Deduplicated gps objects. Detections that map to the same physical object across
    the video have been collapsed into a single averaged bounding box, so the frame
    axis no longer exists.

    Attributes:
    - gps_coords (np.ndarray): Array of shape (N_unique, 4, 2) holding (lat, lon) for
      the four corners of every unique physical object detected in the video, ordered
      [top-left, top-right, bottom-right, bottom-left]. Where N_unique is the number of unique physical objects detected in the video.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gps_coords: np.ndarray
