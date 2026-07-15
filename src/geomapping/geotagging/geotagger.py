import logging

import numpy as np
from pyproj import Geod

from geomapping.data import GeoTaggingInput, GeoTaggingOutput

logger = logging.getLogger("geotag")


class GeoTagger:
    """
    Does all the math for converting the detected bounding boxes into real world gps coordinates.

    Vectorization contract:
    - All bounding-box math is fully vectorized. Detections across all frames are flattened
      to 1D arrays of length ``M = 4 * total_detections`` (4 corners per bbox) for the
      geometry pipeline, then reshaped back to a per-frame ``list[np.ndarray]`` at the end.
    - Frame-level metadata (``latitude``, ``longitude``, ``height_above_ground``,
      ``gimbal_pitch``, ``gimbal_yaw``) is expected to be a 1D array of length ``F``
      (one entry per frame) and is broadcast to the per-corner axis via ``frame_ids``.
    - Camera-level metadata (``image_width``, ``image_height``, FOVs, ``aspect_ratio``)
      remains scalar because it is a property of the camera, not of individual frames.

    Bounding box corner ordering used everywhere below (clockwise starting at top-left):
        0: (x1, y1)   top-left
        1: (x2, y1)   top-right
        2: (x2, y2)   bottom-right
        3: (x1, y2)   bottom-left

    Methods:
    - invoke (GeoTaggingInput) -> GeoTaggingOutput: Takes in data and outputs the gps coordinates of the detected objects.

    Private Methods:
    - _get_aspect_ratio (GeoTaggingInput) -> float: Gets aspect ratio from the input data.
    - _convert_fov (GeoTaggingInput) -> (float, float): Takes in fov (diagonal, horizontal, vertical) and converts it into horizontal and vertical fov (degrees).
    - _extract_bbox_corners (GeoTaggingInput) -> (bbox_x, bbox_y, frame_ids, detections_per_frame): Flattens per-frame xyxy detections into 1D arrays of pixel corners for vectorized math.
    - _get_angle_y (bbox_y, gimbal_pitch, image_height, vfov) -> np.ndarray: Vertical angle from camera to each pixel (alpha), in degrees.
    - _get_angle_x (bbox_x, image_width, hfov) -> np.ndarray: Horizontal angle from camera to each pixel (beta), in degrees.
    - _get_azimuth (gimbal_yaw, angle_x) -> np.ndarray: Compass azimuth from the nadir to each pixel, in degrees.
    - _get_distance (distance_xs, distance_ys) -> np.ndarray: Pythagorean distance from nadir to each pixel.
    - _get_distance_x (distance_y, angle_x) -> np.ndarray: Ground x-distance from nadir to each pixel.
    - _get_distance_y (height, angle_y) -> np.ndarray: Ground y-distance from nadir to each pixel.
    - _get_gps_coord_monocular (GeoTaggingInput) -> list[np.ndarray]: Runs the full vectorized monocular pipeline and returns per-frame (N_f, 4, 2) arrays of (lat, lon) corners.
    """
    _GEOD = Geod(ellps="WGS84")

    def __init__(self):
        pass

    def invoke(self, input_data: GeoTaggingInput) -> GeoTaggingOutput:
        camera_type = input_data.camera_type
        if camera_type == "monocular":
            gps_coords = self._get_gps_coord_monocular(input_data)
            return GeoTaggingOutput(gps_coords=gps_coords)
        raise NotImplementedError(f"Camera type {camera_type} not supported yet.")

    def _get_aspect_ratio(self, data: GeoTaggingInput) -> float:
        if data.aspect_ratio is not None:
            return float(data.aspect_ratio)
        return float(data.image_width) / float(data.image_height)

    def _convert_fov(self, data: GeoTaggingInput) -> tuple[float, float]:
        horizontal_fov = data.horizontal_fov
        vertical_fov = data.vertical_fov
        diagonal_fov = data.diagonal_fov
        if horizontal_fov is not None and vertical_fov is not None:
            logger.info("Using provided horizontal and vertical FOVs.")
            return float(horizontal_fov), float(vertical_fov)
        if diagonal_fov is not None:
            logger.info("Converting diagonal FOV to horizontal and vertical FOVs.")
            aspect_ratio = self._get_aspect_ratio(data)
            d_half = np.radians(diagonal_fov) / 2.0
            denom = np.sqrt(1.0 + aspect_ratio**2)
            h_rad = 2.0 * np.arctan(np.tan(d_half) * aspect_ratio / denom)
            v_rad = 2.0 * np.arctan(np.tan(d_half) / denom)
            return float(np.degrees(h_rad)), float(np.degrees(v_rad))
        logger.error(
            "Insufficient FOV information provided. Please provide either horizontal and vertical FOVs or diagonal FOV."
        )
        raise ValueError(
            "Insufficient FOV information provided. Please provide either horizontal and vertical FOVs or diagonal FOV."
        )

    def _extract_bbox_corners(
        self, data: GeoTaggingInput
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Flatten per-frame xyxy detections into 1D arrays for vectorized processing.

        Each detection contributes 4 corners in the order [top_l, top_r, bot_r, bot_l].

        Returns:
        - bbox_x (np.ndarray): shape (M,) x-pixel coord of each corner.
        - bbox_y (np.ndarray): shape (M,) y-pixel coord of each corner.
        - frame_ids (np.ndarray): shape (M,) frame index each corner belongs to.
        - detections_per_frame (np.ndarray): shape (F,) number of detections in each frame.

        M = 4 * sum(detections_per_frame).
        """
        frames = data.detection_output.annotated_frames
        detections_per_frame = np.array(
            [len(np.asarray(frame.xyxy)) for frame in frames], dtype=np.int64
        )

        bbox_x_parts: list[np.ndarray] = []
        bbox_y_parts: list[np.ndarray] = []
        frame_id_parts: list[np.ndarray] = []
        for f_idx, frame in enumerate(frames):
            xyxy = np.asarray(frame.xyxy, dtype=np.float64)
            if xyxy.size == 0:
                continue
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            corners_x = np.stack([x1, x2, x2, x1], axis=1)
            corners_y = np.stack([y1, y1, y2, y2], axis=1)
            bbox_x_parts.append(corners_x.reshape(-1))
            bbox_y_parts.append(corners_y.reshape(-1))
            frame_id_parts.append(np.full(corners_x.size, f_idx, dtype=np.int64))

        if not bbox_x_parts:
            empty_f = np.empty((0,), dtype=np.float64)
            empty_i = np.empty((0,), dtype=np.int64)
            return empty_f, empty_f, empty_i, detections_per_frame

        bbox_x = np.concatenate(bbox_x_parts)
        bbox_y = np.concatenate(bbox_y_parts)
        frame_ids = np.concatenate(frame_id_parts)
        return bbox_x, bbox_y, frame_ids, detections_per_frame

    def _get_angle_y(
        self,
        bbox_y: np.ndarray,
        gimbal_pitch: np.ndarray,
        image_height: int,
        vfov: float,
    ) -> np.ndarray:
        phis = 90.0 - gimbal_pitch
        taus = (bbox_y / image_height) * vfov
        # NOTE: the math here is simplified to only assume that the object is in front of nadir
        # TODO: implement logic for calculating angles if object is behind nadir
        return phis + vfov / 2.0 - taus

    def _get_angle_x(
        self, bbox_x: np.ndarray, image_width: int, hfov: float
    ) -> np.ndarray:
        sigmas = (bbox_x / image_width) * hfov
        # NOTE: this only assumes that the object is to the right of the nadir
        # TODO: implement logic for calculating angles if object is to the left of nadir
        return sigmas - hfov / 2.0

    def _get_azimuth(
        self, gimbal_yaw: np.ndarray, angle_x: np.ndarray
    ) -> np.ndarray:
        # NOTE: this only assumes that the object is to the right of the nadir
        # TODO: implement logic for calculating azimuth if object is to the left of nadir
        return gimbal_yaw + angle_x

    def _get_distance(
        self, distance_xs: np.ndarray, distance_ys: np.ndarray
    ) -> np.ndarray:
        return np.sqrt(distance_xs**2 + distance_ys**2)

    def _get_distance_x(
        self, distance_y: np.ndarray, angle_x: np.ndarray
    ) -> np.ndarray:
        return distance_y * np.tan(np.radians(angle_x))

    def _get_distance_y(
        self, height: np.ndarray, angle_y: np.ndarray
    ) -> np.ndarray:
        return height * np.tan(np.radians(angle_y))

    def _get_gps_coord_monocular(self, data: GeoTaggingInput) -> list[np.ndarray]:
        """
        Getting the gps coordinates of pixels for a monocular camera.

        Returns:
        - list[np.ndarray]: one entry per frame, each of shape (N_f, 4, 2) holding
          (lat, lon) for the four bbox corners of every detection in that frame.
        """
        horizontal_fov, vertical_fov = self._convert_fov(data)

        bbox_x, bbox_y, frame_ids, detections_per_frame = self._extract_bbox_corners(data)

        gimbal_pitch = np.asarray(data.gimbal_pitch, dtype=np.float64)[frame_ids]
        gimbal_yaw = np.asarray(data.gimbal_yaw, dtype=np.float64)[frame_ids]
        heights = np.asarray(data.height_above_ground, dtype=np.float64)[frame_ids]
        latitudes = np.asarray(data.latitude, dtype=np.float64)[frame_ids]
        longitudes = np.asarray(data.longitude, dtype=np.float64)[frame_ids]

        angle_ys = self._get_angle_y(bbox_y, gimbal_pitch, data.image_height, vertical_fov)
        angle_xs = self._get_angle_x(bbox_x, data.image_width, horizontal_fov)
        distance_ys = self._get_distance_y(heights, angle_ys)
        distance_xs = self._get_distance_x(distance_ys, angle_xs)
        distances = self._get_distance(distance_xs, distance_ys)
        azimuths = self._get_azimuth(gimbal_yaw, angle_xs)

        # pyproj.Geod.fwd is fully vectorized: it accepts arrays and returns
        # (new_lons, new_lats, back_azimuths) each of the same shape.
        new_lons, new_lats, _ = self._GEOD.fwd(longitudes, latitudes, azimuths, distances)

        # Distances and azimuths follow the same shape as the output: a flat 1D array
        # of length 4 * total_detections that we split back into a per-frame list of
        # (N_f, 4, 2) arrays. Corner order is [tl, tr, br, bl]. For drawing the
        # bounding box on a GIS map, the drawing module should repeat the first
        # coordinate as a 5th position to close the polygon.
        gps_flat = np.stack([new_lats, new_lons], axis=-1)
        split_sizes = detections_per_frame * 4
        split_indices = np.cumsum(split_sizes)[:-1]
        per_frame_flat = np.split(gps_flat, split_indices) if split_indices.size else [gps_flat]
        gps_coords = [
            arr.reshape(int(n), 4, 2)
            for arr, n in zip(per_frame_flat, detections_per_frame)
        ]
        return gps_coords
