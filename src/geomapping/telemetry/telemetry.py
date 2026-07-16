import logging
import re

import numpy as np
import pandas as pd

from geomapping.data import TelemetryInput, TelemetryOutput

logger = logging.getLogger("telemetry")


class TelemetryProcessor:
    """
    Class responsible for processing an SRT file to extract the necessary information for geolocation computations.

    TODO: Have a more robust support for srt and also additional drone data.
    Assumes the KABR-style formats in ``data/test_data``:
    - SRT: DJI subtitle cues with ``[latitude: ...] [longitude: ...] [altitude: ...]``
    - Telemetry CSV: columns ``frame``, ``altitude``, ``gimbal_heading(degrees)``,
      ``gimbal_pitch(degrees)``, ``gimbal_roll(degrees)``

    Array contracts for the private parsers:
    - ``_process_srt`` returns shape ``(N, 3)`` with columns
      ``[latitude, longitude, altitude]``, one row per SRT cue (frame order).
    - ``_process_telemetry`` returns shape ``(M, 5)`` with columns
      ``[frame, altitude, gimbal_heading, gimbal_pitch, gimbal_roll]``, sorted by
      ``frame`` ascending. ``gimbal_heading`` is treated as yaw.

    ``invoke`` aligns SRT rows to telemetry by matching SRT row index ``i`` to
    telemetry ``frame == i`` (0-based), which matches the test data, then packs
    the result into ``TelemetryOutput``. Camera metadata not present in either
    file (image size, camera type, FOV) uses the class defaults below, which
    match ``test_5.mp4`` / DJI Mavic 2S.

    Attributes:

    Methods:
    - invoke (TelemetryInput) -> TelemetryOutput: Processes the telemetry data and extracts the necessary information for geolocation computations.

    - _process_srt (str) -> numpy.ndarray: For processing the srt data into the format for remaining steps.
    - _process_telemetry (str) -> numpy.ndarray: For processing the telemetry data into the format for remaining steps.
    """
    # Camera metadata is not present in the SRT / telemetry files for the test
    # dataset; these defaults match test_5.mp4 (1920x1080) and TelemetryOutput's
    # existing diagonal_fov default for the DJI Mavic 2S.
    _DEFAULT_IMAGE_WIDTH = 1920
    _DEFAULT_IMAGE_HEIGHT = 1080
    _DEFAULT_CAMERA_TYPE = "monocular"
    _DEFAULT_DFOV = 88

    def __init__(self):
        pass

    def invoke(self, input: TelemetryInput) -> TelemetryOutput:
        srt_data = self._process_srt(input.srt_file_path)
        telemetry_data = self._process_telemetry(input.telemetry_file_path)

        return TelemetryOutput(
            image_width=self._DEFAULT_IMAGE_WIDTH,
            image_height=self._DEFAULT_IMAGE_HEIGHT,
            camera_type=self._DEFAULT_CAMERA_TYPE,
            diagonal_fov=self._DEFAULT_DFOV,
            latitude=srt_data[:, 0],
            longitude=srt_data[:, 1],
            height_above_ground=srt_data[:, 2],
            gimbal_yaw=telemetry_data[:, 2],
            gimbal_pitch=telemetry_data[:, 3],
        )

    def _align_frames(self, telemetry_data: np.ndarray, srt_data: np.ndarray) -> np.ndarray:
        """
        Some cursed thing that Cursor Composer 2.5 produced about aligning SRT rows and telemetry frames. I might need it for future reference so its going to go here.

        This is not used.
        """
        n_srt = srt_data.shape[0]
        frame_ids = telemetry_data[:, 0].astype(np.int64)
        if frame_ids.min() < 0 or frame_ids.max() >= n_srt:
            raise ValueError(
                f"Telemetry frame ids [{frame_ids.min()}, {frame_ids.max()}] are "
                f"out of range for {n_srt} SRT cues."
            )
        if frame_ids.shape[0] != n_srt or not np.array_equal(
            frame_ids, np.arange(n_srt, dtype=np.int64)
        ):
            # Reindex by frame so missing / out-of-order rows still align to SRT.
            tel_by_frame = np.full((n_srt, telemetry_data.shape[1]), np.nan, dtype=np.float64)
            tel_by_frame[frame_ids] = telemetry_data
            if np.isnan(tel_by_frame).any():
                missing = np.where(np.isnan(tel_by_frame[:, 0]))[0]
                raise ValueError(
                    f"Telemetry is missing frames required to align with SRT "
                    f"(first missing frame={int(missing[0])})."
                )
            return tel_by_frame
        return telemetry_data

    def _process_srt(self, srt_file_path: str) -> np.ndarray:
        """
        Parse a DJI-style SRT file into an ``(N, 3)`` array of
        ``[latitude, longitude, altitude]`` per cue, in cue order.
        """
        with open(srt_file_path, encoding="utf-8") as f:
            text = f.read()

        # Each cue body embeds bracketed key/value pairs. Latitude/longitude/
        # altitude appear together on the telemetry line.
        matches = re.findall(
            r"\[latitude:\s*([-\d.]+)\]\s*\[longitude:\s*([-\d.]+)\]\s*\[altitude:\s*([-\d.]+)\]",
            text,
        )
        if not matches:
            raise ValueError(
                f"No latitude/longitude/altitude fields found in SRT file: {srt_file_path}"
            )

        srt_data = np.array(matches, dtype=np.float64)
        logger.info("Parsed %d SRT cues from %s", srt_data.shape[0], srt_file_path)
        return srt_data

    def _process_telemetry(self, telemetry_file_path: str) -> np.ndarray:
        """
        Parse a KABR-style telemetry CSV into an ``(M, 5)`` array of
        ``[frame, altitude, gimbal_heading, gimbal_pitch, gimbal_roll]``,
        sorted by ``frame`` ascending.
        """
        cols = [
            "frame",
            "altitude",
            "gimbal_heading(degrees)",
            "gimbal_pitch(degrees)",
            "gimbal_roll(degrees)",
        ]
        df = pd.read_csv(telemetry_file_path, usecols=cols).sort_values("frame")
        return df.to_numpy(dtype=np.float64)