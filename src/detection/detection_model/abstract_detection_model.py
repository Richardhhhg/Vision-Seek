from abc import ABC

import numpy as np

from detection.data import DetectionFrame


class AbstractDetectionModel(ABC):
    """
    Abstract class for detection models
    """
    def detect(self, video: np.ndarray) -> list[DetectionFrame]:
        """
        Runs the video detection model on preprocessed video (in numpy array format) and returns results of bounding boxes and classes of detected objects for each frame in the video.

        Input:
        - video (np.ndarray): Preprocessed video in numpy array format. Each frame is represented as a 3D array (height, width, channels).
        """
        raise NotImplementedError("Detect method must be implemented by subclasses")