from abc import ABC

import numpy as np

from detection.data import DetectionModelOutput


class AbstractDetectionModel(ABC):
    """
    Abstract class for detection models
    """
    def detect(self, video: np.ndarray, device: str = "cpu") -> DetectionModelOutput:
        """
        Runs the video detection model on preprocessed video (in numpy array format) and returns results of bounding boxes and classes of detected objects for each frame in the video.

        Input:
        - video (np.ndarray): Preprocessed video in numpy array format. Each frame is represented as a 3D array (height, width, channels).
        - device (str): The device to run the model on.
        """
        raise NotImplementedError("Detect method must be implemented by subclasses")