from abc import ABC

from detection.data import DetectionModelOutput, PreprocessedVideo


class AbstractDetectionModel(ABC):
    """
    Abstract class for detection models
    """
    def detect(
        self, preprocessed_video: PreprocessedVideo, device: str = "cpu"
    ) -> DetectionModelOutput:
        """
        Runs the video detection model on the preprocessed video and returns results
        of bounding boxes and classes of detected objects for each frame.

        Input:
        - preprocessed_video (PreprocessedVideo): The preprocessed video, whose `frames` attribute
          holds the tiled/transformed frames (num_frames, height, width, channels).
        - device (str): The device to run the model on ("cpu" or "cuda").
        """
        raise NotImplementedError("Detect method must be implemented by subclasses")
