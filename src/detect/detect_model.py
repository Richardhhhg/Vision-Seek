import numpy as np
from ultralytics import YOLO


class DetectModel:
    """
    Model for detecting the objects in a given frame.
    Currently this only supports a yolo model.

    Attributes:
    - model (YOLO): The YOLO model used for detection.
    - model_path (str): The path to the YOLO model file.

    Methods:
    - load_model(): Loads the YOLO model from the specified path.
    - detect_image(image: np.ndarray): Performs detection on the input image and returns the results as yolo result format
    - detect_video(video_path: str): Performs detection on the input video and returns a list of detection results for each unique object
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return YOLO(self.model_path)  # Replace with actual model loading code

    def detect_image(self, image: np.ndarray):
        return self.model(image)

    def detect_video(self, video_path: str):
        pass