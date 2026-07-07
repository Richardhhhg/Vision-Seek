# TODO: I don't like that this depends on core events but i think its ok (?)
import os

from core.events import DetectionOutput
from detection.detection_model.model_factory import ModelFactory
from detection.preprocessing.preprocessor import Preprocessor

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "detection_config.py") 

class Detection:
    """
    The actual class which runs the detection. Contains all components of the detection module

    Attributes:
    - detection_model (DetectionModel): Model used for detecting objects
    - preprocessor (Preprocessor): Preprocessor used for preprocessing the video before detection

    Methods:
    TODO: run detection probably shouldn't take in a video path, instead it should probably take in some sort of data wrapper object
    - run_detection (video_path: str): Runs the detection on the given video path and returns the results
    """
    def __init__(self):
        self.detection_model, self.preprocessor = self._load_config(CONFIG_PATH)

    def _load_config(self, config_path: str):
        """
        Loads the configuration from the given path and returns the detection model and preprocessor instances.
        """
        config = {}
        with open(config_path, "r") as f:
            exec(f.read(), config)
        detection_model = ModelFactory.get_model(**config["model"])
        preprocessor = Preprocessor(**config["preprocessing"])
        return detection_model, preprocessor

    # TODO: This should probably not just take in a str
    def run_detection(self, video_path: str) -> DetectionOutput:
        # TODO: I think the only real thing we really need to work on with this is probably just the schema for the output and input
        video = self.preprocessor.preprocess_video(video_path)
        detections = self.detection_model.detect_video(video)

        result = DetectionOutput(detections)
        return result