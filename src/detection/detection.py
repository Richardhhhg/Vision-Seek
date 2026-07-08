# TODO: I don't like that this depends on core events but i think its ok (?)
import os

from core.events import DetectionOutput
from detection.data import DetectionInput
from detection.detection_model.model_factory import ModelFactory
from detection.postprocessing.postprocessor import Postprocessor
from detection.preprocessing.preprocessor import Preprocessor

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "detection_config.py") 

class Detection:
    """
    The actual class which runs the detection. Contains all components of the detection module

    Attributes:
    - detection_model (DetectionModel): Model used for detecting objects
    - preprocessor (Preprocessor): Preprocessor used for preprocessing the video before detection
    - postprocessor (Postprocessor): Postprocessor used for postprocessing the detection results

    Methods:
    TODO: run detection probably shouldn't take in a video path, instead it should probably take in some sort of data wrapper object
    - run_detection (video_path: str): Runs the detection on the given video path and returns the results
    """
    def __init__(self):
        self.detection_model, self.preprocessor, self.postprocessor = self._load_config(CONFIG_PATH)

    def _load_config(self, config_path: str):
        """
        Loads the configuration from the given path and returns the detection model and preprocessor instances.
        """
        config = {}
        with open(config_path, "r") as f:
            exec(f.read(), config)
        detection_model = ModelFactory.get_model(**config["model"])
        preprocessor = Preprocessor(**config["preprocessing"])
        postprocessor = Postprocessor(**config["postprocessing"])
        return detection_model, preprocessor, postprocessor

    def run_detection(self, detection_input: DetectionInput) -> DetectionOutput:
        # TODO: I think the only real thing we really need to work on with this is probably just the schema for the output and input
        # also need to build detection output correctly
        video = self.preprocessor.preprocess_video(detection_input.video_path)
        detections = self.detection_model.detect_video(video)
        postprocessed = self.postprocessor.postprocess_video(detections)

        result = DetectionOutput(postprocessed)
        return result