import json
import os

import torch

from detection.data import DetectionInput, DetectionOutput
from detection.detection_model.model_factory import ModelFactory
from detection.postprocessing.postprocessor import Postprocessor
from detection.preprocessing.preprocessor import Preprocessor

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "detection_config.json")


class Detection:
    """
    The actual class which runs the detection. Contains all components of the detection module

    Attributes:
    - detection_model (DetectionModel): Model used for detecting objects
    - preprocessor (Preprocessor): Preprocessor used for preprocessing the video before detection
    - postprocessor (Postprocessor): Postprocessor used for postprocessing the detection results

    Methods:
    - run_detection (detection_input: DetectionInput): Runs the detection on the given input and returns the results
    """
    def __init__(self, config_path: str = CONFIG_PATH):
        self.detection_model, self.preprocessor, self.postprocessor = self._load_config(config_path)

    def _load_config(self, config_path: str):
        """
        Loads the JSON configuration from the given path and returns the
        detection model, preprocessor, and postprocessor instances.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        detection_model = ModelFactory.get_model(**config["model"])

        preprocessor = Preprocessor()
        preprocessing_cfg = config.get("preprocessing", {})
        for step in preprocessing_cfg.get("steps", []):
            if isinstance(step, str):
                # For steps that don't require additional args
                preprocessor.add_step(step)
            elif isinstance(step, dict):
                # For steps that require additional args
                name = step.pop("name")
                preprocessor.add_step(name, **step)

        postprocessor = Postprocessor()

        return detection_model, preprocessor, postprocessor

    def run_detection(self, detection_input: DetectionInput) -> DetectionOutput:
        preprocessed_video = self.preprocessor.preprocess_video(detection_input.video_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detection_model_output = self.detection_model.detect(preprocessed_video, device)
        postprocessed = self.postprocessor.postprocess_video(detection_model_output)

        return DetectionOutput(
            output_video_path=postprocessed.output_video_path,
            annotated_frames=postprocessed.annotated_frames,
        )
