import logging

from detection.detection_model.abstract_detection_model import AbstractDetectionModel
from detection.detection_model.yolo_detection_model import YOLODetectionModel

logger = logging.getLogger("detect")


class ModelFactory:
    @staticmethod
    def get_model(model_type: str, model_path: str, **kwargs) -> AbstractDetectionModel:
        model_type = model_type.lower()

        if model_type == "yolo":
            return YOLODetectionModel(model_path)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
