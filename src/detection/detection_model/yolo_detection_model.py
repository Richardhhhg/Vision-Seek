import logging

import numpy as np
import torch
from ultralytics import YOLO

from detection.data import DetectionFrame, DetectionModelOutput, PreprocessedVideo
from detection.detection_model.abstract_detection_model import AbstractDetectionModel

logger = logging.getLogger("detect")


class YOLODetectionModel(AbstractDetectionModel):
    """
    Model for detecting the objects in a given frame.
    Currently this only supports a yolo model.

    Attributes:
    - model (YOLO): The YOLO model used for detection.
    - model_path (str): The path to the YOLO model file.

    Methods:
    - load_model(): Loads the YOLO model from the specified path.
    - _detect_image(image: np.ndarray): Performs detection on the input image and returns the results as yolo result format
    - detect(preprocessed_video: PreprocessedVideo, device: str = "cpu"): Performs detection on all frames of the preprocessed video and returns a DetectionModelOutput.
    """
    def __init__(self, model_path: str = "src/detection/detection_model/model/test_model.pt"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        return YOLO(self.model_path)

    def _detect_image(self, image: np.ndarray, device: str = "cpu"):
        # YOLO can in fact handle gpu processing even if image is numpy
        return self.model(image, device=device, verbose=False)

    def detect(
        self, preprocessed_video: PreprocessedVideo, device: str = "cpu"
    ) -> DetectionModelOutput:
        frames = preprocessed_video.frames
        if isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = np.asarray(frames)

        num_frames = frames_np.shape[0]
        annotated_frames: list[DetectionFrame] = []
        use_cuda = device == "cuda"

        for i in range(num_frames):
            frame = frames_np[i]
            results = self._detect_image(frame, device=device)
            result = results[0]
            boxes = result.boxes

            if use_cuda:
                # Keep attributes as torch.Tensor
                xyxy = boxes.xyxy
                xywh = boxes.xywh
                xyxyn = boxes.xyxyn
                conf = boxes.conf
                cls = boxes.cls
            else:
                # Convert attributes to numpy arrays
                xyxy = boxes.xyxy.cpu().numpy()
                xywh = boxes.xywh.cpu().numpy()
                xyxyn = boxes.xyxyn.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

            annotated_frames.append(
                DetectionFrame(
                    frame_number=i,
                    xyxy=xyxy,
                    xywh=xywh,
                    xyxyn=xyxyn,
                    conf=conf,
                    cls=cls,
                )
            )

        return DetectionModelOutput(
            annotated_frames=annotated_frames,
            output_video_path=None,
            preprocessed_video=preprocessed_video,
        )
