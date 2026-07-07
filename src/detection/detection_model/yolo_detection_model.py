import logging

import cv2
import numpy as np
from ultralytics import YOLO

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
    - detect_video(video_path: str): Performs detection on the input video and returns a list of detection results for each unique object
    """
    def __init__(self, model_path: str = "src/detect/detection_model/model/model.pt"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        return YOLO(self.model_path)

    def _detect_image(self, image: np.ndarray):
        return self.model(image)

    def detect(self, video: np.ndarray):
        # TODO: This logic is outdated, update this to support processing with just np.ndarray.
        # Get metadata from video and place to write the annotated video
        # Iterate through frames in the video
        # detect the objects in the frame
        # Annotate the frame with the detected objects and write to the output video
        # store the labels and stuff in some data structure to be returned at end (as of writing this, the data object is not implemented)
        # close all things relating to video and return
        video = cv2.VideoCapture(video_path)
    
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = VIDEOS + "/output/" + "detect_" + Path(video_path).stem + ".mp4"
        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            results = self.model(frame)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"{self.model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # annotated = results.plot()
            output.write(frame)

        output.release()
        video.release()

        return