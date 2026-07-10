import asyncio
from pathlib import Path

from core.events import DetectionCompleteEvent, DetectionRequestEvent
from detection.data import DetectionInput
from detection.detection import Detection
from detection.detection_model.model_factory import ModelFactory
from detection.preprocessing.preprocessor import Preprocessor

DEFAULT_MODEL_PATH = str(Path(__file__) / "detection_model" / "model" / "test_model.pt")
DEFAULT_MODEL_TYPE = "yolo"

class DetectionService:
    """
    Service class for connecting the detection module to the rest of the program.

    Publishes:
    - DetectionOutput: The output of the detection module, which contains all the detections for a given video.

    Attributes:
    - event_bus: The event bus instance for communication between services.
    - queue: An asyncio queue for receiving events from the event bus.
    - detect: the actual detection object which is used on the input video
    """
    def __init__(self, event_bus, model_type: str = DEFAULT_MODEL_TYPE, model_path: str=DEFAULT_MODEL_PATH):
        self.event_bus = event_bus
        self.queue = asyncio.Queue()
        detection_model = ModelFactory.get_model(model_type=model_type, model_path=model_path)
        self.detect = Detection(detection_model, Preprocessor())

        self.event_bus.subscribe(DetectionRequestEvent, self.queue)
        

    async def run(self):
        while True:
            event = await self.queue.get()
            detection_input = DetectionInput(video_path=event.video_path)
            detection_output = self.detect.run_detection(detection_input)
            detection_complete_event = DetectionCompleteEvent(detection_output=detection_output, metadata=event.metadata)
            await self.event_bus.publish(detection_complete_event)
