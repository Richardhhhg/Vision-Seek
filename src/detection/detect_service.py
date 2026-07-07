import asyncio

from core.events import DetectionInput, DetectionOutput
from detection.detection import Detection
from detection.detection_model.detection_model import DetectionModel
from detection.preprocessing.preprocessor import Preprocessor


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
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.queue = asyncio.Queue()
        self.detect = Detection(DetectionModel(), Preprocessor())

        self.event_bus.subscribe(DetectionInput, self.queue)
        

    async def run(self):
        while True:
            event = await self.queue.get()

            # TODO: process this event in some way
            # pass the event into the actual detection model
            # get the result of the detection model
            # create a DetectionOutput event with the result
            # publish the DetectionOutput event to the event bus
            return self.detect.run_detection(event.video_path)

