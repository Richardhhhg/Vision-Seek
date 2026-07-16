import asyncio

from core.events import DetectionCompleteEvent, DetectionRequestEvent
from detection.data import DetectionInput
from detection.detection import CONFIG_PATH, Detection


class DetectionService:
    """
    Service class for connecting the detection module to the rest of the program.

    Publishes:
    - DetectionCompleteEvent: The output of the detection module for a given video.

    Attributes:
    - event_bus: The event bus instance for communication between services.
    - queue: An asyncio queue for receiving events from the event bus.
    - detect: the actual detection object which is used on the input video
    """

    def __init__(self, event_bus, config_path: str = CONFIG_PATH):
        self.event_bus = event_bus
        self.queue = asyncio.Queue()
        self.detect = Detection(config_path=config_path)

        self.event_bus.subscribe(DetectionRequestEvent, self.queue)

    async def run(self):
        while True:
            event = await self.queue.get()
            detection_input = DetectionInput(video_path=event.video_path)
            detection_output = self.detect.invoke(detection_input)
            detection_complete_event = DetectionCompleteEvent(
                detection_output=detection_output, metadata=event.metadata
            )
            await self.event_bus.publish(detection_complete_event)
