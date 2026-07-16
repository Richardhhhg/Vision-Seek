import asyncio

from core.events import GeoMappingCompleteEvent, GeoMappingRequestEvent
from geomapping.data import GeoMappingInput
from geomapping.geomapper import GeoMapper


class GeoMappingService:
    """
    The main point of communication between the GeoMapping module and the rest of the system. Utilizes eventbuses to take in and output data.

    Publishes:
    - GeoMappingCompleteEvent: the mapped objects into the real world.

    Attributes:
    - event_bus: event bus for communicating between services.
    - queue: the queue to which the GeoMapping module is subscribed to
    - geomapper: the geomapping module which does the actual work of mapping detected objects to real world coordinates.

    Methods:
    - run (): runs geomapping module asynchronously
    """

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.queue = asyncio.Queue()
        self.geomapper = GeoMapper()

        self.event_bus.subscribe(GeoMappingRequestEvent, self.queue)

    async def run(self):
        while True:
            event = await self.queue.get()
            geomapping_input = GeoMappingInput(
                detection_output=event.detection_output,
                srt_file_path=event.srt_file_path,
                telemetry_file_path=event.telemetry_file_path,
                video_file_path=event.video_file_path,
            )
            geomapping_output = self.geomapper.invoke(geomapping_input)
            await self.event_bus.publish(
                GeoMappingCompleteEvent(
                    geomapping_output=geomapping_output,
                    metadata=event.metadata,
                )
            )
