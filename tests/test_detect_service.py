# Tests to write:
# 1. Just end to end, passing some data into the detect service and making sure it returns something
# 2. Passing in Malformed data and make sure its handled correctly
import asyncio

import pytest

from core.eventbus import EventBus
from core.events import DetectionCompleteEvent, DetectionRequestEvent
from detection.data import DetectionOutput
from detection.detect_service import DetectionService
from detection.tests.paths import TEST_VIDEO_PATH


@pytest.fixture(scope="function")
def setup():
    eventbus = EventBus()
    detection_service = DetectionService(eventbus)
    output_queue = asyncio.Queue()
    eventbus.subscribe(DetectionCompleteEvent, output_queue)

    return detection_service, eventbus, output_queue

@pytest.mark.asyncio
async def test_detect_service_end_to_end():
    detection_service, eventbus, output_queue = setup()
    event = DetectionRequestEvent(video_path=TEST_VIDEO_PATH)
    detection_task = asyncio.create_task(detection_service.run())
    # NOTE: without this sleep statement, we are in serious risk of race conditions
    await asyncio.sleep(2.0)
    try:
        await eventbus.publish(event)
        result = await asyncio.wait_for(output_queue.get(), timeout=600) # 10 minute timeout window

        assert isinstance(result, DetectionCompleteEvent)
        assert isinstance(result.detection_output, DetectionOutput)
        assert result.metadata is not None
    finally:
        detection_task.cancel()
        try:
            await detection_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_detect_service_malformed_data():
    detection_service, eventbus, output_queue = setup()
    event = DetectionRequestEvent(video_path=TEST_VIDEO_PATH)
    detection_task = asyncio.create_task(detection_service.run())
    # NOTE: without this sleep statement, we are in serious risk of race conditions
    await asyncio.sleep(2.0)
    with pytest.raises(ValueError):
        await eventbus.publish(event)
        result = await output_queue.get()
        assert False