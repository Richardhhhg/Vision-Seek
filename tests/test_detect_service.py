# Tests to write:
# 1. Just end to end, passing some data into the detect service and making sure it returns something
# 2. Passing in Malformed data and make sure its handled correctly
import asyncio

import pytest
from pydantic import ValidationError

from core.eventbus import EventBus
from core.events import DetectionCompleteEvent, DetectionRequestEvent, MetaData
from detection.data import DetectionOutput
from detection.detect_service import DetectionService
from detection.tests.paths import TEST_CONFIG_REAL_PATH, TEST_VIDEO_PATH


@pytest.fixture
def setup():
    eventbus = EventBus()
    detection_service = DetectionService(eventbus, config_path=TEST_CONFIG_REAL_PATH)
    output_queue = asyncio.Queue()
    eventbus.subscribe(DetectionCompleteEvent, output_queue)
    return detection_service, eventbus, output_queue


@pytest.mark.asyncio
async def test_detect_service_end_to_end(setup):
    detection_service, eventbus, output_queue = setup
    event = DetectionRequestEvent(video_path=TEST_VIDEO_PATH, metadata=MetaData())

    detection_task = asyncio.create_task(detection_service.run())
    try:
        await eventbus.publish(event)
        result = await asyncio.wait_for(output_queue.get(), timeout=600)

        assert isinstance(result, DetectionCompleteEvent)
        assert isinstance(result.detection_output, DetectionOutput)
        assert result.detection_output.output_video_path is not None
        assert len(result.detection_output.annotated_frames) > 0
        assert result.metadata == event.metadata
    finally:
        detection_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await detection_task


def test_detect_service_malformed_data():
    # Request events require video_path + metadata; invalid payloads must fail at the boundary.
    with pytest.raises(ValidationError):
        DetectionRequestEvent(video_path=TEST_VIDEO_PATH)

    with pytest.raises(ValidationError):
        DetectionRequestEvent(video_path=123, metadata=MetaData())
