# Tests for running just the geomapping service end to end.
# 1. Smoke test to make sure nothing explodes on end to end run from queue
# 2. Testing malformed data? This is likely just going to be malformed SRT data.
import asyncio
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core.eventbus import EventBus
from core.events import GeoMappingCompleteEvent, GeoMappingRequestEvent, MetaData
from detection.data import DetectionFrame, DetectionOutput
from geomapping.data import GeoMappingOutput
from geomapping.geomapping_service import GeoMappingService
from geomapping.tests.paths import TEST_SRT_PATH, TEST_TELEMETRY_PATH, TEST_VIDEO_PATH


def _fake_detection_output(num_frames: int = 3) -> DetectionOutput:
    frames = []
    for i in range(num_frames):
        frames.append(
            DetectionFrame(
                frame_number=i,
                xyxy=np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float64),
                xywh=np.array([[150.0, 150.0, 100.0, 100.0]], dtype=np.float64),
                xyxyn=np.array([[0.05, 0.09, 0.10, 0.19]], dtype=np.float64),
                conf=np.array([0.9], dtype=np.float64),
                cls=np.array([0], dtype=np.float64),
            )
        )
    return DetectionOutput(output_video_path=TEST_VIDEO_PATH, annotated_frames=frames)


@pytest.fixture
def setup():
    eventbus = EventBus()
    geomapping_service = GeoMappingService(eventbus)
    output_queue = asyncio.Queue()
    eventbus.subscribe(GeoMappingCompleteEvent, output_queue)
    return geomapping_service, eventbus, output_queue


@pytest.mark.asyncio
async def test_geomapping_service_end_to_end(setup):
    geomapping_service, eventbus, output_queue = setup
    event = GeoMappingRequestEvent(
        detection_output=_fake_detection_output(),
        srt_file_path=TEST_SRT_PATH,
        telemetry_file_path=TEST_TELEMETRY_PATH,
        video_file_path=TEST_VIDEO_PATH,
        metadata=MetaData(),
    )

    task = asyncio.create_task(geomapping_service.run())
    try:
        await eventbus.publish(event)
        result = await asyncio.wait_for(output_queue.get(), timeout=60)

        assert isinstance(result, GeoMappingCompleteEvent)
        assert isinstance(result.geomapping_output, GeoMappingOutput)
        assert isinstance(result.geomapping_output.gps_coords, np.ndarray)
        assert result.geomapping_output.gps_coords.ndim == 3
        assert result.geomapping_output.gps_coords.shape[1:] == (4, 2)
        assert result.geomapping_output.geojson_path is not None
        assert Path(result.geomapping_output.geojson_path).is_file()
        assert result.metadata == event.metadata
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_geomapping_service_malformed_srt(setup, tmp_path: Path):
    geomapping_service, eventbus, output_queue = setup
    bad_srt = tmp_path / "bad.srt"
    bad_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,033\nno telemetry fields here\n",
        encoding="utf-8",
    )
    event = GeoMappingRequestEvent(
        detection_output=_fake_detection_output(),
        srt_file_path=str(bad_srt),
        telemetry_file_path=TEST_TELEMETRY_PATH,
        video_file_path=TEST_VIDEO_PATH,
        metadata=MetaData(),
    )

    task = asyncio.create_task(geomapping_service.run())
    try:
        await eventbus.publish(event)
        with pytest.raises(ValueError):
            # The service has no try/except, so the failure surfaces on the task.
            await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


def test_geomapping_service_malformed_request():
    with pytest.raises(ValidationError):
        GeoMappingRequestEvent(
            detection_output=_fake_detection_output(),
            srt_file_path=TEST_SRT_PATH,
            telemetry_file_path=TEST_TELEMETRY_PATH,
            video_file_path=TEST_VIDEO_PATH,
        )

    with pytest.raises(ValidationError):
        GeoMappingRequestEvent(
            detection_output=_fake_detection_output(),
            srt_file_path=123,
            telemetry_file_path=TEST_TELEMETRY_PATH,
            video_file_path=TEST_VIDEO_PATH,
            metadata=MetaData(),
        )
