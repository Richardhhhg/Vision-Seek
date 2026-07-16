# Tests for the full pipeline of running detection on a video, then mapping stuff into the real world.
# 1. General Smoke test, just to make sure the pipeline does in fact run and doesn't explode
# 2. Trying to run the pipeline with malformed input data, and making sure it fails gracefully
import asyncio
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core.eventbus import EventBus
from core.events import (
    DetectionCompleteEvent,
    DetectionRequestEvent,
    GeoMappingCompleteEvent,
    GeoMappingRequestEvent,
    MetaData,
)
from detection.detect_service import DetectionService
from detection.tests.paths import TEST_CONFIG_REAL_PATH
from geomapping.data import GeoMappingOutput
from geomapping.geomapping_service import GeoMappingService
from geomapping.tests.paths import TEST_SRT_PATH, TEST_TELEMETRY_PATH, TEST_VIDEO_PATH


@pytest.fixture
def setup():
    eventbus = EventBus()
    detection_service = DetectionService(eventbus, config_path=TEST_CONFIG_REAL_PATH)
    geomapping_service = GeoMappingService(eventbus)
    detection_queue = asyncio.Queue()
    geomapping_queue = asyncio.Queue()
    eventbus.subscribe(DetectionCompleteEvent, detection_queue)
    eventbus.subscribe(GeoMappingCompleteEvent, geomapping_queue)
    return (
        detection_service,
        geomapping_service,
        eventbus,
        detection_queue,
        geomapping_queue,
    )


@pytest.mark.asyncio
async def test_detect_map_pipeline_smoke(setup):
    (
        detection_service,
        geomapping_service,
        eventbus,
        detection_queue,
        geomapping_queue,
    ) = setup

    detection_task = asyncio.create_task(detection_service.run())
    geomapping_task = asyncio.create_task(geomapping_service.run())
    try:
        await eventbus.publish(
            DetectionRequestEvent(video_path=TEST_VIDEO_PATH, metadata=MetaData())
        )
        detection_result = await asyncio.wait_for(detection_queue.get(), timeout=600)
        assert isinstance(detection_result, DetectionCompleteEvent)

        await eventbus.publish(
            GeoMappingRequestEvent(
                detection_output=detection_result.detection_output,
                srt_file_path=TEST_SRT_PATH,
                telemetry_file_path=TEST_TELEMETRY_PATH,
                video_file_path=TEST_VIDEO_PATH,
                metadata=detection_result.metadata,
            )
        )
        geomapping_result = await asyncio.wait_for(geomapping_queue.get(), timeout=600)
        assert isinstance(geomapping_result, GeoMappingCompleteEvent)
        assert isinstance(geomapping_result.geomapping_output, GeoMappingOutput)
        assert isinstance(geomapping_result.geomapping_output.gps_coords, np.ndarray)
        assert geomapping_result.geomapping_output.gps_coords.ndim == 3
        assert geomapping_result.geomapping_output.gps_coords.shape[1:] == (4, 2)
        assert Path(geomapping_result.geomapping_output.geojson_path).is_file()
    finally:
        detection_task.cancel()
        geomapping_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await detection_task
        with pytest.raises(asyncio.CancelledError):
            await geomapping_task


@pytest.mark.asyncio
async def test_detect_map_pipeline_malformed_srt(setup, tmp_path: Path):
    (
        detection_service,
        geomapping_service,
        eventbus,
        detection_queue,
        geomapping_queue,
    ) = setup

    bad_srt = tmp_path / "bad.srt"
    bad_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,033\nno telemetry fields here\n",
        encoding="utf-8",
    )

    detection_task = asyncio.create_task(detection_service.run())
    geomapping_task = asyncio.create_task(geomapping_service.run())
    try:
        await eventbus.publish(
            DetectionRequestEvent(video_path=TEST_VIDEO_PATH, metadata=MetaData())
        )
        detection_result = await asyncio.wait_for(detection_queue.get(), timeout=600)

        await eventbus.publish(
            GeoMappingRequestEvent(
                detection_output=detection_result.detection_output,
                srt_file_path=str(bad_srt),
                telemetry_file_path=TEST_TELEMETRY_PATH,
                video_file_path=TEST_VIDEO_PATH,
                metadata=detection_result.metadata,
            )
        )
        with pytest.raises(ValueError):
            await asyncio.wait_for(geomapping_task, timeout=30)
    finally:
        if not detection_task.done():
            detection_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await detection_task
        if not geomapping_task.done():
            geomapping_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await geomapping_task


def test_detect_map_malformed_request():
    with pytest.raises(ValidationError):
        DetectionRequestEvent(video_path=TEST_VIDEO_PATH)

    with pytest.raises(ValidationError):
        GeoMappingRequestEvent(
            detection_output=None,
            srt_file_path=TEST_SRT_PATH,
            telemetry_file_path=TEST_TELEMETRY_PATH,
            video_file_path=TEST_VIDEO_PATH,
            metadata=MetaData(),
        )
