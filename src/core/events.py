from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from detection.data import DetectionOutput
from geomapping.data import GeoMappingOutput


class DetectionRequestEvent(BaseModel):
    """
    Event for the input of the detection service. This represents a video that is passed into the detection module.

    Attributes:
    - video_path: Path to the video that is passed into the detection module
    - version: Version of the DetectionRequestEvent Schema
    - metadata: Metadata of the DetectionRequestEvent
    """
    video_path: str
    version: str = "0.1.0"
    metadata: MetaData

class DetectionCompleteEvent(BaseModel):
    """
    Event for the output of the detection service. This represents all the detections for some video that was passed into the detection module.

    Attributes:
    - detection_output: The output of the detection module, which contains all the detections for a given video.
    - version: Version of the DetectionCompleteEvent Schema
    - metadata: Metadata of the DetectionCompleteEvent
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_output: DetectionOutput
    version: str = "0.1.0"
    metadata: MetaData

class GeoMappingRequestEvent(BaseModel):
    """
    Event for the input of the geomapping service.

    Attributes:
    - detection_output: Detection results to geotag.
    - srt_file_path: Path to the SRT file for the video.
    - telemetry_file_path: Path to the additional telemetry CSV.
    - video_file_path: Path to the source video.
    - version: Version of the GeoMappingRequestEvent Schema.
    - metadata: Metadata of the GeoMappingRequestEvent.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection_output: DetectionOutput
    srt_file_path: str
    telemetry_file_path: str
    video_file_path: str
    version: str = "0.1.0"
    metadata: MetaData

class GeoMappingCompleteEvent(BaseModel):
    """
    Output of the GeoMapping module. Represents all objects that have been detected, deduplicated, and mapped into the real world.

    Attributes:
    - geomapping_output: Deduplicated GPS bounding boxes for unique physical objects.
    - version: Version of GeoMappingCompleteEvent Schema.
    - metadata: Metadata of the GeoMappingCompleteEvent.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geomapping_output: GeoMappingOutput
    version: str = "0.1.0"
    metadata: MetaData

class MetaData(BaseModel):
    """
    Represents the metadata of the process

    Attributes:
    - version: Version of Metadata Schema
    """
    version: str = "0.1.0"
