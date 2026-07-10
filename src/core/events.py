from __future__ import annotations

from pydantic import BaseModel

from detection.data import DetectionOutput


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
    detection_output: DetectionOutput
    version: str = "0.1.0"
    metadata: MetaData

class MappingOutput(BaseModel):
    """
    Output event data for the mapping module. This represents all of the different objects that were detected from the detection module and their corresponding locations in the world.
    """
    pass

class MetaData(BaseModel):
    """
    Represents the metadata of the process

    Attributes:
    - version: Version of Metadata Schema
    """
    version: str = "0.1.0"