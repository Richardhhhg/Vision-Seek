from __future__ import annotations

from pydantic import BaseModel


class DetectionRequestEvent(BaseModel):
    """
    Event for the input of the detection service. This represents a video that is passed into the detection module.
    """
    video_path: str
    version: str = "0.1.0"
    metadata: MetaData

class DetectionOutput(BaseModel):
    """
    Event for the output of the detection service. This represents all the detections for some video that was passed into the detection module.
    """
    pass

class MappingOutput(BaseModel):
    """
    Output event data for the mapping module. This represents all of the different objects that were detected from the detection module and their corresponding locations in the world.
    """
    pass

class MetaData(BaseModel):
    """
    Represents the metadata of the process that is working.
    """
    pass