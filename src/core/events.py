from pydantic import BaseModel


class DetectionInput(BaseModel):
    """
    Event for the input of the detection service. This represents a video that is passed into the detection module.
    """
    pass

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