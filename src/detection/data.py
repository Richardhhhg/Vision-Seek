from pydantic import BaseModel


class DetectionInput(BaseModel):
    """
    Input event data for the detection module. This represents a video that is passed into the detection module.
    """
    video_path: str
    version: str = "0.1.0"

class DetectionOutput(BaseModel):
    """
    Output event data for the detection module. This represents all of the different objects that were detected from the detection module and their corresponding locations in the video.
    """
    output_video_path: str
    annotated_frames: list # TODO: idk what it should be
    version: str = "0.1.0"