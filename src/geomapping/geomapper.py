class GeoMapper:
    """
    Class responsible for the mapping of detected objects from detection module to real world coordinates. 

    Attributes:
    - deduplication (Deduplication): Deduplication module that removes duplicate detections.
    - geolocation (Geolocation): Geolocation module that maps the detections to real world coordinates.
    - srt_processor (SRTProcessor): SRT Processor module that processes the SRT files to extract the necessary information for geolocation.

    Methods:
    - invoke (GeoMappingInput) -> GeoMappingOutput: Runs the entire geomapping pipeline.
    """
    pass