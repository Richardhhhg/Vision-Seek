class GeoMappingService:
    """
    The main point of communication between the GeoMapping module and the rest of the system. Utilizes eventbuses to take in and output data.
    
    Publishes:
    - GeoMappingCompleteEvent: the mapped objects into the real world.
    
    Attributes:
    - event_bus: event bus for communicating between services.
    - queue: the queue to which the GeoMapping module is subscribed to
    - geomapper: the geomapping module which does the actual work of mapping detected objects to real world coordinates.

    Methods:
    - run (): runs geomapping module asynchronously
    """
    pass