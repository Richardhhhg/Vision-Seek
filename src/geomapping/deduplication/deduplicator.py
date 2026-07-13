class Deduplicator:
    """
    Removes objects that were detected multiple times. Ensures that on the final output, each object is only represented once in GPS coordinates.

    Methods:
    - invoke (DeduplicationInput) -> DeduplicationOutput: Takes in calculated gps coordinates of detected objects and removes duplicates. Returns same list of objects but with duplicates removed.
    """