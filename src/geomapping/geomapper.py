from geomapping.data import (
    DeduplicationInput,
    DeduplicationOutput,
    GeoMappingInput,
    GeoMappingOutput,
    GeoTaggingInput,
    GeoTaggingOutput,
    TelemetryInput,
    TelemetryOutput,
)
from geomapping.deduplication.deduplicator import Deduplicator
from geomapping.geotagging.geotagger import GeoTagger
from geomapping.telemetry.telemetry import TelemetryProcessor


class GeoMapper:
    """
    Class responsible for the mapping of detected objects from detection module to real world coordinates.

    Attributes:
    - telemetry_processor (TelemetryProcessor): Processes SRT / telemetry files.
    - geotagger (GeoTagger): Maps detections to real-world GPS coordinates.
    - deduplicator (Deduplicator): Collapses duplicate detections across frames.

    Methods:
    - invoke (GeoMappingInput) -> GeoMappingOutput: Runs the entire geomapping pipeline.
    """

    def __init__(self):
        self.telemetry_processor = TelemetryProcessor()
        self.geotagger = GeoTagger()
        self.deduplicator = Deduplicator()

    def invoke(self, input: GeoMappingInput) -> GeoMappingOutput:
        telemetry = self.telemetry_processor.invoke(self._to_telemetry_input(input))
        geotagged = self.geotagger.invoke(self._to_geotagging_input(input, telemetry))
        deduped = self.deduplicator.invoke(self._to_deduplication_input(geotagged))
        return self._to_geomapping_output(deduped)

    def _to_telemetry_input(self, input: GeoMappingInput) -> TelemetryInput:
        return TelemetryInput(
            srt_file_path=input.srt_file_path,
            telemetry_file_path=input.telemetry_file_path,
        )

    def _to_geotagging_input(
        self, input: GeoMappingInput, telemetry: TelemetryOutput
    ) -> GeoTaggingInput:
        return GeoTaggingInput(
            detection_output=input.detection_output,
            camera_type=telemetry.camera_type,
            image_width=telemetry.image_width,
            image_height=telemetry.image_height,
            longitude=telemetry.longitude,
            latitude=telemetry.latitude,
            height_above_ground=telemetry.height_above_ground,
            gimbal_pitch=telemetry.gimbal_pitch,
            gimbal_yaw=telemetry.gimbal_yaw,
            horizontal_fov=telemetry.horizontal_fov,
            vertical_fov=telemetry.vertical_fov,
            diagonal_fov=telemetry.diagonal_fov,
            aspect_ratio=telemetry.aspect_ratio,
        )

    def _to_deduplication_input(self, geotagged: GeoTaggingOutput) -> DeduplicationInput:
        return DeduplicationInput(gps_coords=geotagged.gps_coords)

    def _to_geomapping_output(self, deduped: DeduplicationOutput) -> GeoMappingOutput:
        return GeoMappingOutput(gps_coords=deduped.gps_coords)
