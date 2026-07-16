import json
import os
import uuid
from pathlib import Path

import numpy as np

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GEOJSON_OUTPUTS_DIR = os.path.join(_REPO_ROOT, "geojson_outputs")


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
        geojson_path = self._save_geojson(deduped.gps_coords, input.video_file_path)
        return self._to_geomapping_output(deduped, geojson_path)

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

    def _to_geomapping_output(
        self, deduped: DeduplicationOutput, geojson_path: str
    ) -> GeoMappingOutput:
        return GeoMappingOutput(
            gps_coords=deduped.gps_coords,
            geojson_path=geojson_path,
        )

    def _save_geojson(self, gps_coords: np.ndarray, video_file_path: str) -> str:
        """
        Write a GeoJSON FeatureCollection of closed Polygon rings for each unique object.
        Corner order is [tl, tr, br, bl] as (lat, lon); GeoJSON uses [lon, lat].
        """
        os.makedirs(_GEOJSON_OUTPUTS_DIR, exist_ok=True)
        stem = Path(video_file_path).stem or "geomap"
        output_path = os.path.join(
            _GEOJSON_OUTPUTS_DIR, f"{stem}_{uuid.uuid4().hex}.geojson"
        )

        features = []
        for i, corners in enumerate(np.asarray(gps_coords, dtype=np.float64)):
            # corners: (4, 2) as (lat, lon) -> GeoJSON ring as (lon, lat), closed.
            ring = [[float(lon), float(lat)] for lat, lon in corners]
            ring.append(ring[0])
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
        return output_path
