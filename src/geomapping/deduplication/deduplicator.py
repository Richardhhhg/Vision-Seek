import logging

import numpy as np
from pyproj import Geod
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from geomapping.data import DeduplicationInput, DeduplicationOutput

logger = logging.getLogger("dedup")


class Deduplicator:
    """
    Collapses detections of the same physical object across the video into a single
    averaged bounding box. The GeoTagger emits (lat, lon) bboxes per frame; the same
    physical object typically appears in many consecutive frames, so this pass groups
    detections whose centroids are within ``_DISTANCE_THRESHOLD_M`` meters and averages
    their bbox corners into one representative object.

    Vectorization contract:
    - All per-frame detections are concatenated into a single (M, 4, 2) array up front,
      so the entire pipeline (centroids, pairwise distances, clustering, averaging)
      operates on flat numpy arrays without per-frame Python loops.
    - Pairwise geodesic distances between centroids are computed in a single vectorized
      ``pyproj.Geod.inv`` call over the upper-triangle index pairs.
    - Cluster assignment uses single-linkage hierarchical clustering
      (``scipy.cluster.hierarchy.linkage`` + ``fcluster`` with ``criterion='distance'``),
      so any chain of near-duplicates (A~B, B~C) is merged into one cluster.
    - Cluster averaging is done via ``np.add.at`` + ``np.bincount`` for a vectorized
      group-mean over corner coordinates.

    Bounding box corner ordering (inherited from GeoTagger):
        0: (x1, y1)   top-left
        1: (x2, y1)   top-right
        2: (x2, y2)   bottom-right
        3: (x1, y2)   bottom-left

    Note on averaging (lat, lon): arithmetic mean is used per corner. This is accurate
    for the scales involved (drone footage where all detections of one object are within
    meters of each other) but is not correct near the poles or across the antimeridian.

    Attributes:
    - _DISTANCE_THRESHOLD_M (float): Two detections are considered duplicates when the
      geodesic distance between their bbox centroids is <= this value (meters).
    - _GEOD (pyproj.Geod): WGS84 geoid used for vectorized geodesic distance.

    Methods:
    - invoke (DeduplicationInput) -> DeduplicationOutput: Takes in calculated gps
      coordinates of detected objects and removes duplicates. Returns a single
      (N_unique, 4, 2) array of averaged bboxes.

    Private Methods:
    - _flatten_detections (DeduplicationInput) -> np.ndarray: Concatenate all per-frame
      detections into a single (M, 4, 2) array.
    - _get_centroids (bboxes) -> np.ndarray: Average the 4 corners of each detection to
      get a (M, 2) centroid (lat, lon).
    - _get_pairwise_distances (centroids) -> np.ndarray: Vectorized (M, M) geodesic
      distance matrix between all centroid pairs.
    - _cluster_detections (dist_matrix) -> np.ndarray: Single-linkage cluster labels
      for each detection.
    - _average_clusters (bboxes, labels) -> np.ndarray: Per-cluster mean of bbox
      corners, returning a (N_unique, 4, 2) array.
    - _deduplicate (DeduplicationInput) -> np.ndarray: End-to-end pipeline that ties
      the above together.
    """
    _DISTANCE_THRESHOLD_M = 1.0
    _GEOD = Geod(ellps="WGS84")

    def __init__(self):
        pass

    def invoke(self, input_data: DeduplicationInput) -> DeduplicationOutput:
        gps_coords = self._deduplicate(input_data)
        return DeduplicationOutput(gps_coords=gps_coords)

    def _flatten_detections(self, data: DeduplicationInput) -> np.ndarray:
        """
        Concatenate all per-frame (N_f, 4, 2) detections into a single (M, 4, 2) array.
        Empty frames are skipped. Returns an empty (0, 4, 2) array if there are no
        detections anywhere in the video.
        """
        frames = data.gps_coords
        if not frames:
            return np.empty((0, 4, 2), dtype=np.float64)
        non_empty = [
            np.asarray(f, dtype=np.float64) for f in frames if np.asarray(f).size > 0
        ]
        if not non_empty:
            return np.empty((0, 4, 2), dtype=np.float64)
        return np.concatenate(non_empty, axis=0)

    def _get_centroids(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Average the 4 corners of each bounding box to get a single (lat, lon) centroid
        per detection. Input (M, 4, 2) -> output (M, 2).
        """
        return bboxes.mean(axis=1)

    def _get_pairwise_distances(self, centroids: np.ndarray) -> np.ndarray:
        """
        Vectorized pairwise geodesic distance matrix (meters).

        Input:  centroids of shape (M, 2), where axis 1 is (lat, lon).
        Output: symmetric (M, M) distance matrix with zeros on the diagonal.

        Uses ``pyproj.Geod.inv`` which is fully vectorized: given equal-length arrays
        of (lon1, lat1, lon2, lat2) it returns forward azimuth, back azimuth, and
        distance arrays of matching length. We only evaluate the upper triangle and
        mirror it, avoiding redundant computation.
        """
        m = centroids.shape[0]
        if m < 2:
            return np.zeros((m, m), dtype=np.float64)

        idx_i, idx_j = np.triu_indices(m, k=1)
        lats1 = centroids[idx_i, 0]
        lons1 = centroids[idx_i, 1]
        lats2 = centroids[idx_j, 0]
        lons2 = centroids[idx_j, 1]
        _, _, dist = self._GEOD.inv(lons1, lats1, lons2, lats2)

        d = np.zeros((m, m), dtype=np.float64)
        d[idx_i, idx_j] = dist
        d[idx_j, idx_i] = dist
        return d

    def _cluster_detections(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Single-linkage cluster the detections whose pairwise geodesic centroid distance
        is within ``_DISTANCE_THRESHOLD_M``.

        Input:  (M, M) symmetric distance matrix in meters.
        Output: (M,) integer cluster labels (1-indexed, matching scipy's convention).

        Single-linkage is used so that transitive near-duplicates chain into the same
        cluster: if A~B and B~C then A, B, C are one cluster even if A and C are
        farther apart than the threshold.
        """
        m = dist_matrix.shape[0]
        if m == 0:
            return np.empty((0,), dtype=np.int64)
        if m == 1:
            return np.array([1], dtype=np.int64)

        condensed = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed, method="single")
        labels = fcluster(
            linkage_matrix, t=self._DISTANCE_THRESHOLD_M, criterion="distance"
        )
        return labels.astype(np.int64)

    def _average_clusters(
        self, bboxes: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized per-cluster mean of bounding box corners.

        Input:
        - bboxes: (M, 4, 2) all detected bboxes across the video.
        - labels: (M,) cluster label for each detection.

        Output: (N_unique, 4, 2) with one averaged bbox per cluster.
        """
        if bboxes.size == 0:
            return np.empty((0, 4, 2), dtype=np.float64)

        _, inverse = np.unique(labels, return_inverse=True)
        n_unique = int(inverse.max()) + 1
        sums = np.zeros((n_unique, 4, 2), dtype=np.float64)
        np.add.at(sums, inverse, bboxes)
        counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)
        return sums / counts[:, None, None]

    def _deduplicate(self, data: DeduplicationInput) -> np.ndarray:
        """
        Full cross-video deduplication pipeline.

        Returns a (N_unique, 4, 2) array of averaged (lat, lon) bounding boxes, one
        per unique physical object detected in the video.
        """
        bboxes = self._flatten_detections(data)
        if bboxes.size == 0:
            logger.info("Deduplicator received no detections.")
            return np.empty((0, 4, 2), dtype=np.float64)

        centroids = self._get_centroids(bboxes)
        dist_matrix = self._get_pairwise_distances(centroids)
        labels = self._cluster_detections(dist_matrix)
        deduplicated = self._average_clusters(bboxes, labels)
        logger.info(
            "Deduplicated %d detections into %d unique objects.",
            bboxes.shape[0],
            deduplicated.shape[0],
        )
        return deduplicated
