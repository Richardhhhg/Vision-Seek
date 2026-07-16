import logging

import numpy as np
from pyproj import Geod
from scipy.cluster.hierarchy import fcluster, linkage

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
    - _MIN_DETECTION_TIME_S (float): Minimum time (seconds) an object must appear in
      the video for it to survive the ``_remove_one_offs`` pre-filter.
    - _FPS (int): Assumed video frame rate; used to translate
      ``_MIN_DETECTION_TIME_S`` into a minimum frame count.
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
    - _get_pairwise_distances (centroids) -> np.ndarray: Vectorized geodesic distances
      between all centroid pairs, returned as scipy's condensed 1D upper-triangle
      form (length ``M*(M-1)//2``). This is exactly what ``linkage`` expects, so we
      never materialize the (M, M) matrix.
    - _cluster_detections (condensed_distances, m) -> np.ndarray: Single-linkage
      cluster labels for each detection.
    - _average_clusters (bboxes, labels) -> np.ndarray: Per-cluster mean of bbox
      corners, returning a (N_unique, 4, 2) array.
    - _remove_one_offs (DeduplicationInput) -> DeduplicationInput: Pre-filter that
      drops clusters appearing in fewer frames than ``_MIN_DETECTION_TIME_S * _FPS``.
    - _deduplicate (DeduplicationInput) -> np.ndarray: End-to-end pipeline that ties
      the above together.
    """
    _DISTANCE_THRESHOLD_M = 1.0
    _MIN_DETECTION_TIME_S = 1.0
    _FPS = 30
    _GEOD = Geod(ellps="WGS84")

    def __init__(self):
        pass

    def invoke(self, input_data: DeduplicationInput) -> DeduplicationOutput:
        filtered = self._remove_one_offs(input_data)
        gps_coords = self._deduplicate(filtered)
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
        Vectorized pairwise geodesic distances (meters) in scipy's condensed
        upper-triangle form: a flat array of length ``m*(m-1)//2``.

        Returning the condensed form instead of the full (m, m) matrix roughly halves
        the pairwise-distance memory footprint and removes the extra ``squareform``
        copy inside clustering. ``scipy.cluster.hierarchy.linkage`` accepts this
        format directly.
        """
        # TODO: I believe this is O(m^2) in both memory and time. Use better algorithm or cap input.
        m = centroids.shape[0]
        if m < 2:
            return np.empty((0,), dtype=np.float64)

        idx_i, idx_j = np.triu_indices(m, k=1)
        lats1 = centroids[idx_i, 0]
        lons1 = centroids[idx_i, 1]
        lats2 = centroids[idx_j, 0]
        lons2 = centroids[idx_j, 1]
        _, _, dist = self._GEOD.inv(lons1, lats1, lons2, lats2)
        return dist

    def _cluster_detections(
        self, condensed_distances: np.ndarray, m: int
    ) -> np.ndarray:
        """
        Single-linkage cluster the detections whose pairwise geodesic centroid distance
        is within ``_DISTANCE_THRESHOLD_M``.

        Input:
        - condensed_distances: length ``m*(m-1)//2`` upper-triangle distances (meters).
        - m: number of detections that produced the condensed vector (needed to handle
          the trivial 0/1-detection cases before calling ``linkage``).

        Output: (m,) integer cluster labels (1-indexed, matching scipy's convention).

        Single-linkage is used so that transitive near-duplicates chain into the same
        cluster: if A~B and B~C then A, B, C are one cluster even if A and C are
        farther apart than the threshold.
        """
        if m == 0:
            return np.empty((0,), dtype=np.int64)
        if m == 1:
            return np.array([1], dtype=np.int64)

        linkage_matrix = linkage(condensed_distances, method="single")
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
        condensed = self._get_pairwise_distances(centroids)
        labels = self._cluster_detections(condensed, bboxes.shape[0])
        deduplicated = self._average_clusters(bboxes, labels)
        logger.info(
            "Deduplicated %d detections into %d unique objects.",
            bboxes.shape[0],
            deduplicated.shape[0],
        )
        return deduplicated
    
    def _remove_one_offs(self, data: DeduplicationInput) -> DeduplicationInput:
        """
        Remove detections that only appear in the video for less than
        ``_MIN_DETECTION_TIME_S`` seconds (at ``_FPS``).

        Because "how long does an object appear" requires knowing which
        per-frame detections are the same physical object, we run the same
        spatial single-linkage clustering used by ``_deduplicate``, count the
        number of distinct frames each cluster is present in, and drop every
        cluster whose frame count is below ``_MIN_DETECTION_TIME_S * _FPS``.
        Returns a new ``DeduplicationInput`` with the surviving detections
        preserved in their original per-frame ``(N_f, 4, 2)`` layout.
        """
        frames = data.gps_coords
        bboxes_parts: list[np.ndarray] = []
        frame_ids_parts: list[np.ndarray] = []
        for f_idx, f in enumerate(frames):
            arr = np.asarray(f, dtype=np.float64)
            if arr.size == 0:
                continue
            bboxes_parts.append(arr)
            frame_ids_parts.append(np.full(arr.shape[0], f_idx, dtype=np.int64))

        if not bboxes_parts:
            return data

        bboxes = np.concatenate(bboxes_parts, axis=0)
        frame_ids = np.concatenate(frame_ids_parts)

        centroids = self._get_centroids(bboxes)
        condensed = self._get_pairwise_distances(centroids)
        labels = self._cluster_detections(condensed, bboxes.shape[0])

        _, inverse = np.unique(labels, return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        # Count distinct frames per cluster, not raw detections: a stationary
        # object can appear multiple times in the same frame (e.g. from
        # overlapping SAHI tiles) and we don't want that to inflate its duration.
        frames_per_cluster = np.array(
            [np.unique(frame_ids[inverse == c]).size for c in range(n_clusters)],
            dtype=np.int64,
        )
        min_frames = int(self._MIN_DETECTION_TIME_S * self._FPS)
        keep_cluster = frames_per_cluster >= min_frames
        keep_mask = keep_cluster[inverse]

        kept_bboxes = bboxes[keep_mask]
        kept_frame_ids = frame_ids[keep_mask]

        new_gps_coords: list[np.ndarray] = []
        for f_idx in range(len(frames)):
            in_frame = kept_frame_ids == f_idx
            if in_frame.any():
                new_gps_coords.append(kept_bboxes[in_frame])
            else:
                new_gps_coords.append(np.empty((0, 4, 2), dtype=np.float64))

        logger.info(
            "One-off filter kept %d/%d clusters (min %d frames = %.2fs at %d fps); "
            "%d/%d detections survived.",
            int(keep_cluster.sum()),
            n_clusters,
            min_frames,
            self._MIN_DETECTION_TIME_S,
            self._FPS,
            int(keep_mask.sum()),
            bboxes.shape[0],
        )
        return DeduplicationInput(gps_coords=new_gps_coords)
