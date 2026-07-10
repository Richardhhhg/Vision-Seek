import os
import uuid

import cv2
import numpy as np
import torch
from sahi.postprocess.combine import NMSPostprocess
from sahi.prediction import ObjectPrediction

from detection.data import (
    DetectionFrame,
    DetectionModelOutput,
    PostprocessorOutput,
    PreprocessedVideo,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ANNOTATED_VIDEOS_DIR = os.path.join(_REPO_ROOT, "annotated_videos")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Postprocessor:
    """
    For post processing results of detection to resemble original but with the proper detection results.

    Attributes:
    - steps (list[str]): List of post processing steps to be applied to the video.
    - pre_to_post_map (dict[str, str]): Mapping for preprocessing function to the post processing function.
    - supported_steps (dict[str, callable]): Dictionary of supported post processing steps and their corresponding functions.

    Methods:
    - add_step(step: str): Adds a post processing step to the list of steps to be applied to the video.
    - postprocess_video(detected_video: DetectionModelOutput): Postprocesses the detected video according to the added steps and returns the postprocessed video. Also renders an annotated video at the original resolution when a source video path is available.
    - _bw_to_color(annotated_frames, preprocessed_video): Reverses the bw preprocessing step (no-op for bounding boxes).
    - _stitch(annotated_frames, preprocessed_video): Reverses the resize (tiling) preprocessing step by remapping tile-space bboxes back to the original frame and merging via NMS across tiles.
    - _annotate_source_video(annotated_frames, source_video_path): Draws the postprocessed bounding boxes onto the original source video and writes the annotated video to disk. Returns the output path.
    """
    def __init__(self):
        self.steps: list[str] = []
        self.pre_to_post_map = {
            "bw": "bw_to_color",
            "resize": "stitch",
        }
        self.supported_steps = {
            "bw_to_color": self._bw_to_color,
            "stitch": self._stitch,
        }

    def add_step(self, step: str) -> None:
        """
        Adds a post processing step to the list of steps to be applied to the video.
        """
        self.steps.append(step)

    def postprocess_video(self, detected_video: DetectionModelOutput) -> PostprocessorOutput:
        """
        Postprocesses the detected video by undoing each preprocessing step in reverse order,
        based on the step_names recorded in `detected_video.preprocessed_video`. If a
        `source_video_path` is available on the preprocessed video, an annotated video at the
        original resolution is rendered and its path is used as the output_video_path.
        """
        preprocessed_video = detected_video.preprocessed_video
        annotated_frames = list(detected_video.annotated_frames)

        for pre_step in reversed(preprocessed_video.step_names):
            post_step = self.pre_to_post_map.get(pre_step)
            if post_step is None:
                continue
            post_fn = self.supported_steps.get(post_step)
            if post_fn is None:
                continue
            annotated_frames = post_fn(annotated_frames, preprocessed_video)

        source_video_path = preprocessed_video.source_video_path
        output_video_path = self._annotate_source_video(annotated_frames, source_video_path)

        return PostprocessorOutput(
            output_video_path=output_video_path,
            annotated_frames=annotated_frames,
        )

    def _bw_to_color(
        self,
        annotated_frames: list[DetectionFrame],
        preprocessed_video: PreprocessedVideo,
    ) -> list[DetectionFrame]:
        """
        The bounding box metadata is independent of the color of the source frame, so this is a
        pass-through for the DetectionFrame objects.
        """
        return annotated_frames

    def _stitch(
        self,
        annotated_frames: list[DetectionFrame],
        preprocessed_video: PreprocessedVideo,
    ) -> list[DetectionFrame]:
        """
        Combines per-tile detections back into per-source-frame detections by offsetting bbox
        coordinates according to each tile's position in the original frame and applying NMS
        across overlapping tiles.
        """
        num_tiles = preprocessed_video.num_tiles
        tile_positions = preprocessed_video.tile_positions

        if num_tiles is None or num_tiles <= 1 or not tile_positions:
            return annotated_frames

        grouped: dict[int, list[DetectionFrame]] = {}
        for frame in annotated_frames:
            source_idx = frame.frame_number // num_tiles
            grouped.setdefault(source_idx, []).append(frame)

        nms = NMSPostprocess(match_threshold=0.5, match_metric="IOU", class_agnostic=True)
        merged_frames: list[DetectionFrame] = []

        for source_idx in sorted(grouped.keys()):
            tile_frames = grouped[source_idx]
            predictions: list[ObjectPrediction] = []
            for tile_frame in tile_frames:
                tile_idx = tile_frame.frame_number % num_tiles
                if tile_idx >= len(tile_positions):
                    continue
                offset_x, offset_y = tile_positions[tile_idx]
                xyxy = _to_numpy(tile_frame.xyxy)
                conf = _to_numpy(tile_frame.conf)
                cls = _to_numpy(tile_frame.cls)
                for k in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[k]
                    predictions.append(
                        ObjectPrediction(
                            bbox=[
                                float(x1) + offset_x,
                                float(y1) + offset_y,
                                float(x2) + offset_x,
                                float(y2) + offset_y,
                            ],
                            score=float(conf[k]),
                            category_id=int(cls[k]),
                        )
                    )

            merged_preds = nms(predictions) if predictions else []

            if merged_preds:
                merged_xyxy = np.array(
                    [
                        [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
                        for p in merged_preds
                    ],
                    dtype=np.float32,
                )
                widths = merged_xyxy[:, 2] - merged_xyxy[:, 0]
                heights = merged_xyxy[:, 3] - merged_xyxy[:, 1]
                cx = (merged_xyxy[:, 0] + merged_xyxy[:, 2]) / 2.0
                cy = (merged_xyxy[:, 1] + merged_xyxy[:, 3]) / 2.0
                merged_xywh = np.stack([cx, cy, widths, heights], axis=1).astype(np.float32)
                merged_conf = np.array([p.score.value for p in merged_preds], dtype=np.float32)
                merged_cls = np.array([p.category.id for p in merged_preds], dtype=np.float32)
                merged_xyxyn = np.zeros_like(merged_xyxy)
            else:
                merged_xyxy = np.zeros((0, 4), dtype=np.float32)
                merged_xywh = np.zeros((0, 4), dtype=np.float32)
                merged_xyxyn = np.zeros((0, 4), dtype=np.float32)
                merged_conf = np.zeros((0,), dtype=np.float32)
                merged_cls = np.zeros((0,), dtype=np.float32)

            merged_frames.append(
                DetectionFrame(
                    frame_number=source_idx,
                    xyxy=merged_xyxy,
                    xywh=merged_xywh,
                    xyxyn=merged_xyxyn,
                    conf=merged_conf,
                    cls=merged_cls,
                )
            )

        return merged_frames

    def _annotate_source_video(
        self,
        annotated_frames: list[DetectionFrame],
        source_video_path: str,
    ) -> str:
        """
        Renders bounding boxes onto every frame of the source video at the original resolution
        and writes the annotated video to `annotated_videos/`. Returns the output path.
        """
        os.makedirs(_ANNOTATED_VIDEOS_DIR, exist_ok=True)
        output_path = os.path.join(
            _ANNOTATED_VIDEOS_DIR, f"detect_{uuid.uuid4().hex}.mp4"
        )

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to Open Video: {source_video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_lookup: dict[int, DetectionFrame] = {
            int(f.frame_number): f for f in annotated_frames
        }

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detection = frame_lookup.get(idx)
            if detection is not None:
                xyxy = _to_numpy(detection.xyxy)
                conf = _to_numpy(detection.conf)
                cls = _to_numpy(detection.cls)
                for k in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = map(int, xyxy[k])
                    label = f"{int(cls[k])} {float(conf[k]):.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            writer.write(frame)
            idx += 1

        cap.release()
        writer.release()
        return output_path
