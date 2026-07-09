from functools import partial
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sahi.slicing import slice_image

from detection.data import PreprocessedVideo


class Preprocessor:
    """
    Class for preprocessing video for detection

    Attributes:
    - steps (list): List of preprocessing steps to be applied to the video
    - supported_steps(dict [str, callable]): Dictionary of supported preprocessing steps and their corresponding functions

    Methods: 
    - add_step(step: str): Adds a preprocessing step to the list of steps to be applied to the video
    - preprocess_video(video_path: str): Preprocesses the video at the given path according to the added steps and returns the preprocessed video
    - _make_bw(frames: np.ndarray): Converts all frames to black and white; returns a state update dict.
    - _resize(frames: np.ndarray, width: int, height: int): Tiles each frame into slices of the specified width and height; returns a state update dict (frames + tile metadata).

    Step-function contract:
    - Each step takes the current `frames` ndarray (shape (N, H, W, C)) as its first positional
      argument (plus any kwargs bound via `add_step`) and returns a dict of updates whose keys
      are `PreprocessedVideo` field names. `preprocess_video` merges these updates into the
      running state, so a step can transform the frames and optionally attach metadata (e.g.
      `num_tiles`, `tile_positions`) in one uniform interface with no per-step special cases.
    """
    def __init__(self, **kwargs):
        self.steps: list[partial] = []
        self.step_names: list[str] = []

        self.supported_steps = {
            "bw": self._make_bw,
            "resize": self._resize,
        }

    def add_step(self, step: str, *args, **kwargs) -> None:
        """
        Adds a preprocessing step to the list of steps to be applied to the video.
        """
        if step not in self.supported_steps:
            raise ValueError(f"Unsupported preprocessing step: {step}")
        self.step_names.append(step)
        self.steps.append(partial(self.supported_steps[step], *args, **kwargs))

    def preprocess_video(self, video_path: str) -> PreprocessedVideo:
        """
        Preprocesses the video at the given path according to the added steps and returns the preprocessed video.
        """
        cap = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        state: dict[str, Any] = {"frames": np.array(frames)}
        for step_fn in self.steps:
            state.update(step_fn(state["frames"]))

        return PreprocessedVideo(
            **state,
            step_names=list(self.step_names),
            source_video_path=video_path,
        )

    def _make_bw(self, frames: np.ndarray) -> dict[str, Any]:
        """
        Converts every frame to a 3-channel black and white image (all channels equal).

        Returns a state update dict with the new frames.
        """
        bw_frames = np.array(
            [
                cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                for f in frames
            ]
        )
        return {"frames": bw_frames}

    def _resize(
        self, frames: np.ndarray, width: int, height: int
    ) -> dict[str, Any]:
        """
        Tiles every frame into slices of the specified width and height using sahi.

        Returns a state update dict with:
        - frames: array of shape (num_source_frames * num_tiles, height, width, channels).
        - num_tiles: number of tiles produced per source frame.
        - tile_offset: (step_x, step_y) stride between adjacent tile origins.
        - tile_positions: (x, y) starting pixel of each tile in the original frame.
        """
        all_tiles: list[np.ndarray] = []
        positions: list[tuple[int, int]] | None = None

        for source_frame in frames:
            result = slice_image(
                image=Image.fromarray(source_frame),
                slice_height=height,
                slice_width=width,
                # TODO: this should be set by user in future
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                auto_slice_resolution=False,
                verbose=False,
            )
            for sliced in result.sliced_image_list:
                all_tiles.append(np.array(sliced.image))
            if positions is None:
                positions = [
                    (int(s.starting_pixel[0]), int(s.starting_pixel[1]))
                    for s in result.sliced_image_list
                ]

        positions = positions or []
        xs = sorted({p[0] for p in positions})
        ys = sorted({p[1] for p in positions})
        step_x = xs[1] - xs[0] if len(xs) > 1 else 0
        step_y = ys[1] - ys[0] if len(ys) > 1 else 0

        return {
            "frames": np.array(all_tiles),
            "num_tiles": len(positions),
            "tile_offset": (step_x, step_y),
            "tile_positions": positions,
        }
