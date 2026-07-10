# Tests to write:
# 1. Every different preprocessing step individually
# 1a. Black and white preprocessing -> preprocessed video object with frames in np.ndarray with 3 channels that are the same
# 1b. Resize preprocessing -> should return frames with the specified width and height # NOTE: this resize should be tiling rather then the yolo style of padding and such
# 2. >1 preprocessing step applied on the same video
# 3. No preprocessing steps -> this should just return the original video frames in a preprocessed video object
import numpy as np
import pytest
import torch

from detection.data import PreprocessedVideo
from detection.preprocessing.preprocessor import Preprocessor
from detection.tests.paths import TEST_VIDEO_PATH


def test_preprocessor_bw_gives_ndarray_3_same_channels():
    preprocessor = Preprocessor()
    preprocessor.add_step("bw")
    res = preprocessor.preprocess_video(TEST_VIDEO_PATH)
    assert isinstance(res, PreprocessedVideo)
    assert isinstance(res.frames, np.ndarray | torch.Tensor)
    assert res.frames.ndim == 4  # (num_frames, height, width, channels)
    assert np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 1]) and np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 2]) and np.all(res.frames[0, :, :, 1] == res.frames[0, :, :, 2])
    assert res.step_names == ["bw"]
    assert res.num_tiles is None
    assert res.tile_offset is None

def test_preprocessor_resize():
    preprocessor = Preprocessor()
    preprocessor.add_step("resize", width=640, height=480)
    res = preprocessor.preprocess_video(TEST_VIDEO_PATH)
    assert isinstance(res, PreprocessedVideo)
    assert isinstance(res.frames, np.ndarray | torch.Tensor)
    assert res.frames.ndim == 4  # (num_frames, height, width, channels)
    assert res.frames.shape[1] == 480
    assert res.frames.shape[2] == 640
    assert res.step_names == ["resize"]
    # NOTE: it might not be enough to just check if it is none, also have to check if the number of tiles is the 
    # expected amount given the input dimensions and the original dimensions of the test video
    # same applies for tile offset
    assert res.num_tiles is not None
    assert res.tile_offset is not None

def test_preprocessor_bw_then_resize_does_not_break_data():
    preprocessor = Preprocessor()
    preprocessor.add_step("bw")
    preprocessor.add_step("resize", width=640, height=480)
    res = preprocessor.preprocess_video(TEST_VIDEO_PATH)
    assert isinstance(res, PreprocessedVideo)
    assert isinstance(res.frames, np.ndarray | torch.Tensor)
    assert res.frames.ndim == 4  # (num_frames, height, width, channels)
    assert np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 1]) and np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 2]) and np.all(res.frames[0, :, :, 1] == res.frames[0, :, :, 2])
    assert res.frames.shape[1] == 480
    assert res.frames.shape[2] == 640
    assert res.step_names == ["bw", "resize"]
    assert res.num_tiles is not None
    assert res.tile_offset is not None

def test_preprocessor_resize_then_bw_does_not_break_data():
    preprocessor = Preprocessor()
    preprocessor.add_step("resize", width=640, height=480)
    preprocessor.add_step("bw")
    res = preprocessor.preprocess_video(TEST_VIDEO_PATH)
    assert isinstance(res, PreprocessedVideo)
    assert isinstance(res.frames, np.ndarray | torch.Tensor)
    assert res.frames.ndim == 4  # (num_frames, height, width, channels)
    assert np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 1]) and np.all(res.frames[0, :, :, 0] == res.frames[0, :, :, 2]) and np.all(res.frames[0, :, :, 1] == res.frames[0, :, :, 2])
    assert res.frames.shape[1] == 480 
    assert res.frames.shape[2] == 640
    assert res.step_names == ["resize", "bw"]
    assert res.num_tiles is not None
    assert res.tile_offset is not None

def test_preprocessor_no_steps_returns_valid_preprocessed_video():
    preprocessor = Preprocessor()
    res = preprocessor.preprocess_video(TEST_VIDEO_PATH)
    assert isinstance(res, PreprocessedVideo)
    assert isinstance(res.frames, np.ndarray | torch.Tensor)
    assert res.frames.ndim == 4  # (num_frames, height, width, channels)
    assert res.step_names == []
    assert res.num_tiles is None
    assert res.tile_offset is None