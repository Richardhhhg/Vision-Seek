# Testing the detection model
# 1. pass basic video into the model, make sure it returns a detection output object. Ensure all fields in the detection frames are populated correctly. (may have none if no detections)
# 2. Run on cuda vs cpu, ensure that everything still is populated correctly and doesn't crash
import numpy as np
import pytest
import torch

from detection.data import DetectionFrame, DetectionModelOutput
from detection.tests.paths import TEST_MODEL_PATH, TEST_VIDEO_PATH


@pytest.fixture(scope="function")
def setup():
    import cv2

    from detection.data import PreprocessedVideo
    from detection.detection_model.model_factory import ModelFactory
    
    model = ModelFactory.get_model("yolo", model_path=TEST_MODEL_PATH)
    video = cv2.VideoCapture(TEST_VIDEO_PATH)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    preprocessed_video = PreprocessedVideo(frames=np.array(frames), step_names=[], num_tiles=None, tile_offset=None, source_video_path=TEST_VIDEO_PATH)
    yield model, preprocessed_video

def test_detection_model_can_run_detection(setup):
    model, preprocessed_video = setup
    detection_output = model.detect(preprocessed_video)
    assert isinstance(detection_output, DetectionModelOutput)
    assert detection_output.annotated_frames is not None
    frame_zero = detection_output.annotated_frames[0]
    assert isinstance(frame_zero, DetectionFrame)
    assert isinstance(frame_zero.frame_number, int)
    assert isinstance(frame_zero.xyxy, np.ndarray | torch.Tensor)
    assert isinstance(frame_zero.xywh, np.ndarray | torch.Tensor)
    assert isinstance(frame_zero.xyxyn, np.ndarray | torch.Tensor)
    assert isinstance(frame_zero.conf, np.ndarray | torch.Tensor)
    assert isinstance(frame_zero.cls, np.ndarray | torch.Tensor)

def test_detection_model_cuda_gives_tensors(setup):
    model, preprocessed_video = setup
    if torch.cuda.is_available():
        detection_output = model.detect(preprocessed_video, device="cuda")
        frame_zero = detection_output.annotated_frames[0]
        assert isinstance(frame_zero.xyxy, torch.Tensor)
        assert isinstance(frame_zero.xywh, torch.Tensor)
        assert isinstance(frame_zero.xyxyn, torch.Tensor)
        assert isinstance(frame_zero.conf, torch.Tensor)
        assert isinstance(frame_zero.cls, torch.Tensor)
    else:
        pytest.skip("CUDA is not available, skipping: test_detection_model_cuda_gives_tensors")


def test_detection_model_cpu_gives_numpy(setup):
    model, preprocessed_video = setup
    detection_output = model.detect(preprocessed_video, device="cpu")
    frame_zero = detection_output.annotated_frames[0]
    assert isinstance(frame_zero.xyxy, np.ndarray)
    assert isinstance(frame_zero.xywh, np.ndarray)
    assert isinstance(frame_zero.xyxyn, np.ndarray)
    assert isinstance(frame_zero.conf, np.ndarray)
    assert isinstance(frame_zero.cls, np.ndarray)