# Testing the detection model
# 1. pass basic video into the model, make sure it returns a detection output object. Ensure all fields in the detection frames are populated correctly. (may have none if no detections)
# 2. Run on cuda vs cpu, ensure that everything still is populated correctly and doesn't crash
import numpy as np
import pytest
import torch

from detection.data import DetectionFrame, PostprocessorOutput


@pytest.fixture(scope="module")
def setup():
    import cv2
    from detection.model.model_factory import ModelFactory

    from detection.data import PreprocessedVideo
    
    model = ModelFactory.get_model("yolo", model_path="src/detection/detection_model/model/test_model.pt")
    video = cv2.VideoCapture("data/test_data/test_video.mp4")
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    preprocessed_video = PreprocessedVideo(frames=frames, steps=[], num_tiles=None, tile_offset=None)
    yield model, preprocessed_video

def test_detection_model_can_run_detection(setup):
    model, preprocessed_video = setup
    detection_output = model.run_detection(preprocessed_video)
    assert isinstance(detection_output, PostprocessorOutput)
    assert detection_output.output_video_path is not None
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
        detection_output = model.run_detection(preprocessed_video, device="cuda")
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
    detection_output = model.run_detection(preprocessed_video, device="cpu")
    frame_zero = detection_output.annotated_frames[0]
    assert isinstance(frame_zero.xyxy, np.ndarray)
    assert isinstance(frame_zero.xywh, np.ndarray)
    assert isinstance(frame_zero.xyxyn, np.ndarray)
    assert isinstance(frame_zero.conf, np.ndarray)
    assert isinstance(frame_zero.cls, np.ndarray)