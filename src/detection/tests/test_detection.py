# Tests for the detection module
# 1. Ensuring that with detection input data being passed in, the detection module returns the correct output data structure
# 2. Ensure behaviour is expected when passing invalid config
import pytest
from paths import TEST_CONFIG_FAKE_PATH, TEST_CONFIG_REAL_PATH, TEST_VIDEO_PATH

from detection.data import DetectionInput, DetectionOutput
from detection.detection import Detection


def test_detection_smoke():
    detect = Detection(config_path=TEST_CONFIG_REAL_PATH)
    sample_input = DetectionInput(video_path=TEST_VIDEO_PATH)
    detection_output = detect.run_detection(sample_input)
    assert isinstance(detection_output, DetectionOutput)
    assert detection_output.output_video_path is not None
    assert detection_output.annotated_frames is not None
    assert len(detection_output.annotated_frames) > 0

def test_detection_invalid_config():
    with pytest.raises(ValueError):
        detect = Detection(config_path=TEST_CONFIG_FAKE_PATH)
        detect.run_detection(DetectionInput(video_path=TEST_VIDEO_PATH))