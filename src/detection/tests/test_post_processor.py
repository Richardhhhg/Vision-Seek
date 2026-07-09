# Tests for the post processor module
# 1. Testing each post processing step individually
# a. Test bw_to_color step
# b. Test stitch step
# 2. Testing the postprocess_video method with multiple steps
# 3. Testing the postprocess_video method with no steps in the preprocessed video
import cv2
import pytest

from detection.data import DetectionModelOutput, PostprocessorOutput, PreprocessedVideo
from detection.postprocessing.postprocessor import Postprocessor
from paths import TEST_VIDEO_PATH


@pytest.fixture(scope="module")
def setup():
    input_video = cv2.VideoCapture(TEST_VIDEO_PATH)
    frames = []
    while True:
        ret, frame = input_video.read()
        if not ret:
            break
        frames.append(frame)
    input_video.release()
    preprocessed_video = PreprocessedVideo(frames=frames, steps=[], num_tiles=None, tile_offset=None)
    detection_model_output = DetectionModelOutput(annotated_frames=[], output_video_path="fake_path", preprocessed_video=preprocessed_video)
    return detection_model_output

def test_bw_to_color_smoke(setup):
    detection_model_output = setup
    post_processor = Postprocessor()
    detection_model_output.preprocessed_video.steps.append("bw")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, PostprocessorOutput)

def test_stitch_smoke(setup):
    # NOTE: this might break if the original input video is a single tile but you try to stitch it as if it wasn't
    detection_model_output = setup
    post_processor = Postprocessor()
    detection_model_output.preprocessed_video.steps.append("resize")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, PostprocessorOutput)

def test_postprocess_video_bw_color_and_stitch_smoke(setup):
    detection_model_output = setup
    post_processor = Postprocessor()
    detection_model_output.preprocessed_video.steps.append("bw")
    detection_model_output.preprocessed_video.steps.append("resize")
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, PostprocessorOutput)

def test_postprocess_video_no_steps_smoke(setup):
    detection_model_output = setup
    post_processor = Postprocessor()
    res = post_processor.postprocess_video(detection_model_output)
    assert isinstance(res, PostprocessorOutput)